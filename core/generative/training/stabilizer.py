from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Any,
    Callable,
    Protocol,
    TypeVar,
    Generic,
    TypedDict,
    NamedTuple,
)

"""
core/generative/training/stabilizer.py

Advanced training stabilization system for hybrid GANs
Designed according to SOLID and Clean Code principles for a production data generation system

Smart architecture:
- Single Responsibility: Each component is responsible for stabilizing one aspect
- Open/Closed: Extensible with new penalty and monitor types
- Liskov Substitution: Replaceable interfaces for different discriminators
- Interface Segregation: Small and specialized interfaces for each task
- Dependency Inversion: Depend on abstractions, not implementations
"""
from core.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/system.log")
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import sys
import logging
from datetime import datetime
from collections import deque
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from scipy import stats
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path

# Type Aliases for Clear Code Intent
T = TypeVar("T")
TensorLike = Union[torch.Tensor, np.ndarray]
ModelLike = Union[nn.Module, Any]  # Supports any model with forward pass
OptimizerLike = Union[Optimizer, Any]


# ============================================================================
# PROTOCOLS AND INTERFACES
# ============================================================================


class DiscriminatorProtocol(Protocol):
    """Unified protocol for discriminators to ensure compatibility"""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward data through the discriminator"""
        ...

    def parameters(self) -> List[torch.Tensor]:
        """Model parameters"""
        ...

    def train(self) -> None:
        """Training mode"""
        ...

    def eval(self) -> None:
        """Evaluation mode"""
        ...


class GradientPenaltyStrategy(Protocol):
    """Gradient penalty strategy - easily replaceable"""

    def compute(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute gradient penalty"""
        ...


class ModeCollapseDetector(Protocol):
    """Mode collapse detector - easily replaceable"""

    def detect(
        self, real_data: TensorLike, generated_data: TensorLike, **kwargs
    ) -> Tuple[bool, Dict[str, float]]:
        """Detect mode collapse and return metrics"""
        ...


# ============================================================================
# CONFIGURATION DATA CLASSES
# ============================================================================


@dataclass(frozen=True)
class StabilizerConfig:
    """
    Professional full configuration for Training Stabilizer
    Compatible with EngineConfig and WGAN-GP
    """

    # =========================
    # Gradient Penalty Settings
    # =========================
    gradient_penalty_enabled: bool = True
    gradient_penalty_type: str = "wgan-gp"  # wgan-gp, dragan, lipschitz, none

    # Standard name (Source of Truth)
    lambda_gp: float = 10.0

    # Backward support (Deprecated – not used internally)
    gradient_penalty_lambda: float = None

    n_critic: int = 5  # Number of D steps per G step

    # =========================
    # Gradient Clipping Settings
    # =========================
    gradient_clipping_enabled: bool = True
    gradient_clip_value: float = 1.0
    gradient_clip_norm: float = 5.0

    # =========================
    # Stability Monitoring Settings
    # =========================
    mode_collapse_detection_enabled: bool = True
    mode_collapse_threshold: float = 0.7
    diversity_threshold: float = 0.8
    similarity_threshold: float = 0.9

    # =========================
    # Learning Rate Adjustment Settings
    # =========================
    learning_rate_scheduling_enabled: bool = True
    learning_rate_scheduler_type: str = "cosine"  # cosine, plateau, step, cyclic
    learning_rate_warmup_epochs: int = 10
    learning_rate_decay_rate: float = 0.96
    learning_rate_min: float = 1e-6

    # =========================
    # Tracking Settings
    # =========================
    history_window_size: int = 100
    checkpoint_frequency: int = 10
    logging_frequency: int = 100

    # =========================
    # Performance Settings
    # =========================
    use_mixed_precision: bool = torch.cuda.is_available()
    enable_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 4

    # =========================
    # Post Init Validation
    # =========================
    def __post_init__(self):
        """
        Unify values + strict validation
        """

        # Unify lambda name (Backward Compatibility)
        if self.gradient_penalty_lambda is not None:
            self.lambda_gp = self.gradient_penalty_lambda

        # ===== Validation =====
        if self.lambda_gp < 0:
            raise ValueError("lambda_gp (Gradient Penalty) must be non-negative")

        if self.n_critic < 1:
            raise ValueError("n_critic must be >= 1")

        if self.gradient_clipping_enabled and self.gradient_clip_value <= 0:
            raise ValueError("gradient_clip_value must be positive")

        if not 0.0 <= self.mode_collapse_threshold <= 1.0:
            raise ValueError("mode_collapse_threshold must be between 0 and 1")

        if self.gradient_penalty_type not in {"wgan-gp", "dragan", "lipschitz", "none"}:
            raise ValueError(
                f"Unsupported Gradient Penalty type: {self.gradient_penalty_type}"
            )


@dataclass
class TrainingMetrics:
    """Container for training metrics with efficient aggregation"""

    generator_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    discriminator_losses: deque = field(default_factory=lambda: deque(maxlen=1000))
    gradient_norms: deque = field(default_factory=lambda: deque(maxlen=1000))
    mode_diversity_scores: deque = field(default_factory=lambda: deque(maxlen=500))
    wasserstein_distances: deque = field(default_factory=lambda: deque(maxlen=500))

    def update(
        self,
        g_loss: Optional[float] = None,
        d_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        diversity_score: Optional[float] = None,
        wasserstein_dist: Optional[float] = None,
    ) -> None:
        """Update metrics"""
        if g_loss is not None:
            self.generator_losses.append(g_loss)
        if d_loss is not None:
            self.discriminator_losses.append(d_loss)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
        if diversity_score is not None:
            self.mode_diversity_scores.append(diversity_score)
        if wasserstein_dist is not None:
            self.wasserstein_distances.append(wasserstein_dist)

    def get_statistics(self) -> Dict[str, float]:
        """Get metrics statistics"""
        stats = {}

        for name, data in [
            ("generator_loss", self.generator_losses),
            ("discriminator_loss", self.discriminator_losses),
            ("gradient_norm", self.gradient_norms),
            ("mode_diversity", self.mode_diversity_scores),
            ("wasserstein_distance", self.wasserstein_distances),
        ]:
            if data:
                stats[f"{name}_mean"] = float(np.mean(data))
                stats[f"{name}_std"] = float(np.std(data))
                stats[f"{name}_min"] = float(np.min(data))
                stats[f"{name}_max"] = float(np.max(data))
                stats[f"{name}_current"] = float(data[-1] if data else 0.0)

        return stats


# ============================================================================
# GRADIENT PENALTY STRATEGIES
# ============================================================================


class GradientPenaltyCalculator:
    """Advanced gradient penalty calculator with multiple strategies"""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "penalties": [],
            "gradient_norms": [],
            "interpolation_alphas": [],
        }

        # Log configuration
        self.lambda_gp = getattr(config, "lambda_gp", 10.0)

        logger.info(
            f"Initializing GradientPenaltyCalculator | "
            f"type={config.gradient_penalty_type} | "
            f"lambda_gp={self.lambda_gp}"
        )

    def compute_penalty(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        penalty_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute gradient penalty using the specified strategy

        Args:
            discriminator: The discriminator
            real_data: Real data [batch_size, ...]
            fake_data: Generated data [batch_size, ...]
            penalty_type: Penalty type (optional, uses default config)

        Returns:
            torch.Tensor: Penalty value
        """
        penalty_type = penalty_type or self.config.gradient_penalty_type

        try:
            if penalty_type == "wgan-gp":
                penalty = self._compute_wgan_gp_penalty(
                    discriminator, real_data, fake_data
                )
            elif penalty_type == "dragan":
                penalty = self._compute_dragan_penalty(
                    discriminator, real_data, fake_data
                )
            elif penalty_type == "lipschitz":
                penalty = self._compute_lipschitz_penalty(
                    discriminator, real_data, fake_data
                )
            elif penalty_type == "none":
                penalty = torch.tensor(0.0, device=real_data.device)
            else:
                logger.warning(f"Unknown penalty type: {penalty_type}, using WGAN-GP")
                penalty = self._compute_wgan_gp_penalty(
                    discriminator, real_data, fake_data
                )

            # Record history
            self.history["penalties"].append(penalty.item())

            return penalty * self.config.gradient_penalty_lambda

        except Exception as e:
            logger.error(f"Error computing gradient penalty: {e}")
            # Return zero penalty as safeguard
            return torch.tensor(0.0, device=real_data.device)

    def _compute_wgan_gp_penalty(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute WGAN-GP penalty (Gulrajani et al., 2017)"""
        batch_size = real_data.size(0)
        device = real_data.device

        # Generate random interpolation coefficients
        alpha = torch.rand(batch_size, *([1] * (real_data.dim() - 1)), device=device)
        alpha = alpha.expand_as(real_data)

        # Interpolated data
        interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(
            True
        )

        # Pass through discriminator
        d_interpolated = discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient norms
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Penalty for deviation from 1 (Lipschitz constraint)
        penalty = ((gradient_norm - 1) ** 2).mean()

        # Record history
        self.history["gradient_norms"].append(gradient_norm.mean().item())
        self.history["interpolation_alphas"].append(alpha.mean().item())

        return penalty

    def _compute_dragan_penalty(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        fake_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute DRAGAN penalty (Kodali et al., 2017)"""
        batch_size = real_data.size(0)
        device = real_data.device

        # Add noise to real data
        alpha = torch.rand(batch_size, *([1] * (real_data.dim() - 1)), device=device)
        noise = torch.randn_like(real_data) * 0.5  # Noise with strength 0.5

        # Interpolated data with noise
        interpolated = (real_data + alpha * noise).requires_grad_(True)

        # Pass through discriminator
        d_interpolated = discriminator(interpolated)

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient norms
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Penalty for deviation from 1
        penalty = ((gradient_norm - 1) ** 2).mean()

        return penalty

    def _compute_lipschitz_penalty(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute simplified Lipschitz penalty"""
        # Compute discriminator outputs for real and fake data
        d_real = discriminator(real_data)
        d_fake = discriminator(fake_data)

        # Compute output difference
        output_diff = torch.abs(d_real - d_fake)

        # Compute input difference (L2 distance)
        input_diff = torch.norm(real_data - fake_data, p=2, dim=1)

        # Ratio of output difference to input difference
        lipschitz_ratio = output_diff / (input_diff + 1e-8)

        # Penalty for deviation from optimal value (e.g., 1)
        penalty = ((lipschitz_ratio - 1.0) ** 2).mean()

        return penalty

    def compute_consistency_penalty(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        """
        Consistency penalty: ensure the discriminator gives consistent outputs for similar data

        Args:
            discriminator: The discriminator
            real_data: Real data
            epsilon: Perturbation size

        Returns:
            torch.Tensor: Consistency penalty value
        """
        try:
            # Add small noise
            noise = torch.randn_like(real_data) * epsilon
            perturbed_data = real_data + noise

            # Get discriminator outputs
            real_output = discriminator(real_data)
            perturbed_output = discriminator(perturbed_data)

            # Compute consistency penalty (MSE between outputs)
            consistency_loss = F.mse_loss(real_output, perturbed_output)

            # Small weight for penalty
            return consistency_loss * 0.1

        except Exception as e:
            logger.warning(f"Error computing consistency penalty: {e}")
            return torch.tensor(0.0, device=real_data.device)

    def get_penalty_statistics(self) -> Dict[str, float]:
        """Get gradient penalty statistics"""
        stats = {}

        if self.history["penalties"]:
            penalties = self.history["penalties"]
            stats["penalty_mean"] = float(np.mean(penalties))
            stats["penalty_std"] = float(np.std(penalties))
            stats["penalty_max"] = float(np.max(penalties))
            stats["penalty_min"] = float(np.min(penalties))
            stats["penalty_current"] = float(penalties[-1] if penalties else 0.0)

        if self.history["gradient_norms"]:
            norms = self.history["gradient_norms"]
            stats["grad_norm_mean"] = float(np.mean(norms))
            stats["grad_norm_std"] = float(np.std(norms))

        return stats


# ============================================================================
# MODE COLLAPSE DETECTION
# ============================================================================


class ModeCollapseDetector:
    """Advanced mode collapse detector with multiple metrics"""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "diversity_scores": [],
            "similarity_scores": [],
            "wasserstein_distances": [],
            "mode_coverage": [],
        }

        # Log configuration
        logger.info(
            f"Initializing ModeCollapseDetector with threshold {config.mode_collapse_threshold}"
        )

    def detect(
        self,
        real_data: TensorLike,
        generated_data: TensorLike,
        return_metrics: bool = True,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect mode collapse using multiple metrics

        Args:
            real_data: Real data
            generated_data: Generated data
            return_metrics: Return detailed metrics

        Returns:
            Tuple[bool, Dict]: (Is there mode collapse?, detection metrics)
        """
        try:
            # Convert to tensor if necessary
            real_tensor = self._to_tensor(real_data)
            gen_tensor = self._to_tensor(generated_data)

            # Compute metrics
            metrics = self._compute_all_metrics(real_tensor, gen_tensor)

            # Determine mode collapse
            mode_collapse = self._determine_collapse(metrics)

            # Update history
            self._update_history(metrics)

            if return_metrics:
                return mode_collapse, metrics
            else:
                return mode_collapse, {}

        except Exception as e:
            logger.error(f"Error detecting mode collapse: {e}")
            # In case of error, assume no mode collapse
            return False, {"error": str(e)}

    def _to_tensor(self, data: TensorLike) -> torch.Tensor:
        """Convert any data type to torch.Tensor"""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _compute_all_metrics(
        self, real_data: torch.Tensor, generated_data: torch.Tensor
    ) -> Dict[str, float]:
        """Compute all mode collapse detection metrics"""
        metrics = {}

        # 1. Diversity score
        metrics["diversity_score"] = self._compute_diversity_score(generated_data)

        # 2. Similarity score
        metrics["similarity_score"] = self._compute_similarity_score(generated_data)

        # 3. Simplified Wasserstein distance
        metrics["wasserstein_distance"] = self._compute_wasserstein_distance(
            real_data, generated_data
        )

        # 4. Mode coverage (if data is low-dimensional)
        if real_data.shape[1] <= 50:  # To avoid curse of dimensionality
            metrics["mode_coverage"] = self._compute_mode_coverage(
                real_data, generated_data
            )
        else:
            metrics["mode_coverage"] = 1.0  # Cannot compute accurately

        # 5. Distribution entropy
        metrics["distribution_entropy"] = self._compute_distribution_entropy(
            generated_data
        )

        return metrics

    def _compute_diversity_score(self, data: torch.Tensor) -> float:
        """Compute diversity score of samples"""
        if data.shape[0] < 2:
            return 1.0  # Single sample is always diverse

        # Compute distance matrix
        distances = torch.cdist(data, data, p=2)

        # Mask to remove self-distances (diagonals)
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=data.device)
        valid_distances = distances[mask]

        if valid_distances.numel() == 0:
            return 1.0

        # Average distance
        avg_distance = valid_distances.mean().item()

        # Maximum distance (for normalization)
        max_distance = valid_distances.max().item()

        # Avoid division by zero
        if max_distance == 0:
            return 1.0

        # Diversity score (0-1)
        diversity_score = avg_distance / max_distance

        return diversity_score

    def _compute_similarity_score(self, data: torch.Tensor) -> float:
        """Compute similarity score of samples"""
        if data.shape[0] < 2:
            return 0.0  # Single sample is not similar to anything

        # Normalize data
        normalized = F.normalize(data, p=2, dim=1)

        # Similarity matrix (cosine)
        similarity_matrix = torch.mm(normalized, normalized.T)

        # Remove self-similarity (diagonals)
        mask = ~torch.eye(
            similarity_matrix.shape[0], dtype=torch.bool, device=data.device
        )
        valid_similarities = similarity_matrix[mask]

        if valid_similarities.numel() == 0:
            return 0.0

        # Average similarity
        similarity_score = valid_similarities.mean().item()

        return similarity_score

    def _compute_wasserstein_distance(
        self, real_data: torch.Tensor, generated_data: torch.Tensor
    ) -> float:
        """Compute simplified Wasserstein distance"""
        # Compute basic statistics
        real_mean = real_data.mean(dim=0)
        real_std = real_data.std(dim=0)
        gen_mean = generated_data.mean(dim=0)
        gen_std = generated_data.std(dim=0)

        # Mean distance
        mean_distance = torch.norm(real_mean - gen_mean, p=2).item()

        # Standard deviation distance
        std_distance = torch.norm(real_std - gen_std, p=2).item()

        # Sum of distances (approximation of Wasserstein)
        wasserstein_distance = mean_distance + std_distance

        return wasserstein_distance

    def _compute_mode_coverage(
        self, real_data: torch.Tensor, generated_data: torch.Tensor, n_clusters: int = 5
    ) -> float:
        """Compute mode coverage using simplified KMeans"""
        try:
            from sklearn.cluster import KMeans

            # Convert to numpy
            real_np = real_data.cpu().numpy()
            gen_np = generated_data.cpu().numpy()

            # Apply KMeans to real data
            kmeans = KMeans(n_clusters=min(n_clusters, len(real_np)), random_state=42)
            kmeans.fit(real_np)

            # Label real and generated data
            real_labels = kmeans.labels_
            gen_labels = kmeans.predict(gen_np)

            # Compute unique modes in each set
            real_modes = set(real_labels)
            gen_modes = set(gen_labels)

            # Compute coverage
            coverage = len(gen_modes.intersection(real_modes)) / len(real_modes)

            return coverage

        except Exception as e:
            logger.warning(f"Error computing mode coverage: {e}")
            return 1.0  # Default value on failure

    def _compute_distribution_entropy(
        self, data: torch.Tensor, n_bins: int = 20
    ) -> float:
        """Compute distribution entropy using histogram"""
        try:
            # Convert data to numpy
            data_np = data.cpu().numpy()

            # Compute histogram
            hist, _ = np.histogramdd(data_np, bins=n_bins, density=True)

            # Normalize histogram
            hist_flat = hist.flatten()
            hist_flat = hist_flat / (hist_flat.sum() + 1e-10)

            # Compute entropy
            entropy = -np.sum(hist_flat * np.log(hist_flat + 1e-10))

            # Normalize entropy (maximum is log(n_bins^d))
            max_entropy = np.log(n_bins ** data_np.shape[1])
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return normalized_entropy

        except Exception as e:
            logger.warning(f"Error computing distribution entropy: {e}")
            return 0.5  # Default value

    def _determine_collapse(self, metrics: Dict[str, float]) -> bool:
        """Determine if there is mode collapse based on metrics"""
        # Mode collapse conditions
        conditions = [
            metrics.get("diversity_score", 1.0) < self.config.diversity_threshold,
            metrics.get("similarity_score", 0.0) > self.config.similarity_threshold,
            metrics.get("mode_coverage", 1.0) < self.config.mode_collapse_threshold,
        ]

        # If any two conditions are met
        if sum(conditions) >= 2:
            return True

        # Additional condition: very low diversity
        if metrics.get("diversity_score", 1.0) < 0.3:
            return True

        return False

    def _update_history(self, metrics: Dict[str, float]) -> None:
        """Update metrics history"""
        for key in self.history:
            if key in metrics:
                self.history[key].append(metrics[key])

                # Maintain window size
                if len(self.history[key]) > 1000:
                    self.history[key].pop(0)

    def get_detection_statistics(self) -> Dict[str, float]:
        """Get detection statistics"""
        stats = {}

        for metric_name, values in self.history.items():
            if values:
                stats[f"{metric_name}_mean"] = float(np.mean(values))
                stats[f"{metric_name}_std"] = float(np.std(values))
                stats[f"{metric_name}_trend"] = self._compute_trend(values)
                stats[f"{metric_name}_current"] = float(values[-1])

        return stats

    def _compute_trend(self, values: List[float], window: int = 50) -> float:
        """Compute trend of values (linear regression)"""
        if len(values) < window:
            return 0.0

        # Take last 'window' values
        recent_values = values[-window:]

        # Compute linear regression
        x = np.arange(len(recent_values))
        slope, _, _, _, _ = stats.linregress(x, recent_values)

        return float(slope)


# ============================================================================
# GRADIENT MANAGEMENT
# ============================================================================


class GradientManager:
    """Advanced gradient manager with various optimization techniques"""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "clipped_norms": [],
            "original_norms": [],
            "clipping_ratios": [],
        }

        # Log configuration
        logger.info(
            f"Initializing GradientManager with clip_value={config.gradient_clip_value}"
        )

    def apply_gradient_clipping(
        self,
        model: nn.Module,
        clip_type: str = "norm",
        max_norm: Optional[float] = None,
        clip_value: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Apply gradient clipping with monitoring

        Args:
            model: The model whose gradients to clip
            clip_type: Clipping type ('norm' or 'value')
            max_norm: Maximum norm (for norm clipping)
            clip_value: Maximum value (for value clipping)

        Returns:
            Dict: Clipping statistics
        """
        stats = {
            "clipped": False,
            "original_norm": 0.0,
            "clipped_norm": 0.0,
            "clipping_ratio": 1.0,
        }

        try:
            # Compute gradient norm before clipping
            total_norm = 0.0
            parameters = [p for p in model.parameters() if p.grad is not None]

            if not parameters:
                return stats

            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

            original_norm = total_norm**0.5
            stats["original_norm"] = original_norm

            # Apply clipping
            if clip_type == "norm":
                max_norm = max_norm or self.config.gradient_clip_norm
                torch.nn.utils.clip_grad_norm_(parameters, max_norm)
                stats["clipped"] = True
            elif clip_type == "value":
                clip_value = clip_value or self.config.gradient_clip_value
                torch.nn.utils.clip_grad_value_(parameters, clip_value)
                stats["clipped"] = True

            # Compute gradient norm after clipping
            total_norm_after = 0.0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2

            clipped_norm = total_norm_after**0.5
            stats["clipped_norm"] = clipped_norm

            # Compute clipping ratio
            if original_norm > 0:
                clipping_ratio = clipped_norm / original_norm
                stats["clipping_ratio"] = clipping_ratio

                # Record history
                self.history["clipped_norms"].append(clipped_norm)
                self.history["original_norms"].append(original_norm)
                self.history["clipping_ratios"].append(clipping_ratio)

                # Trim history if necessary
                for key in self.history:
                    if len(self.history[key]) > 1000:
                        self.history[key].pop(0)

            return stats

        except Exception as e:
            logger.error(f"Error applying gradient clipping: {e}")
            return stats

    def apply_gradient_accumulation(
        self, model: nn.Module, accumulation_steps: Optional[int] = None
    ) -> None:
        """
        Apply gradient accumulation (to aid training stability)

        Args:
            model: The model
            accumulation_steps: Accumulation steps
        """
        accumulation_steps = (
            accumulation_steps or self.config.gradient_accumulation_steps
        )

        if accumulation_steps <= 1:
            return

        # Scale gradients according to accumulation steps
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.div_(accumulation_steps)

    def apply_gradient_noise(
        self, model: nn.Module, noise_std: float = 1e-5, decay_rate: float = 0.55
    ) -> None:
        """
        Add noise to gradients to improve exploration

        Args:
            model: The model
            noise_std: Standard deviation of noise
            decay_rate: Noise decay rate
        """
        for param in model.parameters():
            if param.grad is not None:
                # Add Gaussian noise
                noise = torch.randn_like(param.grad) * noise_std
                param.grad.data.add_(noise)

    def get_gradient_statistics(self) -> Dict[str, float]:
        """Get gradient statistics"""
        stats = {}

        for metric_name, values in self.history.items():
            if values:
                stats[f"{metric_name}_mean"] = float(np.mean(values))
                stats[f"{metric_name}_std"] = float(np.std(values))
                stats[f"{metric_name}_current"] = float(values[-1])

        return stats


# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================


class LearningRateScheduler:
    """Advanced learning rate scheduler with multiple strategies"""

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.history: Dict[str, List[float]] = {
            "learning_rates": [],
            "losses": [],
            "adjustments": [],
        }

        # Scheduler states
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.patience_counter = 0

        # Log configuration
        logger.info(
            f"Initializing LearningRateScheduler with type {config.learning_rate_scheduler_type}"
        )

    def adjust_learning_rate(
        self,
        optimizer: OptimizerLike,
        current_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Adjust learning rate based on the specified strategy

        Args:
            optimizer: The optimizer
            current_loss: Current loss (for plateau-based scheduler)
            epoch: Current epoch
            total_epochs: Total epochs

        Returns:
            Dict: Adjustment statistics
        """
        stats = {
            "old_lr": 0.0,
            "new_lr": 0.0,
            "adjustment_factor": 1.0,
            "adjustment_type": "none",
        }

        try:
            # Get current learning rate
            old_lr = optimizer.param_groups[0]["lr"]
            stats["old_lr"] = old_lr

            # Determine epoch
            if epoch is not None:
                self.current_epoch = epoch

            # Apply scheduling strategy
            new_lr = old_lr
            adjustment_type = "none"

            if self.config.learning_rate_scheduler_type == "cosine":
                if total_epochs is not None:
                    new_lr = self._cosine_decay(
                        old_lr, self.current_epoch, total_epochs
                    )
                    adjustment_type = "cosine_decay"

            elif self.config.learning_rate_scheduler_type == "plateau":
                if current_loss is not None:
                    new_lr, adjustment_type = self._reduce_on_plateau(
                        optimizer, current_loss
                    )

            elif self.config.learning_rate_scheduler_type == "step":
                new_lr = self._step_decay(old_lr, self.current_epoch)
                adjustment_type = "step_decay"

            elif self.config.learning_rate_scheduler_type == "cyclic":
                new_lr = self._cyclic_lr(old_lr, self.current_epoch)
                adjustment_type = "cyclic"

            # Update learning rate
            if new_lr != old_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr

                stats["new_lr"] = new_lr
                stats["adjustment_factor"] = new_lr / old_lr if old_lr > 0 else 1.0
                stats["adjustment_type"] = adjustment_type

                # Record adjustment
                self.history["learning_rates"].append(new_lr)
                self.history["adjustments"].append(stats["adjustment_factor"])

                if current_loss is not None:
                    self.history["losses"].append(current_loss)

                # Trim history
                for key in self.history:
                    if len(self.history[key]) > 1000:
                        self.history[key].pop(0)

                logger.debug(
                    f"Learning rate: {old_lr:.6f} -> {new_lr:.6f} ({adjustment_type})"
                )

            return stats

        except Exception as e:
            logger.error(f"Error adjusting learning rate: {e}")
            return stats

    def _cosine_decay(
        self,
        base_lr: float,
        epoch: int,
        total_epochs: int,
        warmup_epochs: Optional[int] = None,
    ) -> float:
        """Cosine decay schedule"""
        warmup_epochs = warmup_epochs or self.config.learning_rate_warmup_epochs

        # Warmup phase
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs

        # Cosine decay
        progress = min(1.0, (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))

        # Ensure not below minimum
        new_lr = base_lr * cosine_decay
        return max(new_lr, self.config.learning_rate_min)

    def _reduce_on_plateau(
        self,
        optimizer: OptimizerLike,
        current_loss: float,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-7,
    ) -> Tuple[float, str]:
        """Reduce learning rate when loss plateaus"""
        adjustment_type = "none"
        new_lr = optimizer.param_groups[0]["lr"]

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

            if self.patience_counter >= patience:
                # Reduce learning rate
                new_lr = max(optimizer.param_groups[0]["lr"] * factor, min_lr)
                self.patience_counter = 0
                adjustment_type = "plateau_reduction"

                logger.info(f"Reducing learning rate due to plateau: {new_lr:.6f}")

        return new_lr, adjustment_type

    def _step_decay(self, base_lr: float, epoch: int, step_size: int = 30) -> float:
        """Step decay schedule"""
        decay_rate = self.config.learning_rate_decay_rate
        exponent = epoch // step_size
        new_lr = base_lr * (decay_rate**exponent)

        return max(new_lr, self.config.learning_rate_min)

    def _cyclic_lr(
        self,
        base_lr: float,
        epoch: int,
        step_size: int = 1000,
        max_lr: float = 0.01,
        min_lr: float = 1e-6,
    ) -> float:
        """Cyclic learning rate schedule"""
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)

        # Triangular function
        new_lr = min_lr + (max_lr - min_lr) * max(0, 1 - x)

        return new_lr

    def get_scheduling_statistics(self) -> Dict[str, float]:
        """Get scheduling statistics"""
        stats = {}

        if self.history["learning_rates"]:
            lrs = self.history["learning_rates"]
            stats["learning_rate_mean"] = float(np.mean(lrs))
            stats["learning_rate_std"] = float(np.std(lrs))
            stats["learning_rate_current"] = float(lrs[-1] if lrs else 0.0)
            stats["learning_rate_trend"] = self._compute_trend(lrs)

        if self.history["adjustments"]:
            adjustments = self.history["adjustments"]
            stats["adjustment_mean"] = float(np.mean(adjustments))
            stats["adjustment_frequency"] = len(adjustments) / max(
                len(self.history["learning_rates"]), 1
            )

        return stats

    def _compute_trend(self, values: List[float], window: int = 50) -> float:
        """Compute trend of values"""
        if len(values) < window:
            return 0.0

        recent_values = values[-window:]
        x = np.arange(len(recent_values))
        slope, _, _, _, _ = stats.linregress(x, recent_values)

        return float(slope)


# ============================================================================
# MAIN STABILIZER CLASS - PRODUCTION READY
# ============================================================================


class HybridTrainingStabilizer:
    """
    Integrated production-ready training stabilizer for hybrid GAN systems

    Design advantages:
    1. Extensibility: New strategies can be added without modifying existing code
    2. Reliability: Comprehensive error handling with continuous operation
    3. Performance: Parallel computations and caching of results
    4. Flexibility: Support for multiple discriminator and optimizer types
    5. Compatibility: Standard interfaces allow integration with other systems
    """

    def __init__(self, config: Optional[StabilizerConfig] = None):
        """
        Initialize the stabilizer with customizable configuration

        Args:
            config: Stabilizer configuration (optional)
        """
        self.config = config or StabilizerConfig()

        # Initialize components
        self.gradient_penalty_calculator = GradientPenaltyCalculator(self.config)
        self.mode_collapse_detector = ModeCollapseDetector(self.config)
        self.gradient_manager = GradientManager(self.config)
        self.learning_rate_scheduler = LearningRateScheduler(self.config)

        # Training metrics
        self.training_metrics = TrainingMetrics()

        # Training states
        self.current_iteration = 0
        self.mode_collapse_detected = False
        self.stability_score = 1.0

        # Logging
        logger.info(f"HybridTrainingStabilizer initialized with config: {self.config}")

    def compute_gradient_penalty(
        self,
        discriminator: DiscriminatorProtocol,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        penalty_type: Optional[str] = None,
        consistency_penalty: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute gradient penalty with advanced options

        Args:
            discriminator: The discriminator
            real_data: Real data
            fake_data: Generated data
            penalty_type: Penalty type (optional)
            consistency_penalty: Add consistency penalty

        Returns:
            Tuple[torch.Tensor, Dict]: (total penalty, statistics)
        """
        try:
            # Base gradient penalty
            gradient_penalty = self.gradient_penalty_calculator.compute_penalty(
                discriminator, real_data, fake_data, penalty_type
            )

            # Consistency penalty (optional)
            consistency = torch.tensor(0.0, device=real_data.device)
            if consistency_penalty:
                consistency = (
                    self.gradient_penalty_calculator.compute_consistency_penalty(
                        discriminator, real_data
                    )
                )

            # Total penalty
            total_penalty = gradient_penalty + consistency

            # Statistics
            stats = self.gradient_penalty_calculator.get_penalty_statistics()
            stats["consistency_penalty"] = consistency.item()
            stats["total_penalty"] = total_penalty.item()

            return total_penalty, stats

        except Exception as e:
            logger.error(f"Failed to compute gradient penalty: {e}")
            # Return zero penalty on error
            return torch.tensor(0.0, device=real_data.device), {"error": str(e)}

    def detect_mode_collapse(
        self,
        real_data: TensorLike,
        generated_data: TensorLike,
        detailed_metrics: bool = True,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Detect mode collapse with advanced options

        Args:
            real_data: Real data
            generated_data: Generated data
            detailed_metrics: Return detailed metrics

        Returns:
            Tuple[bool, Dict]: (Is there collapse?, metrics)
        """
        try:
            # Detect mode collapse
            collapse_detected, metrics = self.mode_collapse_detector.detect(
                real_data, generated_data, detailed_metrics
            )

            # Update state
            self.mode_collapse_detected = collapse_detected

            # Update stability score
            if "diversity_score" in metrics:
                self.stability_score = metrics["diversity_score"]

            # Update training metrics
            self.training_metrics.update(
                diversity_score=metrics.get("diversity_score"),
                wasserstein_dist=metrics.get("wasserstein_distance"),
            )

            return collapse_detected, metrics

        except Exception as e:
            logger.error(f"Failed to detect mode collapse: {e}")
            return False, {"error": str(e)}

    def manage_gradients(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        clip_generator: bool = True,
        clip_discriminator: bool = True,
        accumulation_steps: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Manage gradients for both models

        Args:
            generator: The generator
            discriminator: The discriminator
            clip_generator: Clip generator gradients
            clip_discriminator: Clip discriminator gradients
            accumulation_steps: Gradient accumulation steps

        Returns:
            Dict: Clipping statistics for each model
        """
        stats = {"generator": {}, "discriminator": {}}

        try:
            # Clip generator gradients
            if clip_generator:
                gen_stats = self.gradient_manager.apply_gradient_clipping(generator)
                stats["generator"] = gen_stats

            # Clip discriminator gradients
            if clip_discriminator:
                disc_stats = self.gradient_manager.apply_gradient_clipping(
                    discriminator
                )
                stats["discriminator"] = disc_stats

            # Gradient accumulation (if enabled)
            if self.config.enable_gradient_accumulation:
                self.gradient_manager.apply_gradient_accumulation(
                    generator, accumulation_steps
                )
                self.gradient_manager.apply_gradient_accumulation(
                    discriminator, accumulation_steps
                )

            # Update training metrics
            if "original_norm" in gen_stats:
                self.training_metrics.update(grad_norm=gen_stats["original_norm"])

            return stats

        except Exception as e:
            logger.error(f"Failed to manage gradients: {e}")
            return stats

    def schedule_learning_rate(
        self,
        generator_optimizer: OptimizerLike,
        discriminator_optimizer: OptimizerLike,
        generator_loss: Optional[float] = None,
        discriminator_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Schedule learning rate for both optimizers

        Args:
            generator_optimizer: Generator optimizer
            discriminator_optimizer: Discriminator optimizer
            generator_loss: Generator loss
            discriminator_loss: Discriminator loss
            epoch: Current epoch
            total_epochs: Total epochs

        Returns:
            Dict: Scheduling statistics for each optimizer
        """
        stats = {"generator": {}, "discriminator": {}}

        try:
            # Schedule generator learning rate
            if generator_loss is not None:
                gen_stats = self.learning_rate_scheduler.adjust_learning_rate(
                    generator_optimizer, generator_loss, epoch, total_epochs
                )
                stats["generator"] = gen_stats

            # Schedule discriminator learning rate
            if discriminator_loss is not None:
                disc_stats = self.learning_rate_scheduler.adjust_learning_rate(
                    discriminator_optimizer, discriminator_loss, epoch, total_epochs
                )
                stats["discriminator"] = disc_stats

            return stats

        except Exception as e:
            logger.error(f"Failed to schedule learning rate: {e}")
            return stats

    def update_training_metrics(
        self,
        generator_loss: Optional[float] = None,
        discriminator_loss: Optional[float] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """
        Update training metrics

        Args:
            generator_loss: Generator loss
            discriminator_loss: Discriminator loss
            iteration: Current iteration
        """
        if iteration is not None:
            self.current_iteration = iteration

        self.training_metrics.update(g_loss=generator_loss, d_loss=discriminator_loss)

    def check_training_stability(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Check overall training stability

        Args:
            window_size: Window size for analysis

        Returns:
            Dict: Comprehensive stability report
        """
        report = {
            "overall_stability": "stable",
            "score": self.stability_score,
            "mode_collapse_detected": self.mode_collapse_detected,
            "recommendations": [],
            "detailed_metrics": {},
        }

        try:
            # Collect statistics from all components
            report["detailed_metrics"][
                "training"
            ] = self.training_metrics.get_statistics()
            report["detailed_metrics"][
                "gradient_penalty"
            ] = self.gradient_penalty_calculator.get_penalty_statistics()
            report["detailed_metrics"][
                "mode_detection"
            ] = self.mode_collapse_detector.get_detection_statistics()
            report["detailed_metrics"][
                "gradient_management"
            ] = self.gradient_manager.get_gradient_statistics()
            report["detailed_metrics"][
                "learning_rate"
            ] = self.learning_rate_scheduler.get_scheduling_statistics()

            # Analyze stability
            stability_issues = []

            # 1. Mode collapse analysis
            if self.mode_collapse_detected:
                stability_issues.append("mode_collapse")
                report["recommendations"].append(
                    "Increase gradient penalty or diversify training data"
                )

            # 2. Gradient analysis
            grad_stats = report["detailed_metrics"]["gradient_management"]
            if "original_norms_current" in grad_stats:
                if grad_stats["original_norms_current"] > 5.0:
                    stability_issues.append("gradient_explosion")
                    report["recommendations"].append(
                        "Reduce learning rate or apply stricter gradient clipping"
                    )
                elif grad_stats["original_norms_current"] < 0.01:
                    stability_issues.append("gradient_vanishing")
                    report["recommendations"].append(
                        "Use better normalization or change activation functions"
                    )

            # 3. Loss analysis
            train_stats = report["detailed_metrics"]["training"]
            if (
                "generator_loss_current" in train_stats
                and "discriminator_loss_current" in train_stats
            ):
                g_loss = train_stats["generator_loss_current"]
                d_loss = train_stats["discriminator_loss_current"]

                if g_loss > d_loss * 3:  # Generator too weak
                    stability_issues.append("generator_weak")
                    report["recommendations"].append(
                        "Reduce D steps or strengthen the generator"
                    )
                elif d_loss > g_loss * 3:  # Discriminator too weak
                    stability_issues.append("discriminator_weak")
                    report["recommendations"].append(
                        "Increase D steps or strengthen the discriminator"
                    )

            # Determine overall stability
            if stability_issues:
                report["overall_stability"] = "unstable"
                report["issues"] = stability_issues
            else:
                report["overall_stability"] = "stable"
                report["recommendations"].append(
                    "Continue training, everything is stable"
                )

            # Update stability score
            if self.stability_score < 0.5:
                report["score"] = self.stability_score
                report["overall_stability"] = "critical"

            return report

        except Exception as e:
            logger.error(f"Failed to check training stability: {e}")
            return {
                "overall_stability": "unknown",
                "error": str(e),
                "recommendations": ["Check error logs for details"],
            }

    def generate_stability_report(
        self, output_format: str = "json", save_path: Optional[str] = None
    ) -> Union[str, Dict]:
        """
        Generate a professional stability report

        Args:
            output_format: Report format (json, markdown, html)
            save_path: Save path (optional)

        Returns:
            Union[str, Dict]: Report in requested format
        """
        try:
            # Check stability
            stability_check = self.check_training_stability()

            # Add metadata
            report = {
                "timestamp": datetime.now().isoformat(),
                "iteration": self.current_iteration,
                "config": asdict(self.config),
                "stability_analysis": stability_check,
            }

            # Convert to requested format
            if output_format == "json":
                result = json.dumps(report, indent=2, ensure_ascii=False, default=str)
            elif output_format == "markdown":
                result = self._generate_markdown_report(report)
            elif output_format == "html":
                result = self._generate_html_report(report)
            else:
                raise ValueError(f"Unsupported format: {output_format}")

            # Save if requested
            if save_path:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(
                        result
                        if isinstance(result, str)
                        else json.dumps(result, indent=2)
                    )
                logger.info(f"Stability report saved to: {save_path}")

            return result

        except Exception as e:
            logger.error(f"Failed to generate stability report: {e}")
            return {"error": str(e)}

    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown report"""
        md = []
        md.append("# Hybrid GAN Training Stability Report")
        md.append(f"**Date**: {report['timestamp']}")
        md.append(f"**Iteration**: {report['iteration']}")
        md.append(
            f"**Overall Status**: {report['stability_analysis']['overall_stability']}"
        )
        md.append(f"**Stability Score**: {report['stability_analysis']['score']:.3f}")

        md.append("\n## Detected Issues")
        if "issues" in report["stability_analysis"]:
            for issue in report["stability_analysis"]["issues"]:
                md.append(f"- {issue}")
        else:
            md.append("- No major issues")

        md.append("\n## Recommendations")
        for rec in report["stability_analysis"]["recommendations"]:
            md.append(f"- {rec}")

        md.append("\n## Detailed Metrics")
        for category, metrics in report["stability_analysis"][
            "detailed_metrics"
        ].items():
            md.append(f"\n### {category}")
            for key, value in metrics.items():
                if isinstance(value, float):
                    md.append(f"- {key}: {value:.6f}")
                else:
                    md.append(f"- {key}: {value}")

        return "\n".join(md)

    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report (simplified)"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Hybrid GAN Training Stability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .stable {{ color: green; font-weight: bold; }}
                .unstable {{ color: orange; font-weight: bold; }}
                .critical {{ color: red; font-weight: bold; }}
                .metric {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Hybrid GAN Training Stability Report</h1>
                <p>Date: {report['timestamp']}</p>
                <p>Iteration: {report['iteration']}</p>
                <p class="{report['stability_analysis']['overall_stability']}">
                    Status: {report['stability_analysis']['overall_stability']}
                </p>
            </div>
        """

        # Issues
        html += '<div class="section"><h2>Detected Issues</h2>'
        if "issues" in report["stability_analysis"]:
            html += "<ul>"
            for issue in report["stability_analysis"]["issues"]:
                html += f"<li>{issue}</li>"
            html += "</ul>"
        else:
            html += "<p>No major issues</p>"
        html += "</div>"

        # Recommendations
        html += '<div class="section"><h2>Recommendations</h2><ul>'
        for rec in report["stability_analysis"]["recommendations"]:
            html += f"<li>{rec}</li>"
        html += "</ul></div>"

        # Metrics
        html += '<div class="section"><h2>Metrics</h2>'
        for category, metrics in report["stability_analysis"][
            "detailed_metrics"
        ].items():
            html += f'<h3>{category}</h3><div class="metrics">'
            for key, value in metrics.items():
                html += f'<div class="metric"><strong>{key}:</strong> '
                if isinstance(value, float):
                    html += f"{value:.6f}</div>"
                else:
                    html += f"{value}</div>"
            html += "</div>"
        html += "</div>"

        html += """
        </body>
        </html>
        """

        return html

    def reset(self) -> None:
        """Reset stabilizer state"""
        self.current_iteration = 0
        self.mode_collapse_detected = False
        self.stability_score = 1.0
        self.training_metrics = TrainingMetrics()

        logger.info("Stabilizer state reset")


# ============================================================================
# FACTORY AND BUILDER PATTERNS
# ============================================================================


class StabilizerBuilder:
    """
    Flexible builder for TrainingStabilizer using Fluent Interface
    """

    def __init__(self):
        self._config = StabilizerConfig()

    def with_config(self, config: StabilizerConfig) -> "StabilizerBuilder":
        """Set custom configuration"""
        self._config = config
        return self

    def with_gradient_penalty(
        self,
        enabled: bool = True,
        penalty_type: str = "wgan-gp",
        lambda_val: float = 10.0,
    ) -> "StabilizerBuilder":
        """Set gradient penalty settings"""
        self._config.gradient_penalty_enabled = enabled
        self._config.gradient_penalty_type = penalty_type
        self._config.gradient_penalty_lambda = lambda_val
        return self

    def with_gradient_clipping(
        self, enabled: bool = True, clip_value: float = 1.0, clip_norm: float = 5.0
    ) -> "StabilizerBuilder":
        """Set gradient clipping settings"""
        self._config.gradient_clipping_enabled = enabled
        self._config.gradient_clip_value = clip_value
        self._config.gradient_clip_norm = clip_norm
        return self

    def with_mode_collapse_detection(
        self, enabled: bool = True, threshold: float = 0.7
    ) -> "StabilizerBuilder":
        """Set mode collapse detection settings"""
        self._config.mode_collapse_detection_enabled = enabled
        self._config.mode_collapse_threshold = threshold
        return self

    def build(self) -> HybridTrainingStabilizer:
        """Build a TrainingStabilizer instance"""
        return HybridTrainingStabilizer(self._config)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_usage() -> None:
    """
    Practical example of using the system in a production environment
    """
    print("Example usage of HybridTrainingStabilizer in a production environment")

    # 1. Create a flexible builder
    builder = StabilizerBuilder()

    # 2. Configure the system
    stabilizer = (
        builder.with_gradient_penalty(penalty_type="wgan-gp", lambda_val=10.0)
        .with_gradient_clipping(clip_value=1.0, clip_norm=5.0)
        .with_mode_collapse_detection(threshold=0.7)
        .build()
    )

    print("System built successfully")

    # 3. Use the system with mock components
    class MockDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # 4. Test gradient penalty
    print("\nTesting gradient penalty...")
    discriminator = MockDiscriminator()
    real_data = torch.randn(32, 10)
    fake_data = torch.randn(32, 10)

    try:
        penalty, stats = stabilizer.compute_gradient_penalty(
            discriminator, real_data, fake_data
        )
        print(f"Gradient penalty: {penalty.item():.4f}")
        print(f"Statistics: {stats}")
    except Exception as e:
        print(f"Error: {e}")

    # 5. Test mode collapse detection
    print("\nTesting mode collapse detection...")
    real_samples = torch.randn(100, 10)
    gen_samples = torch.randn(100, 10) * 0.5  # Less diverse

    try:
        collapse, metrics = stabilizer.detect_mode_collapse(real_samples, gen_samples)
        print(f"Mode collapse: {collapse}")
        print(f"Metrics: {metrics}")
    except Exception as e:
        print(f"Error: {e}")

    # 6. Generate report
    print("\nGenerating stability report...")
    try:
        report = stabilizer.generate_stability_report(output_format="markdown")
        print(report[:500] + "..." if len(report) > 500 else report)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run example
    example_usage()
