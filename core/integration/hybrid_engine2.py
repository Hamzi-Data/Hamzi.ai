# core/integration/hybrid_engine2.py
"""
Production-Grade Hybrid Synthetic Data Generation Engine
Fully integrated with train_hybrid_gan.py training orchestrator

Zero-redundancy architecture with complete training pipeline integration.
Supports 10M+ record generation with memory-efficient processing.

Author: Titan AI Architecture Team
Version: 5.0 Enterprise
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import time
import gc
import json
import yaml
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Import base components from File 1
from core.integration.hybrid_engine import (
    EngineState,
    FeatureType,
    FeatureSchema,
    FeatureSchemaRegistry,
    VectorizedDataProcessor,
    DataLoaderFactory,
    MetricsTracker,
    SystemStateManager,
    CircuitBreaker,
    setup_advanced_logger,
)


# Import model components
from core.generative.cgs_gan.conditional_encoder import ConditionalEncoder
from core.generative.cgs_gan.generator import ConditionalGenerator as CGSGenerator
from core.generative.cgs_gan.discriminator import (
    HybridDiscriminator as CGSDiscriminator,
)

logger = setup_advanced_logger(__name__, log_file=Path("logs/hybrid_engine2.log"))


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class HybridEngineConfig:
    """Complete configuration for Hybrid Synthetic Engine."""

    # Project
    project_name: str = "titan_synthetic"
    version: str = "5.0"

    # Paths
    data_path: str = ""
    output_dir: str = "output/training"
    checkpoint_dir: str = "checkpoints"
    save_dir: Path = field(default_factory=lambda: Path("models/hybrid_engine"))

    # Data
    sample_size: Optional[int] = None
    validation_split: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.0002
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    encoder_lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0

    # Model Architecture
    noise_dim: int = 128
    hidden_dim: int = 256
    embedding_dim: int = 64
    num_attention_blocks: int = 1
    attention_heads: int = 4
    generator_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    discriminator_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2

    # Training Stabilization
    n_critic: int = 5
    apply_gradient_penalty: bool = True
    lambda_gp: float = 10.0
    gradient_clip_norm: float = 1.0
    use_spectral_norm: bool = True

    # Learning Rate Scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"
    min_lr: float = 1e-6

    # Mixed Precision
    use_mixed_precision: bool = False
    use_mixed_precision_generation: bool = False

    # Generation
    generation_batch_size: int = 5000
    save_samples: bool = True
    sample_size_eval: int = 1000

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_frequency: int = 10
    keep_last_n_checkpoints: int = 5
    resume_from: Optional[str] = None

    # Monitoring
    log_interval: int = 10
    log_frequency: int = 10
    validate_interval: int = 50
    diversity_check_frequency: int = 10
    mode_collapse_threshold: float = 0.85

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Distributed
    distributed: bool = False
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    dist_backend: str = "nccl"

    # Memory Management
    max_memory_gb: float = 16.0

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60

    # Early Stopping
    early_stopping_patience: int = 20

    def __post_init__(self):
        """Post-initialization validation."""
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(
        cls, config_path: Union[str, Path], **overrides
    ) -> HybridEngineConfig:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        config_dict.update(overrides)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> HybridEngineConfig:
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, save_path: Union[str, Path]):
        """Save configuration to file."""
        save_path = Path(save_path)

        if save_path.suffix == ".yaml" or save_path.suffix == ".yml":
            with open(save_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif save_path.suffix == ".json":
            with open(save_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {save_path.suffix}")


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================


class CheckpointManager:
    """Advanced checkpoint management with rotation and recovery."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_last_n: int = 5,
        logger: Optional[logging.Logger] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.logger = logger or logging.getLogger(__name__)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        epoch: int,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, torch.optim.Optimizer],
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        try:
            checkpoint_data = {
                "epoch": epoch,
                "models": {name: model.state_dict() for name, model in models.items()},
                "optimizers": {
                    name: opt.state_dict() for name, opt in optimizers.items()
                },
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }

            torch.save(checkpoint_data, checkpoint_path)

            # Save metadata
            metadata = {
                "epoch": epoch,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
            }
            metadata_path = checkpoint_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save best model
            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint_data, best_path)
                self.logger.info(f"✓ Saved best model to {best_path}")

            # Rotate old checkpoints
            self._rotate_checkpoints()

            self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        models: Dict[str, nn.Module],
        optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

        # Load model states
        for name, model in models.items():
            if name in checkpoint_data["models"]:
                model.load_state_dict(checkpoint_data["models"][name])

        # Load optimizer states
        if optimizers:
            for name, optimizer in optimizers.items():
                if name in checkpoint_data["optimizers"]:
                    optimizer.load_state_dict(checkpoint_data["optimizers"][name])

        self.logger.info(f"✓ Loaded checkpoint from epoch {checkpoint_data['epoch']}")

        return checkpoint_data

    def _rotate_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        to_remove = checkpoints[: -self.keep_last_n]

        for checkpoint in to_remove:
            try:
                checkpoint.unlink()
                metadata_path = checkpoint.with_suffix(".json")
                if metadata_path.exists():
                    metadata_path.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find latest checkpoint."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        if checkpoints:
            return str(checkpoints[-1])
        return None


# ============================================================================
# TRAINING PIPELINE
# ============================================================================


class GradientPenalty:
    """Compute gradient penalty for WGAN-GP."""

    @staticmethod
    def compute_gp(
        discriminator: nn.Module,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        lambda_gp: float = 10.0,
    ) -> torch.Tensor:
        """Compute gradient penalty."""
        batch_size = real_data.size(0)
        device = real_data.device

        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp

        return gradient_penalty


# ============================================================================
# MAIN HYBRID ENGINE
# ============================================================================


class HybridSyntheticEngine:
    """
    Production-Grade Hybrid Synthetic Data Generation Engine.
    Fully integrated with train_hybrid_gan.py orchestrator.
    """

    def __init__(self, config: Union[HybridEngineConfig, str, Path, Dict]):
        """Initialize the hybrid engine."""

        # Load configuration
        if isinstance(config, HybridEngineConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            config_path = Path(config)
            if config_path.suffix in [".yaml", ".yml"]:
                self.config = HybridEngineConfig.from_yaml(config_path)
            elif config_path.suffix == ".json":
                self.config = HybridEngineConfig.from_json(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        elif isinstance(config, dict):
            self.config = HybridEngineConfig(**config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Setup device
        self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.state_manager = SystemStateManager(self.config)
        self.metrics_tracker = MetricsTracker()
        self.schema_registry = FeatureSchemaRegistry()
        self.data_processor: Optional[VectorizedDataProcessor] = None

        # Models
        self.generator: Optional[nn.Module] = None
        self.discriminator: Optional[nn.Module] = None
        self.encoder: Optional[nn.Module] = None

        # Optimizers
        self.optimizer_g: Optional[torch.optim.Optimizer] = None
        self.optimizer_d: Optional[torch.optim.Optimizer] = None

        # Scalers for mixed precision
        self.scaler_g: Optional[GradScaler] = None
        self.scaler_d: Optional[GradScaler] = None

        # Data
        self.real_data: Optional[pd.DataFrame] = None
        self.feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None
        self.dataloader: Optional[DataLoader] = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        logger.info(f" Hybrid Synthetic Engine initialized (v{self.config.version})")

    def load_data(
        self,
        data: Union[pd.DataFrame, str, Path],
        feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Load and prepare data for training."""
        logger.info("Loading data...")

        # Load DataFrame
        if isinstance(data, pd.DataFrame):
            self.real_data = data
        else:
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            if data_path.suffix == ".csv":
                # Try different delimiters
                with open(data_path, "r") as f:
                    first_line = f.readline()
                    delimiter = ";" if ";" in first_line else ","
                self.real_data = pd.read_csv(data_path, sep=delimiter)
            elif data_path.suffix == ".parquet":
                self.real_data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info(
            f"Loaded {len(self.real_data):,} records with {len(self.real_data.columns)} features"
        )

        # Store feature metadata
        if feature_metadata is not None:
            self.feature_metadata = feature_metadata

            # Register schemas
            for col_name, meta in feature_metadata.items():
                ftype_str = meta.get("type", "continuous")

                # Map to FeatureType
                if ftype_str in ["categorical", "ordinal", "binary"]:
                    ftype = FeatureType.CATEGORICAL
                elif ftype_str == "continuous":
                    ftype = FeatureType.CONTINUOUS
                else:
                    ftype = FeatureType.CONTINUOUS

                schema = FeatureSchema(
                    name=col_name,
                    feature_type=ftype,
                    min_value=meta.get("min"),
                    max_value=meta.get("max"),
                    categories=None,
                    output_dim=meta.get("output_dim", 1),
                )

                self.schema_registry.register(schema)

        # Initialize data processor
        self.data_processor = VectorizedDataProcessor(
            schema_registry=self.schema_registry,
            use_gpu_acceleration=(self.config.device == "cuda"),
        )

        # Fit and transform
        self.data_processor.fit(self.real_data)
        transformed_data = self.data_processor.transform(self.real_data)

        # Create DataLoader
        dataset = TensorDataset(transformed_data)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=(self.config.device == "cuda"),
        )

        logger.info(" Data loading complete")

    def build_models(self):
        """Build generator and discriminator models."""
        logger.info("Building models...")

        if self.feature_metadata is None:
            raise RuntimeError(
                "Feature metadata not available. Call load_data() first."
            )

        # Build Conditional Encoder
        self.encoder = ConditionalEncoder(
            feature_metadata=self.feature_metadata,
            embedding_dim=self.config.embedding_dim,
            transformer_layers=0,
            transformer_heads=self.config.attention_heads,
            fusion_hidden=self.config.hidden_dim * 2,
            dropout_p=self.config.dropout_rate,
        ).to(self.device)

        # ============================================================
        # Build Generator (Metadata-Aware Reconstruction)
        # ============================================================
        # (Continuous vs Categorical)
        self.generator = CGSGenerator(
            feature_metadata=self.feature_metadata,
            noise_dim=self.config.noise_dim,
            condition_dim=self.encoder.output_dim(),
            output_dim=len(self.feature_metadata),
            hidden_dims=self.config.generator_hidden_dims,
            dropout_p=self.config.dropout_rate,
        ).to(self.device)

        logger.info(
            f" Generator built with Metadata-Aware heads | Noise: {self.config.noise_dim}"
        )

        # Build Discriminator
        input_dim = self.encoder.output_dim()

        self.discriminator = CGSDiscriminator(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_scales=3,
            num_attention_blocks=self.config.num_attention_blocks,
            attention_heads=self.config.attention_heads,
            dropout_p=self.config.dropout_rate,
            use_spectral_norm=self.config.use_spectral_norm,
            feature_metadata=self.feature_metadata,
        ).to(self.device)

        # Initialize optimizers
        self.optimizer_g = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.encoder.parameters()),
            lr=self.config.generator_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        self.optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_lr,
            betas=(self.config.beta1, self.config.beta2),
            weight_decay=self.config.weight_decay,
        )

        # Mixed precision scalers
        if self.config.use_mixed_precision:
            self.scaler_g = GradScaler()
            self.scaler_d = GradScaler()

        # Count parameters
        gen_params = sum(p.numel() for p in self.generator.parameters())
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        enc_params = sum(p.numel() for p in self.encoder.parameters())

        logger.info(f" Models built:")
        logger.info(f"  Generator: {gen_params:,} parameters")
        logger.info(f"  Discriminator: {disc_params:,} parameters")
        logger.info(f"  Encoder: {enc_params:,} parameters")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        self.encoder.train()

        epoch_metrics = {
            "g_loss": [],
            "d_loss": [],
            "gp": [],
        }

        for batch_idx, (real_batch,) in enumerate(self.dataloader):
            real_batch = real_batch.to(self.device)
            batch_size = real_batch.size(0)

            # ==========================================
            # Train Discriminator
            # ==========================================
            for _ in range(self.config.n_critic):
                self.optimizer_d.zero_grad()

                # Generate fake data
                noise = torch.randn(
                    batch_size, self.config.noise_dim, device=self.device
                )

                with autocast(enabled=self.config.use_mixed_precision):
                    # Create condition from real data
                    real_conditions = self._tensor_to_conditions(real_batch)
                    condition_vector = self.encoder(real_conditions)

                    # Generate fake samples
                    fake_batch = self.generator(noise, condition_vector)
                    fake_conditions = self._tensor_to_conditions(fake_batch)
                    fake_condition_vector = self.encoder(fake_conditions)

                    # Discriminator predictions
                    real_encoded = self.encoder(real_conditions)
                    fake_encoded = self.encoder(fake_conditions)

                    real_validity = self.discriminator(real_encoded)
                    fake_validity = self.discriminator(fake_encoded.detach())

                    # Wasserstein loss
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)

                    # Gradient penalty
                    gp = torch.tensor(0.0, device=self.device)
                    if self.config.apply_gradient_penalty:
                        gp = GradientPenalty.compute_gp(
                            self.discriminator,
                            real_encoded,
                            fake_encoded.detach(),
                            self.config.lambda_gp,
                        )
                        d_loss = d_loss + gp

                # Backward
                if self.scaler_d:
                    self.scaler_d.scale(d_loss).backward()
                    self.scaler_d.unscale_(self.optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.config.gradient_clip_norm
                    )
                    self.scaler_d.step(self.optimizer_d)
                    self.scaler_d.update()
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), self.config.gradient_clip_norm
                    )
                    self.optimizer_d.step()

                epoch_metrics["d_loss"].append(d_loss.item())
                epoch_metrics["gp"].append(gp.item())

            # ==========================================
            # Train Generator
            # ==========================================
            self.optimizer_g.zero_grad()

            noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)

            with autocast(enabled=self.config.use_mixed_precision):
                real_conditions = self._tensor_to_conditions(real_batch)
                condition_vector = self.encoder(real_conditions)

                fake_batch = self.generator(noise, condition_vector)
                fake_conditions = self._tensor_to_conditions(fake_batch)
                fake_encoded = self.encoder(fake_conditions)

                fake_validity = self.discriminator(fake_encoded)

                # Generator loss
                g_loss = -torch.mean(fake_validity)

            # Backward
            if self.scaler_g:
                self.scaler_g.scale(g_loss).backward()
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    list(self.generator.parameters()) + list(self.encoder.parameters()),
                    self.config.gradient_clip_norm,
                )
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.generator.parameters()) + list(self.encoder.parameters()),
                    self.config.gradient_clip_norm,
                )
                self.optimizer_g.step()

            epoch_metrics["g_loss"].append(g_loss.item())
            self.global_step += 1

            # Logging
            if batch_idx % self.config.log_frequency == 0:
                logger.info(
                    f"Epoch [{epoch}/{self.config.epochs}] "
                    f"Batch [{batch_idx}/{len(self.dataloader)}] "
                    f"G_loss: {g_loss.item():.4f} D_loss: {d_loss.item():.4f}"
                )

        # Aggregate metrics
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        self.current_epoch = epoch

        return avg_metrics

    def _tensor_to_conditions(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert flat tensor to condition dictionary."""
        conditions = {}
        col_idx = 0

        for col_name, meta in self.feature_metadata.items():
            ftype = meta.get("type", "continuous")

            if ftype in ["categorical", "ordinal", "binary"]:
                # Categorical: single column (class index)
                conditions[col_name] = tensor[:, col_idx].long()
                col_idx += 1
            else:
                # Continuous: single column
                conditions[col_name] = tensor[:, col_idx : col_idx + 1]
                col_idx += 1

        return conditions

    def generate_samples(self, num_samples: int) -> pd.DataFrame:
        """Generate synthetic samples."""
        self.generator.eval()
        self.encoder.eval()

        all_samples = []
        batch_size = self.config.generation_batch_size
        num_batches = int(np.ceil(num_samples / batch_size))

        with torch.no_grad():
            for batch_idx in range(num_batches):
                current_batch_size = min(
                    batch_size, num_samples - batch_idx * batch_size
                )

                # Generate noise
                noise = torch.randn(
                    current_batch_size, self.config.noise_dim, device=self.device
                )

                # Sample random conditions from real data
                sample_indices = np.random.choice(
                    len(self.real_data), current_batch_size
                )
                sample_data = self.real_data.iloc[sample_indices]

                # Transform to tensor
                sample_tensor = self.data_processor.transform(sample_data).to(
                    self.device
                )
                sample_conditions = self._tensor_to_conditions(sample_tensor)
                condition_vector = self.encoder(sample_conditions)

                # Generate
                fake_batch = self.generator(noise, condition_vector)

                # Convert to DataFrame
                fake_df = self.data_processor.inverse_transform(fake_batch)
                all_samples.append(fake_df)

        synthetic_data = pd.concat(all_samples, axis=0, ignore_index=True)

        return synthetic_data

    def evaluate_quality(self, synthetic_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate quality of synthetic data."""
        metrics = {}

        # Statistical tests for continuous features
        for col in self.real_data.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_data.columns:
                continue

            real_values = self.real_data[col].dropna()
            synth_values = synthetic_data[col].dropna()

            # KS test
            ks_stat, ks_pval = stats.ks_2samp(real_values, synth_values)
            metrics[f"{col}_ks_statistic"] = ks_stat
            metrics[f"{col}_ks_pvalue"] = ks_pval

        # Overall quality score
        ks_pvalues = [v for k, v in metrics.items() if "ks_pvalue" in k]
        if ks_pvalues:
            metrics["avg_ks_pvalue"] = np.mean(ks_pvalues)
            metrics["quality_score"] = sum(1 for p in ks_pvalues if p > 0.05) / len(
                ks_pvalues
            )

        return metrics

    def save_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Save model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "encoder": self.encoder.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config.to_dict(),
        }

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f" Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model checkpoint with support for emergency saves."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint_data = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "models" in checkpoint_data:
            models = checkpoint_data["models"]
            self.generator.load_state_dict(
                models.get("generator", models.get("model_g", {}))
            )
            self.discriminator.load_state_dict(
                models.get("discriminator", models.get("model_d", {}))
            )
            self.encoder.load_state_dict(models.get("encoder", {}))
        else:

            self.generator.load_state_dict(checkpoint_data["generator"])
            self.discriminator.load_state_dict(checkpoint_data["discriminator"])
            self.encoder.load_state_dict(checkpoint_data["encoder"])

        if "optimizers" in checkpoint_data:
            opts = checkpoint_data["optimizers"]
            if self.optimizer_g and "g_opt" in opts:
                self.optimizer_g.load_state_dict(opts["g_opt"])
            if self.optimizer_d and "d_opt" in opts:
                self.optimizer_d.load_state_dict(opts["d_opt"])
        elif "optimizer_g" in checkpoint_data:
            self.optimizer_g.load_state_dict(checkpoint_data["optimizer_g"])
            self.optimizer_d.load_state_dict(checkpoint_data["optimizer_d"])

        self.current_epoch = checkpoint_data.get("epoch", 0)
        self.global_step = checkpoint_data.get("global_step", 0)

        logger.info(f"✓ Checkpoint loaded successfully from epoch {self.current_epoch}")

    def generate_training_report(self) -> str:
        """Generate comprehensive training report."""
        report_lines = [
            "=" * 80,
            "HYBRID SYNTHETIC ENGINE - TRAINING REPORT",
            "=" * 80,
            f"\nProject: {self.config.project_name}",
            f"Version: {self.config.version}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n" + "-" * 80,
            "CONFIGURATION",
            "-" * 80,
            f"Device: {self.config.device}",
            f"Epochs: {self.config.epochs}",
            f"Batch Size: {self.config.batch_size}",
            f"Learning Rate (G): {self.config.generator_lr}",
            f"Learning Rate (D): {self.config.discriminator_lr}",
            f"Noise Dimension: {self.config.noise_dim}",
            f"Hidden Dimension: {self.config.hidden_dim}",
            "\n" + "-" * 80,
            "DATA STATISTICS",
            "-" * 80,
        ]

        if self.real_data is not None:
            report_lines.extend(
                [
                    f"Training Samples: {len(self.real_data):,}",
                    f"Features: {len(self.real_data.columns)}",
                    f"Feature Names: {', '.join(self.real_data.columns)}",
                ]
            )

        report_lines.extend(
            [
                "\n" + "-" * 80,
                "MODEL ARCHITECTURE",
                "-" * 80,
            ]
        )

        if self.generator:
            gen_params = sum(p.numel() for p in self.generator.parameters())
            report_lines.append(f"Generator Parameters: {gen_params:,}")

        if self.discriminator:
            disc_params = sum(p.numel() for p in self.discriminator.parameters())
            report_lines.append(f"Discriminator Parameters: {disc_params:,}")

        if self.encoder:
            enc_params = sum(p.numel() for p in self.encoder.parameters())
            report_lines.append(f"Encoder Parameters: {enc_params:,}")

        report_lines.extend(
            [
                "\n" + "-" * 80,
                "TRAINING STATUS",
                "-" * 80,
                f"Current Epoch: {self.current_epoch}",
                f"Global Steps: {self.global_step}",
                f"State: {self.state_manager.state.name}",
            ]
        )

        # Add metrics if available
        if self.metrics_tracker.history:
            report_lines.extend(
                [
                    "\n" + "-" * 80,
                    "METRICS SUMMARY",
                    "-" * 80,
                ]
            )

            for metric_name, values in self.metrics_tracker.history.items():
                if values:
                    avg_value = np.mean(values[-10:])  # Last 10 values
                    report_lines.append(f"{metric_name}: {avg_value:.4f}")

        report_lines.extend(
            [
                "\n" + "=" * 80,
                "END OF REPORT",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)


# ============================================================================
# QUALITY AUDITOR
# ============================================================================


class QualityAuditor:
    """Comprehensive quality auditing for synthetic data."""

    def __init__(self, real_data: pd.DataFrame):
        self.real_data = real_data
        self.audit_results: Dict[str, Any] = {}

    def audit(
        self,
        synthetic_data: pd.DataFrame,
        tests: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive quality audit."""

        available_tests = {
            "ks_test": self._ks_test,
            "chi_square": self._chi_square_test,
            "js_divergence": self._js_divergence,
            "correlation": self._correlation_test,
        }

        tests = tests or list(available_tests.keys())

        logger.info(f"Starting quality audit with {len(tests)} tests")

        results = {
            "timestamp": datetime.now().isoformat(),
            "num_real_samples": len(self.real_data),
            "num_synthetic_samples": len(synthetic_data),
            "tests": {},
        }

        for test_name in tests:
            if test_name in available_tests:
                try:
                    logger.info(f"Running test: {test_name}")
                    test_result = available_tests[test_name](synthetic_data)
                    results["tests"][test_name] = test_result
                except Exception as e:
                    logger.error(f"Test {test_name} failed: {e}")
                    results["tests"][test_name] = {"error": str(e)}

        # Overall quality score
        results["overall_score"] = self._compute_overall_score(results["tests"])

        self.audit_results = results
        return results

    def _ks_test(self, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for continuous features."""
        results = {}

        for col in self.real_data.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_data.columns:
                continue

            real_values = self.real_data[col].dropna()
            synth_values = synthetic_data[col].dropna()

            statistic, pvalue = stats.ks_2samp(real_values, synth_values)

            results[col] = {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "passed": pvalue > 0.05,
            }

        pass_rate = (
            sum(1 for v in results.values() if v["passed"]) / len(results)
            if results
            else 0
        )

        return {
            "feature_results": results,
            "pass_rate": pass_rate,
            "summary": f"{pass_rate:.1%} of features passed KS test",
        }

    def _chi_square_test(self, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Chi-square test for categorical features."""
        results = {}

        categorical_cols = self.real_data.select_dtypes(
            include=["object", "category"]
        ).columns

        for col in categorical_cols:
            if col not in synthetic_data.columns:
                continue

            real_counts = self.real_data[col].value_counts()
            synth_counts = synthetic_data[col].value_counts()

            all_categories = set(real_counts.index) | set(synth_counts.index)

            real_freq = np.array([real_counts.get(cat, 0) for cat in all_categories])
            synth_freq = np.array([synth_counts.get(cat, 0) for cat in all_categories])

            real_freq = real_freq / real_freq.sum()
            synth_freq = synth_freq / synth_freq.sum()

            try:
                statistic, pvalue = stats.chisquare(synth_freq, real_freq)

                results[col] = {
                    "statistic": float(statistic),
                    "pvalue": float(pvalue),
                    "passed": pvalue > 0.05,
                }
            except Exception as e:
                results[col] = {"error": str(e)}

        pass_rate = (
            sum(1 for v in results.values() if v.get("passed", False)) / len(results)
            if results
            else 0
        )

        return {
            "feature_results": results,
            "pass_rate": pass_rate,
            "summary": f"{pass_rate:.1%} of categorical features passed chi-square test",
        }

    def _js_divergence(self, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Jensen-Shannon divergence for distribution comparison."""
        results = {}

        for col in self.real_data.select_dtypes(include=[np.number]).columns:
            if col not in synthetic_data.columns:
                continue

            real_values = self.real_data[col].dropna()
            synth_values = synthetic_data[col].dropna()

            bins = np.linspace(
                min(real_values.min(), synth_values.min()),
                max(real_values.max(), synth_values.max()),
                50,
            )

            real_hist, _ = np.histogram(real_values, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_values, bins=bins, density=True)

            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()

            js_div = jensenshannon(real_hist, synth_hist)

            results[col] = {
                "divergence": float(js_div),
                "similarity": 1.0 - float(js_div),
                "passed": js_div < 0.1,
            }

        avg_similarity = (
            np.mean([v["similarity"] for v in results.values()]) if results else 0
        )

        return {
            "feature_results": results,
            "average_similarity": avg_similarity,
            "summary": f"Average JS similarity: {avg_similarity:.1%}",
        }

    def _correlation_test(self, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare correlation structures."""
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numeric_cols if col in synthetic_data.columns]

        if len(common_cols) < 2:
            return {"error": "Insufficient numeric features for correlation test"}

        real_corr = self.real_data[common_cols].corr()
        synth_corr = synthetic_data[common_cols].corr()

        corr_diff = np.linalg.norm(real_corr.values - synth_corr.values, "fro")
        max_possible_diff = np.sqrt(2 * len(common_cols) ** 2)

        similarity = 1.0 - (corr_diff / max_possible_diff)

        return {
            "correlation_difference": float(corr_diff),
            "correlation_similarity": float(similarity),
            "passed": similarity > 0.8,
            "summary": f"Correlation similarity: {similarity:.1%}",
        }

    def _compute_overall_score(self, test_results: Dict[str, Any]) -> float:
        """Compute overall quality score from test results."""
        scores = []

        for test_name, result in test_results.items():
            if "error" in result:
                continue

            if "pass_rate" in result:
                scores.append(result["pass_rate"])
            elif "similarity" in result:
                scores.append(result["similarity"])
            elif "correlation_similarity" in result:
                scores.append(result["correlation_similarity"])
            elif "passed" in result:
                scores.append(1.0 if result["passed"] else 0.0)

        return np.mean(scores) if scores else 0.0

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate human-readable audit report."""
        if not self.audit_results:
            return "No audit results available. Run audit() first."

        report_lines = [
            "=" * 80,
            "SYNTHETIC DATA QUALITY AUDIT REPORT",
            "=" * 80,
            f"\nGenerated: {self.audit_results['timestamp']}",
            f"Real samples: {self.audit_results['num_real_samples']:,}",
            f"Synthetic samples: {self.audit_results['num_synthetic_samples']:,}",
            f"\nOverall Quality Score: {self.audit_results['overall_score']:.2%}",
            "\n" + "-" * 80,
            "TEST RESULTS:",
            "-" * 80,
        ]

        for test_name, result in self.audit_results["tests"].items():
            report_lines.append(f"\n{test_name.upper().replace('_', ' ')}:")

            if "error" in result:
                report_lines.append(f"   Error: {result['error']}")
            elif "summary" in result:
                report_lines.append(f"  {result['summary']}")

            if "passed" in result:
                status = " PASSED" if result["passed"] else " FAILED"
                report_lines.append(f"  Status: {status}")

        report = "\n".join(report_lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

            logger.info(f"Audit report saved to: {output_path}")

        return report


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Configuration
    "HybridEngineConfig",
    # Checkpoint Management
    "CheckpointManager",
    # Training Components
    "GradientPenalty",
    # Main Engine
    "HybridSyntheticEngine",
    # Quality Assurance
    "QualityAuditor",
]


# ============================================================================
# CLI ENTRY POINT
# ============================================================================


def main_cli():
    """Command line interface entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Titan Hybrid Synthetic Data Generation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data", type=str, required=True, help="Path to input data")
    parser.add_argument("--output", type=str, default="output/synthetic_data.parquet")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--mode", type=str, choices=["train", "generate", "both"], default="both"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = HybridEngineConfig.from_yaml(args.config)
    else:
        config = HybridEngineConfig()

    # Override with CLI args
    if args.epochs:
        config.epochs = args.epochs

    # Initialize engine
    engine = HybridSyntheticEngine(config)

    # Load data (with auto-detection of feature metadata)
    logger.info(f"Loading data from: {args.data}")

    # For CLI, we need to auto-detect metadata
    data = pd.read_csv(args.data)

    # Auto-detect feature metadata
    feature_metadata = {}
    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            feature_metadata[col] = {
                "type": "continuous",
                "output_dim": 1,
                "min": float(data[col].min()),
                "max": float(data[col].max()),
            }
        else:
            cardinality = data[col].nunique()
            feature_metadata[col] = {
                "type": "categorical",
                "output_dim": cardinality,
                "num_classes": cardinality,
            }

    engine.load_data(data, feature_metadata)

    # Build models
    engine.build_models()

    # Train
    if args.mode in ["train", "both"]:
        logger.info("Starting training...")
        for epoch in range(1, config.epochs + 1):
            metrics = engine.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch} - G_loss: {metrics['g_loss']:.4f}, D_loss: {metrics['d_loss']:.4f}"
            )

    # Generate
    if args.mode in ["generate", "both"]:
        logger.info(f"Generating {args.num_samples:,} samples...")
        synthetic_data = engine.generate_samples(args.num_samples)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".parquet":
            synthetic_data.to_parquet(output_path, index=False)
        else:
            synthetic_data.to_csv(output_path, index=False)

        logger.info(f" Synthetic data saved to: {output_path}")

    logger.info(" Pipeline completed successfully!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main_cli()
    else:
        print("=" * 80)
        print("TITAN HYBRID SYNTHETIC DATA GENERATION ENGINE v5.0")
        print("=" * 80)
        print("\nEnterprise-Grade Features:")
        print("  ✓ Advanced GAN architecture with conditional encoding")
        print("  ✓ Production-grade training pipeline")
        print("  ✓ 10M+ sample generation capability")
        print("  ✓ Comprehensive quality auditing")
        print("  ✓ Zero-redundancy architecture")
        print("  ✓ Full integration with training orchestrator")
        print("\nUsage:")
        print(
            "  python hybrid_engine2.py --data data.csv --mode both --num-samples 1000000"
        )
        print("\nFor help:")
        print("  python hybrid_engine2.py --help")
        print("=" * 80)
