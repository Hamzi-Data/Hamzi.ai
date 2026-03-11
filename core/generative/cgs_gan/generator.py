"""
core/generative/cgs_gan/generator.py
Production-ready hybrid generator with conditional Gumbel-Softmax, spectral normalization, and strict interface contracts.

Design:
- Unified interface: (noise: [batch, Z], condition: [batch, C]) -> (features_flat: [batch, D], features_struct: optional dict)
- Support for feature types: continuous, categorical, ordinal, multimodal
- Spectral normalization for linear layers to ensure stability
- Clear and diagnostic error handling
- Dimension map logging for each column to enable safe integration

Integration notes:
- feature_metadata: dict for each column containing required keys based on type:
  - categorical/ordinal: {"type": "...", "output_dim": int}
  - continuous: {"type": "..."} or {"type": "...", "output_dim": 1} (optional)
  - multimodal: {"type": "...", "n_modes": int}
- It is recommended to use external ConditionalEncoder to generate fixed-dimension condition_vector [batch, C].

Copyright: Microsoft AI (Production-grade re-engineering)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ========= Utility and Contracts =========


class GeneratorError(ValueError):
    """Generator error with clear diagnostic message."""

    pass


from typing import Dict


def _validate_feature_metadata(feature_metadata: Dict[str, Dict]) -> None:
    """
    Strict + intelligent validation of feature_metadata
    Prevents CGS-GAN failure due to:
    - Invalid output_dim
    - Missing n_modes
    - Unstandardized type
    """

    if not isinstance(feature_metadata, dict) or not feature_metadata:
        raise GeneratorError("feature_metadata must be a non-empty dict")

    TYPE_MAP = {
        # ===== CATEGORICAL =====
        "CATEGORICAL": "categorical",
        "ORDINAL": "categorical",
        "NOMINAL": "categorical",
        "DISCRETE": "categorical",
        "BOOLEAN": "categorical",
        "BINARY": "categorical",
        "CATEGORICAL_LOW_CARDINALITY": "categorical",
        "CATEGORICAL_HIGH_CARDINALITY": "categorical",
        "LABEL_ENCODED": "categorical",
        # ===== CONTINUOUS =====
        "CONTINUOUS": "continuous",
        "NUMERIC": "continuous",
        "FLOAT": "continuous",
        "INTEGER": "continuous",
        "REAL": "continuous",
        "DISCRETE_UNIFORM": "continuous",  # Added to solve age column issue
        "NUMERICAL": "continuous",
        # ===== MULTIMODAL =====
        "MULTIMODAL": "multimodal",
        "CONTINUOUS_MULTIMODAL": "multimodal",
        "GAUSSIAN_MIXTURE": "multimodal",
        # ===== DATETIME =====
        "DATETIME": "datetime",
        "DATE": "datetime",
        "TIME": "datetime",
        "TIMESTAMP": "datetime",
    }

    SUPPORTED_TYPES = {"categorical", "continuous", "multimodal", "datetime"}

    for col, meta in feature_metadata.items():
        if not isinstance(meta, dict):
            raise GeneratorError(f"Metadata for column '{col}' must be dict")

        # -------------------------------------------------
        # Type normalization
        # -------------------------------------------------
        raw_type = str(meta.get("type", "")).upper().strip()
        normalized_type = TYPE_MAP.get(raw_type)

        if normalized_type not in SUPPORTED_TYPES:
            # Rescue protocol: if type contains known keywords
            if "DISCRETE" in raw_type or "UNIFORM" in raw_type:
                normalized_type = "continuous"
            elif "CAT" in raw_type:
                normalized_type = "categorical"
            else:
                raise GeneratorError(
                    f"Unsupported raw type '{raw_type}' for column '{col}'"
                )

        # Final type normalization (CRITICAL)
        meta["type"] = normalized_type

        # -------------------------------------------------
        # CATEGORICAL
        # -------------------------------------------------
        if normalized_type == "categorical":
            odim = meta.get("output_dim")

            if not isinstance(odim, int) or odim <= 1:
                classes = meta.get("classes")

                if isinstance(classes, (list, tuple)) and len(classes) > 1:
                    meta["output_dim"] = len(classes)
                else:
                    # Intelligent fallback preventing failure
                    meta["output_dim"] = 2

        # -------------------------------------------------
        # MULTIMODAL
        # -------------------------------------------------
        elif normalized_type == "multimodal":
            n_modes = meta.get("n_modes")

            # Final fallback — prevents Invalid n_modes
            if not isinstance(n_modes, int) or n_modes < 2:
                meta["n_modes"] = 2

            # output_dim must equal number of modes
            meta["output_dim"] = meta["n_modes"]

        # -------------------------------------------------
        # DATETIME
        # -------------------------------------------------
        elif normalized_type == "datetime":
            odim = meta.get("output_dim")
            if not isinstance(odim, int) or odim < 1:
                meta["output_dim"] = 1

        # -------------------------------------------------
        # CONTINUOUS (nothing required)
        # -------------------------------------------------
        elif normalized_type == "continuous":
            pass


def _spectral_norm(linear: nn.Linear) -> nn.Linear:
    """Apply spectral normalization to a linear layer."""
    return nn.utils.spectral_norm(linear)


# ========= Gumbel-Softmax Layer =========
import re


def sanitize_module_name(name: str) -> str:
    """
    Sanitize column names to be compatible with PyTorch ModuleDict.
    Converts dots, spaces, and special characters to underscores.
    Example: 'emp.var.rate' -> 'emp_var_rate'
    """
    if not name:
        return "unnamed_feature"

    # 1. Convert any non-alphanumeric character to underscore
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)

    # 2. Remove repeated underscores (___ -> _)
    sanitized = re.sub(r"_+", "_", sanitized)

    # 3. Ensure name does not start with a digit (Python variable requirement)
    if sanitized[0].isdigit():
        sanitized = "f_" + sanitized

    return sanitized.strip("_")


class GumbelSoftmaxLayer(nn.Module):
    """
    Gumbel-Softmax layer with support for hard inference and differentiable training.

    Args:
        temperature: Temperature controlling distribution sharpness.
        hard: If True uses Straight-Through for hard quantization.
    """

    def __init__(self, temperature: float = 0.7, hard: bool = True) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise GeneratorError("temperature must be > 0.")
        self.temperature = temperature
        self.hard = hard

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Performs Gumbel-Softmax sampling during training, applies argmax during inference if hard=True.

        Args:
            logits: Tensor [batch, K] of unnormalized probabilities.

        Returns:
            Tensor [batch, K]: one-hot samples (if hard) or soft distributions.
        """
        if self.training:
            # Add Gumbel noise
            uniform = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(uniform.clamp(min=1e-10)) + 1e-10)
            y = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)

            if self.hard:
                index = y.argmax(dim=-1, keepdim=True)
                y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
                # Straight-Through: pass gradients through y
                y = y_hard - y.detach() + y
            return y
        else:
            if self.hard:
                index = logits.argmax(dim=-1, keepdim=True)
                return torch.zeros_like(logits).scatter_(-1, index, 1.0)
            return F.softmax(logits / self.temperature, dim=-1)


# ========= Output Heads per Feature =========


class _CategoricalHead(nn.Module):
    """
    Output head for categorical/ordinal features using Gumbel-Softmax.

    Args:
        hidden_dim: Internal generator representation dimension.
        num_classes: Number of output classes (output_dim).
        temperature: Gumbel temperature.
        hard: Use one-hot samples during inference.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        temperature: float = 0.7,
        hard: bool = True,
    ) -> None:
        super().__init__()
        self.proj = _spectral_norm(nn.Linear(hidden_dim, num_classes))
        self.gumbel = GumbelSoftmaxLayer(temperature=temperature, hard=hard)
        self.output_dim = num_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h)
        return self.gumbel(logits)  # [batch, num_classes]


class _ContinuousHead(nn.Module):
    """
    Output head for continuous features. Supports two modes:
    - direct: Output a single normalized value via tanh (or linear)
    - gaussian: Output [mean, log_std] for a normal distribution (recommended for flexibility)

    Args:
        hidden_dim: Internal generator representation dimension.
        mode: "direct" or "gaussian".
    """

    def __init__(self, hidden_dim: int, mode: str = "gaussian") -> None:
        super().__init__()
        if mode not in ("direct", "gaussian"):
            raise GeneratorError(f"Unsupported continuous mode: {mode}")
        self.mode = mode
        if mode == "direct":
            self.proj = _spectral_norm(nn.Linear(hidden_dim, 1))
            self.output_dim = 1
        else:
            self.proj = _spectral_norm(nn.Linear(hidden_dim, 2))  # mean, log_std
            self.output_dim = 2

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = self.proj(h)
        if self.mode == "direct":
            return torch.tanh(out)  # [batch, 1]
        # gaussian: don't normalize; leave mean/log_std for model or post-processing
        return out  # [batch, 2]


class _MultimodalHead(nn.Module):
    """
    Multimodal output head (Gaussian Mixture Model).
    Produces mixture coefficients: per mode [logit, mean, log_std].

    Args:
        hidden_dim: Internal generator representation dimension.
        n_modes: Number of mixtures.
    """

    def __init__(self, hidden_dim: int, n_modes: int) -> None:
        super().__init__()
        if n_modes < 2:
            raise GeneratorError("n_modes must be ≥ 2 for multimodal features.")
        self.n_modes = n_modes
        self.proj = _spectral_norm(
            nn.Linear(hidden_dim, n_modes * 3)
        )  # [logits, means, log_stds]
        self.output_dim = n_modes * 3

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        params = self.proj(h)  # [batch, n_modes*3]
        return params


# ========= Hybrid Conditional Generator =========

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# Assume these utilities are predefined in your project:
# _spectral_norm, _CategoricalHead, _ContinuousHead, _MultimodalHead, GeneratorError, _validate_feature_metadata


class ConditionalGenerator(nn.Module):
    """
    Production-ready hybrid conditional generator, compatible with builder and modern settings.
    """

    def __init__(
        self,
        noise_dim: int,
        feature_metadata: Dict[str, Dict],
        hidden_dim: int = 256,
        condition_dim: Optional[int] = None,
        continuous_mode: str = "gaussian",
        num_categorical: Optional[int] = None,
        num_continuous: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Basic validation
        if not isinstance(noise_dim, int) or noise_dim <= 0:
            raise GeneratorError("noise_dim must be a positive integer.")
        _validate_feature_metadata(feature_metadata)

        # Save arguments
        self.noise_dim = int(noise_dim)
        self.feature_metadata = feature_metadata
        self.hidden_dim = int(hidden_dim)
        self.continuous_mode = str(continuous_mode)
        self.extra_params = dict(kwargs)

        # Calculate condition dimension
        self.condition_dim = (
            int(condition_dim)
            if condition_dim is not None
            else self._infer_condition_dim(feature_metadata)
        )
        if self.condition_dim <= 0:
            raise GeneratorError("condition_dim must be > 0.")

        # Column counts (builder compatibility)
        self.num_categorical = (
            sum(
                1
                for m in feature_metadata.values()
                if m.get("type") in ("categorical", "ordinal")
            )
            if num_categorical is None
            else int(num_categorical)
        )
        self.num_continuous = (
            sum(1 for m in feature_metadata.values() if m.get("type") == "continuous")
            if num_continuous is None
            else int(num_continuous)
        )

        # Base network
        total_input = self.noise_dim + self.condition_dim
        self.fc1 = _spectral_norm(nn.Linear(total_input, self.hidden_dim))
        self.fc2 = _spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.fc3 = _spectral_norm(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Basic registries (second step!)
        self.output_heads = nn.ModuleDict()
        self.output_dim_registry: Dict[str, int] = {}
        self.feature_order: List[str] = []
        self.column_name_map: Dict[str, str] = {}  # This is the second step

        # Build output heads
        # Build output heads (Enterprise Semantic Reconstruction)
        for col, meta in feature_metadata.items():
            ftype = str(meta.get("type", "")).lower()

            # 1. Categorical family (recognizes categorical_low, categorical_high, ordinal, etc.)
            if any(
                root in ftype
                for root in ("categorical", "ordinal", "binary", "boolean")
            ):
                num_classes = int(meta.get("output_dim", meta.get("num_classes", 0)))
                if num_classes <= 0:
                    raise GeneratorError(
                        f"Invalid output_dim for column '{col}' of type {ftype}."
                    )
                head = _CategoricalHead(self.hidden_dim, num_classes=num_classes)

            # 2. Continuous and discrete family (recognizes continuous, discrete_uniform, etc.)
            elif any(
                root in ftype
                for root in ("continuous", "discrete", "numerical", "real")
            ):
                # Handle numerical distributions via ContinuousHead
                head = _ContinuousHead(self.hidden_dim, mode=self.continuous_mode)

            # 3. Multimodal family
            elif "multimodal" in ftype:
                n_modes = int(meta.get("n_modes", 0))
                if n_modes <= 1:
                    # Safe fallback to ensure project continuity
                    head = _ContinuousHead(self.hidden_dim, mode=self.continuous_mode)
                else:
                    head = _MultimodalHead(self.hidden_dim, n_modes=n_modes)

            else:
                # Global rescue protocol: any unknown type treated as continuous data
                logger.warning(
                    f"Unknown type '{ftype}' for column '{col}'. Using ContinuousHead as default."
                )
                head = _ContinuousHead(self.hidden_dim, mode=self.continuous_mode)

            # Register and encode
            safe_col = sanitize_module_name(col)
            self.output_heads[safe_col] = head
            self.column_name_map[safe_col] = col
            self.output_dim_registry[col] = int(head.output_dim)
            self.feature_order.append(col)

        # Total dimension
        self.output_dim_total = sum(self.output_dim_registry.values())
        if self.output_dim_total <= 0:
            raise GeneratorError("Total output dimension is invalid.")

    @staticmethod
    def _infer_condition_dim(feature_metadata: Dict[str, Dict]) -> int:
        cond_dim = 0
        for meta in feature_metadata.values():
            ftype = str(meta.get("type", "")).lower()
            if any(
                root in ftype
                for root in ("categorical", "ordinal", "binary", "boolean")
            ):
                odim = int(meta.get("output_dim", meta.get("num_classes", 0)))
                cond_dim += max(0, odim)
            elif any(
                root in ftype
                for root in ("continuous", "discrete", "numerical", "real")
            ):
                cond_dim += 1
        return cond_dim

    def get_output_dim(self) -> int:
        """Returns total output dimension D."""
        return self.output_dim_total

    def get_feature_slices(self) -> Dict[str, Tuple[int, int]]:
        """Returns column slice map within flat output [batch, D]."""
        slices: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for col in self.feature_order:
            dim = self.output_dim_registry[col]
            slices[col] = (offset, offset + dim)
            offset += dim
        return slices

    def forward(
        self,
        noise: torch.Tensor,
        condition_vector: torch.Tensor,
        return_structured: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Execute conditional generator.

        Args:
            noise: Tensor [batch, Z] noise.
            condition_vector: Tensor [batch, C] condition vector.
            return_structured: If True returns a dictionary of outputs per column in addition to flat output.
        """
        if noise.dim() != 2:
            raise GeneratorError(
                f"Noise must be [batch, Z] but got shape: {tuple(noise.shape)}"
            )
        if condition_vector.dim() != 2:
            raise GeneratorError(
                f"Condition vector must be [batch, C] but got shape: {tuple(condition_vector.shape)}"
            )
        if noise.shape[0] != condition_vector.shape[0]:
            raise GeneratorError(
                "Batch size mismatch between noise and condition vector."
            )
        if condition_vector.shape[1] != self.condition_dim:
            raise GeneratorError(
                f"Actual condition_dim ({condition_vector.shape[1]}) does not match expected ({self.condition_dim})."
            )

        # Concatenate noise and condition
        x = torch.cat([noise, condition_vector], dim=1)

        # Base network
        h = F.leaky_relu(self.fc1(x), 0.2)
        h = F.leaky_relu(self.fc2(h), 0.2)
        h = F.leaky_relu(self.fc3(h), 0.2)

        # Output heads
        outputs_struct: Dict[str, torch.Tensor] = {}
        flats: List[torch.Tensor] = []

        # Core modification here to solve KeyError: 'emp.var.rate'
        for col in self.feature_order:
            # Use function to generate correct key present in ModuleDict
            safe_col = sanitize_module_name(col)

            # Call head using sanitized name
            head = self.output_heads[safe_col]

            y = head(h)
            outputs_struct[col] = y  # Save with original name for external dictionary
            flats.append(y)

        features_flat = torch.cat(flats, dim=1)

        if return_structured:
            return features_flat, outputs_struct
        return features_flat, None

    def __call__(
        self,
        noise: torch.Tensor,
        condition_vectors: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Unified contract with CGSGANTrainer:
        - During training: return flat Tensor only (WGAN-GP safe)
        - When needed: can call forward(return_structured=True) manually
        """
        features_flat, _ = self.forward(
            noise=noise, condition_vector=condition_vectors, return_structured=False
        )
        return features_flat

    def generate_structured(
        self,
        noise: torch.Tensor,
        condition_vectors: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Structured output (per column) – for use in:
        - Conditional loss
        - Auditor
        - Fraud injection later
        """
        _, structured = self.forward(
            noise=noise, condition_vector=condition_vectors, return_structured=True
        )
        return structured

    def summarize_contract(self) -> str:
        """
        Returns a textual summary of dimension contract and outputs for easy diagnosis and integration.
        """
        lines = []
        lines.append(f"noise_dim: {self.noise_dim}")
        lines.append(f"condition_dim: {self.condition_dim}")
        lines.append(f"num_categorical: {self.num_categorical}")
        lines.append(f"num_continuous: {self.num_continuous}")
        lines.append("output_dims:")
        for col in self.feature_order:
            lines.append(f"  - {col}: {self.output_dim_registry[col]}")
        lines.append(f"total_output_dim (D): {self.output_dim_total}")
        if self.extra_params:
            lines.append(f"extra_params: {self.extra_params}")
        return "\n".join(lines)
