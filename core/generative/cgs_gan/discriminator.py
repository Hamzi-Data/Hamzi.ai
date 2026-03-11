# core/generative/cgs_gan/discriminator.py
"""
Production-Grade Multi-Scale Hybrid Discriminator for Titan Synthetic Data Platform
High-Performance implementation for 10M record pipeline

Features:
- Multi-scale architecture for simultaneous continuous/categorical discrimination
- Spectral normalization for training stability
- Feature-aware processing with schema validation
- Optional self-attention for complex feature relationships
- Gradient penalty support for WGAN-GP training
- Full compatibility with hybrid_engine.py, hybrid_engine2.py, and trainer.py

Architecture:
- Separate processing pathways for different feature types
- Multi-scale feature extraction at different resolutions
- Cross-feature attention mechanisms
- Ensemble-based final prediction

Enterprise-grade with strict type safety, comprehensive validation, and zero-redundancy.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logger = logging.getLogger(__name__)


# ================== Configuration & Types ==================
class FeatureType(Enum):
    """Feature type enumeration for type-safe processing."""

    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    CONTINUOUS = "continuous"
    NUMERICAL = "numerical"
    BINARY = "binary"
    BOOLEAN = "boolean"
    MULTIMODAL = "multimodal"

    @classmethod
    def is_categorical(cls, ftype: str) -> bool:
        """Check if feature type is categorical-like."""
        ftype_lower = ftype.lower()
        return ftype_lower in {
            cls.CATEGORICAL.value,
            cls.ORDINAL.value,
            cls.BINARY.value,
            cls.BOOLEAN.value,
        }

    @classmethod
    def is_continuous(cls, ftype: str) -> bool:
        """Check if feature type is continuous-like."""
        ftype_lower = ftype.lower()
        return ftype_lower in {
            cls.CONTINUOUS.value,
            cls.NUMERICAL.value,
            cls.MULTIMODAL.value,
        }


@dataclass
class DiscriminatorConfig:
    """Configuration for HybridDiscriminator."""

    input_dim: int
    hidden_dim: int = 256
    num_scales: int = 3
    num_attention_blocks: int = 1
    attention_heads: int = 4
    dropout_p: float = 0.2
    use_spectral_norm: bool = True
    use_gradient_penalty: bool = True
    activation: str = "leaky_relu"
    feature_aware: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {self.num_scales}")
        if self.num_attention_blocks < 0:
            raise ValueError(f"num_attention_blocks must be non-negative")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {self.dropout_p}")


# ================== Custom Exceptions ==================
class DiscriminatorError(RuntimeError):
    """Base exception for Discriminator errors."""

    pass


class DimensionMismatchError(DiscriminatorError):
    """Raised when tensor dimensions don't match expected shapes."""

    pass


class ConfigurationError(DiscriminatorError):
    """Raised when configuration is invalid."""

    pass


# ================== Utility Functions ==================
def _spectral_norm(module: nn.Module, use_sn: bool = True) -> nn.Module:
    """
    Apply spectral normalization to a module if enabled.

    Args:
        module: PyTorch module (typically Linear or Conv)
        use_sn: Whether to apply spectral normalization

    Returns:
        Module with optional spectral normalization
    """
    if use_sn and isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return nn.utils.spectral_norm(module)
    return module


def _get_activation(name: str) -> nn.Module:
    """
    Factory function for activation layers.

    Args:
        name: Activation function name

    Returns:
        PyTorch activation module
    """
    activations = {
        "leaky_relu": nn.LeakyReLU(0.2),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name.lower()]


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP training.

    Args:
        discriminator: Discriminator model
        real_data: Real data samples [B, D]
        fake_data: Fake data samples [B, D]
        lambda_gp: Gradient penalty coefficient

    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    device = real_data.device

    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, device=device)

    # Interpolated samples
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)

    # Get discriminator output
    d_interpolates = discriminator(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


# ================== Attention Mechanisms ==================
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for feature relationship modeling.
    Operates on flattened feature representations.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout_p: float = 0.1,
        use_spectral_norm: bool = True,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = max(1, feature_dim // num_heads)
        self.scale = self.head_dim**-0.5

        # Q, K, V projections
        self.query = _spectral_norm(
            nn.Linear(feature_dim, feature_dim), use_spectral_norm
        )
        self.key = _spectral_norm(
            nn.Linear(feature_dim, feature_dim), use_spectral_norm
        )
        self.value = _spectral_norm(
            nn.Linear(feature_dim, feature_dim), use_spectral_norm
        )

        # Output projection
        self.out_proj = _spectral_norm(
            nn.Linear(feature_dim, feature_dim), use_spectral_norm
        )

        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor [B, D]

        Returns:
            Attended features [B, D]
        """
        if x.dim() != 2:
            raise DimensionMismatchError(
                f"Expected 2D input [B, D], got shape {x.shape}"
            )

        batch_size, feature_dim = x.shape

        # Add sequence dimension: [B, 1, D]
        x_seq = x.unsqueeze(1)

        # Compute Q, K, V
        Q = self.query(x_seq).view(batch_size, 1, self.num_heads, self.head_dim)
        K = self.key(x_seq).view(batch_size, 1, self.num_heads, self.head_dim)
        V = self.value(x_seq).view(batch_size, 1, self.num_heads, self.head_dim)

        # Transpose for attention: [B, num_heads, 1, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        out = torch.matmul(attention, V)  # [B, num_heads, 1, head_dim]

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, feature_dim)
        out = self.out_proj(out).squeeze(1)  # [B, D]

        # Residual connection + layer norm
        out = self.layer_norm(x + self.dropout(out))

        return out


# ================== Multi-Scale Processing ==================
class MultiScaleProcessor(nn.Module):
    """
    Multi-scale feature processing for capturing patterns at different resolutions.
    Uses parallel pathways with different receptive fields.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_scales: int = 3,
        use_spectral_norm: bool = True,
        activation: str = "leaky_relu",
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Create parallel processing paths at different scales
        self.scale_paths = nn.ModuleList()

        for scale_idx in range(num_scales):
            # Different hidden dimensions for each scale
            scale_hidden = hidden_dim // (2**scale_idx)
            scale_hidden = max(32, scale_hidden)  # Minimum 32

            path = nn.Sequential(
                _spectral_norm(nn.Linear(input_dim, scale_hidden), use_spectral_norm),
                _get_activation(activation),
                nn.Dropout(dropout_p),
                _spectral_norm(
                    nn.Linear(scale_hidden, scale_hidden), use_spectral_norm
                ),
                _get_activation(activation),
            )
            self.scale_paths.append(path)

        # Fusion layer
        total_scale_dim = sum(max(32, hidden_dim // (2**i)) for i in range(num_scales))

        self.fusion = nn.Sequential(
            _spectral_norm(nn.Linear(total_scale_dim, hidden_dim), use_spectral_norm),
            nn.LayerNorm(hidden_dim),
            _get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through multiple scales.

        Args:
            x: Input tensor [B, D]

        Returns:
            Multi-scale features [B, hidden_dim]
        """
        scale_outputs = []

        for path in self.scale_paths:
            scale_out = path(x)
            scale_outputs.append(scale_out)

        # Concatenate all scales
        combined = torch.cat(scale_outputs, dim=1)

        # Fuse scales
        fused = self.fusion(combined)

        return fused


# ================== Feature-Aware Processing ==================
class FeatureAwareDiscriminator(nn.Module):
    """
    Feature-aware discriminator that processes different feature types
    through specialized pathways before combining them.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        use_spectral_norm: bool = True,
        activation: str = "leaky_relu",
        dropout_p: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_metadata = feature_metadata or {}

        # Analyze feature types
        self.num_categorical = 0
        self.num_continuous = 0

        for meta in self.feature_metadata.values():
            ftype = meta.get("type", "continuous")
            if FeatureType.is_categorical(ftype):
                self.num_categorical += 1
            elif FeatureType.is_continuous(ftype):
                self.num_continuous += 1

        total_features = self.num_categorical + self.num_continuous

        # If we have type information, create specialized pathways
        if total_features > 0:
            # Categorical pathway
            if self.num_categorical > 0:
                cat_ratio = self.num_categorical / total_features
                cat_hidden = max(32, int(hidden_dim * cat_ratio))

                self.categorical_path = nn.Sequential(
                    _spectral_norm(nn.Linear(input_dim, cat_hidden), use_spectral_norm),
                    _get_activation(activation),
                    nn.Dropout(dropout_p),
                )
            else:
                self.categorical_path = None

            # Continuous pathway
            if self.num_continuous > 0:
                cont_ratio = self.num_continuous / total_features
                cont_hidden = max(32, int(hidden_dim * cont_ratio))

                self.continuous_path = nn.Sequential(
                    _spectral_norm(
                        nn.Linear(input_dim, cont_hidden), use_spectral_norm
                    ),
                    _get_activation(activation),
                    nn.Dropout(dropout_p),
                )
            else:
                self.continuous_path = None

            # Fusion layer
            fusion_input_dim = 0
            if self.categorical_path:
                fusion_input_dim += cat_hidden
            if self.continuous_path:
                fusion_input_dim += cont_hidden

            self.fusion = nn.Sequential(
                _spectral_norm(
                    nn.Linear(fusion_input_dim, hidden_dim), use_spectral_norm
                ),
                nn.LayerNorm(hidden_dim),
                _get_activation(activation),
            )
        else:
            # Fallback: unified pathway
            self.categorical_path = None
            self.continuous_path = None
            self.fusion = nn.Sequential(
                _spectral_norm(nn.Linear(input_dim, hidden_dim), use_spectral_norm),
                _get_activation(activation),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through type-aware pathways.

        Args:
            x: Input tensor [B, D]

        Returns:
            Processed features [B, hidden_dim]
        """
        if self.categorical_path is None and self.continuous_path is None:
            # Unified pathway
            return self.fusion(x)

        # Type-aware processing
        outputs = []

        if self.categorical_path is not None:
            cat_out = self.categorical_path(x)
            outputs.append(cat_out)

        if self.continuous_path is not None:
            cont_out = self.continuous_path(x)
            outputs.append(cont_out)

        # Fuse pathways
        combined = torch.cat(outputs, dim=1)
        fused = self.fusion(combined)

        return fused


# ================== Main Discriminator ==================
class HybridDiscriminator(nn.Module):
    """
    Production-Grade Multi-Scale Hybrid Discriminator for Titan Platform.

    Features:
    - Multi-scale architecture for pattern detection at different resolutions
    - Feature-aware processing for categorical/continuous features
    - Optional self-attention for complex relationships
    - Spectral normalization for training stability
    - Gradient penalty support for WGAN-GP
    - Comprehensive validation and error handling
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 256,
        num_scales: int = 3,
        num_attention_blocks: int = 1,
        attention_heads: int = 4,
        dropout_p: float = 0.2,
        use_spectral_norm: bool = True,
        activation: str = "leaky_relu",
        feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        feature_aware: bool = True,
        **kwargs,
    ):
        """
        Initialize HybridDiscriminator.

        Args:
            input_dim: Input feature dimension (must match encoder output)
            hidden_dim: Hidden layer dimension
            num_scales: Number of scales for multi-scale processing
            num_attention_blocks: Number of self-attention blocks
            attention_heads: Number of attention heads
            dropout_p: Dropout probability
            use_spectral_norm: Whether to use spectral normalization
            activation: Activation function name
            feature_metadata: Feature schema metadata
            feature_aware: Whether to use feature-aware processing
            **kwargs: Additional arguments
        """
        super().__init__()

        # ============================
        # Validation
        # ============================
        if input_dim is None:
            raise ConfigurationError(
                "input_dim must be explicitly provided and match encoder output_dim"
            )

        self.config = DiscriminatorConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_scales=num_scales,
            num_attention_blocks=num_attention_blocks,
            attention_heads=attention_heads,
            dropout_p=dropout_p,
            use_spectral_norm=use_spectral_norm,
            activation=activation,
            feature_aware=feature_aware,
        )

        self.feature_metadata = feature_metadata or {}
        self.extra_params = kwargs

        # ============================
        # Architecture Components
        # ============================

        # 1. Multi-Scale Processing
        self.multi_scale = MultiScaleProcessor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_scales=self.config.num_scales,
            use_spectral_norm=self.config.use_spectral_norm,
            activation=self.config.activation,
            dropout_p=self.config.dropout_p,
        )

        # 2. Feature-Aware Processing (Optional)
        if self.config.feature_aware and self.feature_metadata:
            self.feature_aware_processor = FeatureAwareDiscriminator(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                feature_metadata=self.feature_metadata,
                use_spectral_norm=self.config.use_spectral_norm,
                activation=self.config.activation,
                dropout_p=self.config.dropout_p,
            )
        else:
            self.feature_aware_processor = None

        # 3. Self-Attention Blocks
        self.attention_blocks = nn.ModuleList()
        for _ in range(self.config.num_attention_blocks):
            self.attention_blocks.append(
                MultiHeadSelfAttention(
                    feature_dim=self.config.hidden_dim,
                    num_heads=self.config.attention_heads,
                    dropout_p=self.config.dropout_p,
                    use_spectral_norm=self.config.use_spectral_norm,
                )
            )

        # 4. Deep Processing Layers
        self.deep_layers = nn.Sequential(
            _spectral_norm(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                self.config.use_spectral_norm,
            ),
            nn.LayerNorm(self.config.hidden_dim),
            _get_activation(self.config.activation),
            nn.Dropout(self.config.dropout_p),
            _spectral_norm(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                self.config.use_spectral_norm,
            ),
            nn.LayerNorm(self.config.hidden_dim // 2),
            _get_activation(self.config.activation),
            nn.Dropout(self.config.dropout_p),
        )

        # 5. Output Layer
        self.output_layer = _spectral_norm(
            nn.Linear(self.config.hidden_dim // 2, 1), self.config.use_spectral_norm
        )

        logger.info(
            f"HybridDiscriminator initialized: input_dim={self.config.input_dim}, "
            f"hidden_dim={self.config.hidden_dim}, num_scales={self.config.num_scales}"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with comprehensive validation.

        Args:
            features: Input feature tensor [B, D]

        Returns:
            Discriminator scores [B, 1]

        Raises:
            DimensionMismatchError: If input dimensions don't match
        """
        # =====================================================
        # Input Validation
        # =====================================================
        if features.dim() != 2:
            raise DimensionMismatchError(
                f"Expected 2D input [B, D], got shape {features.shape}"
            )

        if features.shape[1] != self.config.input_dim:
            raise DimensionMismatchError(
                f"Input dimension {features.shape[1]} doesn't match "
                f"expected {self.config.input_dim}"
            )

        batch_size = features.shape[0]

        # =====================================================
        # Multi-Scale Processing
        # =====================================================
        x = self.multi_scale(features)

        # =====================================================
        # Feature-Aware Processing (if enabled)
        # =====================================================
        if self.feature_aware_processor is not None:
            x_aware = self.feature_aware_processor(features)
            # Combine multi-scale and feature-aware
            x = (x + x_aware) / 2.0

        # =====================================================
        # Self-Attention Blocks
        # =====================================================
        for attention_block in self.attention_blocks:
            x = attention_block(x)

        # =====================================================
        # Deep Processing
        # =====================================================
        x = self.deep_layers(x)

        # =====================================================
        # Output
        # =====================================================
        output = self.output_layer(x)

        # =====================================================
        # Validation
        # =====================================================
        if output.shape != (batch_size, 1):
            raise DimensionMismatchError(
                f"Invalid output shape: {output.shape}, expected [{batch_size}, 1]"
            )

        return output

    def compute_loss(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        loss_type: str = "wgan-gp",
        lambda_gp: float = 10.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute discriminator loss with optional gradient penalty.

        Args:
            real_data: Real data samples [B, D]
            fake_data: Fake data samples [B, D]
            loss_type: Loss type ('wgan-gp', 'vanilla', 'hinge')
            lambda_gp: Gradient penalty coefficient

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Get predictions
        real_pred = self(real_data)
        fake_pred = self(fake_data)

        metrics = {}

        # Compute base loss
        if loss_type == "wgan-gp":
            # Wasserstein loss
            d_loss = fake_pred.mean() - real_pred.mean()

            # Gradient penalty
            gp = compute_gradient_penalty(self, real_data, fake_data, lambda_gp)
            total_loss = d_loss + gp

            metrics["d_loss_wasserstein"] = d_loss.item()
            metrics["gradient_penalty"] = gp.item()

        elif loss_type == "vanilla":
            # Standard GAN loss
            real_loss = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )
            total_loss = (real_loss + fake_loss) / 2

            metrics["d_loss_real"] = real_loss.item()
            metrics["d_loss_fake"] = fake_loss.item()

        elif loss_type == "hinge":
            # Hinge loss
            real_loss = F.relu(1.0 - real_pred).mean()
            fake_loss = F.relu(1.0 + fake_pred).mean()
            total_loss = real_loss + fake_loss

            metrics["d_loss_real"] = real_loss.item()
            metrics["d_loss_fake"] = fake_loss.item()

        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Additional metrics
        metrics["d_loss_total"] = total_loss.item()
        metrics["real_score"] = real_pred.mean().item()
        metrics["fake_score"] = fake_pred.mean().item()

        return total_loss, metrics

    def get_config(self) -> Dict[str, Any]:
        """Get discriminator configuration."""
        return {
            "input_dim": self.config.input_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_scales": self.config.num_scales,
            "num_attention_blocks": self.config.num_attention_blocks,
            "attention_heads": self.config.attention_heads,
            "dropout_p": self.config.dropout_p,
            "use_spectral_norm": self.config.use_spectral_norm,
            "activation": self.config.activation,
            "feature_aware": self.config.feature_aware,
        }

    def summarize_contract(self) -> str:
        """Get human-readable contract summary."""
        return "\n".join(
            [
                "=" * 60,
                "HybridDiscriminator Contract",
                "=" * 60,
                f"Input Dimension: {self.config.input_dim}",
                f"Hidden Dimension: {self.config.hidden_dim}",
                f"Number of Scales: {self.config.num_scales}",
                f"Attention Blocks: {self.config.num_attention_blocks}",
                f"Attention Heads: {self.config.attention_heads}",
                f"Dropout: {self.config.dropout_p}",
                f"Spectral Norm: {self.config.use_spectral_norm}",
                f"Feature Aware: {self.config.feature_aware}",
                f"Output Shape: [batch_size, 1]",
                "=" * 60,
            ]
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"HybridDiscriminator(\n"
            f"  input_dim={self.config.input_dim},\n"
            f"  hidden_dim={self.config.hidden_dim},\n"
            f"  num_scales={self.config.num_scales},\n"
            f"  attention_blocks={self.config.num_attention_blocks},\n"
            f"  feature_aware={self.config.feature_aware}\n"
            f")"
        )


# ================== Factory ==================
class DiscriminatorFactory:
    """
    Factory for creating discriminator instances with consistent configuration.
    Supports dependency injection and centralized configuration management.
    """

    def __init__(
        self,
        default_hidden_dim: int = 256,
        default_num_scales: int = 3,
        default_num_attention_blocks: int = 1,
        default_dropout_p: float = 0.2,
        default_use_spectral_norm: bool = True,
        default_activation: str = "leaky_relu",
    ):
        """Initialize factory with default parameters."""
        self.defaults = {
            "hidden_dim": default_hidden_dim,
            "num_scales": default_num_scales,
            "num_attention_blocks": default_num_attention_blocks,
            "dropout_p": default_dropout_p,
            "use_spectral_norm": default_use_spectral_norm,
            "activation": default_activation,
        }

    def build(
        self,
        input_dim: int,
        feature_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **overrides,
    ) -> HybridDiscriminator:
        """
        Build discriminator with optional parameter overrides.

        Args:
            input_dim: Input feature dimension
            feature_metadata: Feature schema metadata
            **overrides: Parameter overrides

        Returns:
            Initialized HybridDiscriminator instance
        """
        config = {**self.defaults, **overrides}

        return HybridDiscriminator(
            input_dim=input_dim,
            feature_metadata=feature_metadata,
            **config,
        )

    def from_config(
        self, config: Union[DiscriminatorConfig, Dict[str, Any]]
    ) -> HybridDiscriminator:
        """
        Build discriminator from configuration object.

        Args:
            config: DiscriminatorConfig or dict

        Returns:
            Initialized HybridDiscriminator instance
        """
        # if isinstance(config, dict):
