"""
core/generative/attention/attention_engine.py
Production-ready multi-head, multi-scale attention engine with scalable hierarchical blocks.

Features:
- MultiHeadAttention: Efficient attention compatible with masks, supports multiple heads and clean separation of concerns.
- MultiScaleAttentionBlock: Long/short-term integration with pooling and stable scale fusion.
- HierarchicalAttention: Multi-level hierarchical attention with safe down/up-sampling paths.
- Flexible interfaces, diagnostic error handling, and complete type hints.

Copyright: Microsoft AI (Production-grade re-engineering)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= Errors =========


class AttentionError(ValueError):
    """Custom exception for attention module errors with precise diagnostic messages."""

    pass


# ========= Utilities =========


def _validate_heads(embed_dim: int, num_heads: int) -> Tuple[int, int]:
    """
    Validates that embed_dim is divisible by num_heads and returns head_dim and validations.
    Raises:
        AttentionError: If embed_dim or num_heads are invalid.
    """
    if embed_dim <= 0:
        raise AttentionError("embed_dim must be > 0.")
    if num_heads <= 0:
        raise AttentionError("num_heads must be > 0.")
    if embed_dim % num_heads != 0:
        raise AttentionError(
            f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}."
        )
    head_dim = embed_dim // num_heads
    return head_dim, num_heads


def _shape3(x: torch.Tensor, name: str) -> Tuple[int, int, int]:
    """Validates that input is [batch, seq_len, embed_dim] and returns the shape."""
    if x.dim() != 3:
        raise AttentionError(
            f"{name} must be of shape [batch, seq_len, embed_dim] but got: {tuple(x.shape)}"
        )
    return x.shape[0], x.shape[1], x.shape[2]


# ========= Multi-Head Attention =========


class MultiHeadAttention(nn.Module):
    """
    Production-ready multi-head attention, with separate Q/K/V projections, mask support, and stable residual connection.

    IO:
    - forward(query, key, value, mask=None) -> Tuple[Tensor, Tensor]
      Inputs shape [batch, seq_len, embed_dim], mask optional [batch, 1, 1, seq_len] or [batch, seq_len].

    Args:
        embed_dim: Embedding dimension of inputs.
        num_heads: Number of heads.
        dropout_p: Dropout probability for attention and residual outputs.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        head_dim, num_heads = _validate_heads(embed_dim, num_heads)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Q/K/V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout_p)
        self.resid_dropout = nn.Dropout(dropout_p)

        # Layer norm after residual connection
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes multi-head attention.

        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim]
            value: [batch, seq_len, embed_dim]
            mask: Optional, [batch, 1, 1, seq_len] or [batch, seq_len] (0=blocked, 1=allowed)

        Returns:
            Tuple[Tensor, Tensor]: (output: [batch, seq_len, embed_dim], attn_weights: [batch, heads, seq_len, seq_len])
        """
        bq, nq, dq = _shape3(query, "query")
        bk, nk, dk = _shape3(key, "key")
        bv, nv, dv = _shape3(value, "value")
        if not (dq == dk == dv == self.embed_dim):
            raise AttentionError("Input embed_dim does not match module embed_dim.")
        if not (nq == nk == nv):
            raise AttentionError("seq_len mismatch between query/key/value.")
        batch_size, seq_len = bq, nq

        # Projections
        Q = self.q_proj(query)  # [B, T, E]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Heads: [B, H, T, D]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # [B, H, T, T]

        # Apply mask if provided
        if mask is not None:
            # Support [B, T] by converting to [B, 1, 1, T]
            if mask.dim() == 2 and mask.shape == (batch_size, seq_len):
                mask = mask.unsqueeze(1).unsqueeze(1)
            if mask.dim() != 4 or mask.shape[-1] != seq_len:
                raise AttentionError(
                    f"Invalid mask shape: {tuple(mask.shape)}, expected [B, 1, 1, T]."
                )
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, D]

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # Final projection + residual + norm
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        output = self.layer_norm(query + attn_output)

        return output, attn_weights


# ========= Multi-Scale Attention Block =========


class MultiScaleAttentionBlock(nn.Module):
    """
    Multi-scale attention block combining long/short-term attention, stable scale fusion, and FFN.

    IO:
    - forward(x, attention_mask=None) -> Tuple[Tensor, Tensor]
      x: [batch, seq_len, embed_dim], fused_scales: [batch, embed_dim]

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout_p: Dropout probability.
    """

    def __init__(
        self, embed_dim: int, num_heads: int = 4, dropout_p: float = 0.1
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Long/short-range attention
        self.long_range_attention = MultiHeadAttention(embed_dim, num_heads, dropout_p)
        self.short_range_attention = MultiHeadAttention(embed_dim, num_heads, dropout_p)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_p),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        # Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # yields [B, E, 1]
        self.local_pool = nn.AdaptiveMaxPool1d(3)  # yields [B, E, 3]

        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes multi-scale attention, returns sequential output and a fused scale vector.

        Args:
            x: [batch, seq_len, embed_dim]
            attention_mask: Optional [batch, 1, 1, seq_len] or [batch, seq_len]

        Returns:
            Tuple[Tensor, Tensor]: (output: [batch, seq_len, embed_dim], fused_scales: [batch, embed_dim])
        """
        b, t, e = _shape3(x, "x")
        # Long and short attention
        long_out, _ = self.long_range_attention(x, x, x, attention_mask)
        short_out, _ = self.short_range_attention(x, x, x, attention_mask)

        # Long-range pooling: [B, E]
        long_pooled = self.global_pool(long_out.transpose(1, 2)).squeeze(-1)
        # Short-range pooling: [B, E]
        short_pooled_local = self.local_pool(short_out.transpose(1, 2))  # [B, E, 3]
        short_pooled = short_pooled_local.mean(dim=-1)

        # Scale fusion: [B, 2E] -> [B, E]
        fused = torch.cat([long_pooled, short_pooled], dim=-1)
        fused = self.scale_fusion(fused)

        # Enhance outputs
        enhanced = long_out + short_out
        enhanced = enhanced + x  # Residual connection

        # FFN
        ffn_in = self.ffn_norm(enhanced)
        ffn_out = self.ffn(ffn_in)
        output = enhanced + ffn_out

        return output, fused


# ========= Hierarchical Attention =========


class HierarchicalAttention(nn.Module):
    """
    Multi-level hierarchical attention with safe down/up-sampling paths.

    IO:
    - forward(x) -> Tensor [batch, seq_len, embed_dim_at_level0]

    Args:
        embed_dim: Dimension at top level.
        num_levels: Number of hierarchical levels.
        num_heads: Number of heads per level.
        dropout_p: Internal dropout for attention layers (optional via MultiHeadAttention).
    """

    def __init__(
        self,
        embed_dim: int,
        num_levels: int = 3,
        num_heads: int = 4,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if num_levels <= 0:
            raise AttentionError("num_levels must be > 0.")

        self.embed_dim = embed_dim
        self.num_levels = num_levels

        self.level_attn = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()

        # Create pyramid levels
        dim_at_level = embed_dim
        for i in range(num_levels):
            # Attention for the level
            self.level_attn.append(
                MultiHeadAttention(dim_at_level, num_heads, dropout_p)
            )

            # Setup down/up-sampling paths between levels
            if i < num_levels - 1:
                next_dim = max(1, dim_at_level // 2)
                self.downsample_layers.append(
                    nn.Conv1d(
                        dim_at_level, next_dim, kernel_size=3, stride=2, padding=1
                    )
                )
                self.upsample_layers.append(
                    nn.ConvTranspose1d(
                        next_dim,
                        dim_at_level,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    )
                )
                dim_at_level = next_dim  # Update dimension for next level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes hierarchical attention across multiple levels.

        Args:
            x: [batch, seq_len, embed_dim]

        Returns:
            Tensor: [batch, seq_len, embed_dim] at top level (0).
        """
        b, t, e = _shape3(x, "x")
        if e != self.embed_dim:
            raise AttentionError(
                f"Input embed_dim ({e}) does not match module embed_dim ({self.embed_dim})."
            )

        # Top-down: work with [B, E, T] representation for convolutional paths
        current = x.transpose(1, 2)  # [B, E, T]
        pyramid_feats: list[torch.Tensor] = []

        # Traverse levels downwards
        for idx, attn in enumerate(self.level_attn):
            level_input = current.transpose(1, 2)  # [B, T, E_level]
            attended, _ = attn(level_input, level_input, level_input)  # [B, T, E_level]
            pyramid_feats.append(attended.transpose(1, 2))  # [B, E_level, T]

            if idx < len(self.downsample_layers):
                current = self.downsample_layers[idx](current)  # [B, E_next, T/2]

        # Bottom-up: upsample and fuse
        for i in range(len(pyramid_feats) - 2, -1, -1):
            upsampled = self.upsample_layers[i](pyramid_feats[i + 1])  # [B, E_i, ~T]
            # Adjust temporal length if necessary (mismatch due to convolution)
            if upsampled.shape[-1] != pyramid_feats[i].shape[-1]:
                # Simple trim/extend for matching
                min_t = min(upsampled.shape[-1], pyramid_feats[i].shape[-1])
                upsampled = upsampled[..., :min_t]
                pyramid_feats[i] = pyramid_feats[i][..., :min_t]
            pyramid_feats[i] = (
                pyramid_feats[i] + upsampled
            )  # Residual connection across levels

        output = pyramid_feats[0].transpose(1, 2)  # [B, T, E0]
        return output


# ========= Alias for backward compatibility =========

MultiScaleAttention = MultiScaleAttentionBlock
