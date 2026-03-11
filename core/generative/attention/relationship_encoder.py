"""
core/generative/attention/relationship_encoder.py
Production-grade encoder for inter-column relationships with multi-head attention, optional transformer, and graph model.

Features:
- CrossColumnAttention: Cross-attention between columns, supports attention masks, residual connection, and stable layer norm.
- ColumnRelationshipEncoder: Sequential attention layers + FFN + optional TransformerEncoder + pairwise relation matrix.
- GraphBasedRelationModel: Simple graph-based model to enhance representation using a learnable adjacency matrix.
- Flexible interfaces, complete type hints, and precise diagnostic error handling.

Copyright: Microsoft AI (Production-grade re-engineering)
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= Errors =========


class RelationshipError(ValueError):
    """Custom exception for relationship encoder errors with precise diagnostic messages."""

    pass


# ========= Utilities =========


def _shape3(x: torch.Tensor, name: str) -> Tuple[int, int, int]:
    """
    Validates that input is [batch, num_columns, column_dim] and returns the shape.

    Args:
        x: Tensor
        name: Input name for diagnostics

    Raises:
        RelationshipError: If dimension is not three.

    Returns:
        (batch, num_columns, column_dim)
    """
    if x.dim() != 3:
        raise RelationshipError(
            f"{name} must be of shape [batch, num_columns, column_dim] but got: {tuple(x.shape)}"
        )
    return x.shape[0], x.shape[1], x.shape[2]


def _validate_heads(column_dim: int, num_heads: int) -> Tuple[int, int]:
    """
    Validates head count and returns head_dim.

    Raises:
        RelationshipError: If column_dim is not divisible by num_heads or values are invalid.
    """
    if column_dim <= 0:
        raise RelationshipError("column_dim must be > 0.")
    if num_heads <= 0:
        raise RelationshipError("num_heads must be > 0.")
    if column_dim % num_heads != 0:
        raise RelationshipError(
            f"column_dim={column_dim} is not divisible by num_heads={num_heads}."
        )
    return column_dim // num_heads, num_heads


# ========= Cross-Column Attention =========


class CrossColumnAttention(nn.Module):
    """
    Cross-attention between columns to understand inter-column relationships.

    IO:
    - forward(column_features, attention_mask=None) -> Tuple[Tensor, Tensor]
      column_features: [batch, num_columns, column_dim]
      attention_mask: optional [batch, 1, 1, num_columns] or [batch, num_columns] (0=blocked, 1=allowed)

    Args:
        column_dim: Dimension per column.
        num_heads: Number of attention heads.
        dropout_p: Dropout probability.
    """

    def __init__(
        self, column_dim: int, num_heads: int = 4, dropout_p: float = 0.1
    ) -> None:
        super().__init__()
        head_dim, num_heads = _validate_heads(column_dim, num_heads)
        self.column_dim = column_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Q/K/V projections: project to [num_heads * head_dim] = column_dim
        self.query_proj = nn.Linear(column_dim, column_dim)
        self.key_proj = nn.Linear(column_dim, column_dim)
        self.value_proj = nn.Linear(column_dim, column_dim)

        # Output projection
        self.output_proj = nn.Linear(column_dim, column_dim)

        # Regularization
        self.dropout = nn.Dropout(dropout_p)
        self.layer_norm = nn.LayerNorm(column_dim)

    def forward(
        self,
        column_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes cross-attention between columns.

        Returns:
            Tuple[Tensor, Tensor]: (output: [batch, num_columns, column_dim], attn_weights: [batch, heads, num_columns, num_columns])
        """
        batch_size, num_columns, cdim = _shape3(column_features, "column_features")
        if cdim != self.column_dim:
            raise RelationshipError(
                f"Input column_dim ({cdim}) does not match expected ({self.column_dim})."
            )

        # Projections
        Q = self.query_proj(column_features)
        K = self.key_proj(column_features)
        V = self.value_proj(column_features)

        # To heads: [B, H, C, D]
        Q = Q.view(batch_size, num_columns, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        K = K.view(batch_size, num_columns, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, num_columns, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        # Attention scores: [B, H, C, C]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Optional attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.shape == (
                batch_size,
                num_columns,
            ):
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,C]
            if attention_mask.dim() != 4 or attention_mask.shape[-1] != num_columns:
                raise RelationshipError(
                    f"Invalid mask shape: {tuple(attention_mask.shape)}, expected [B, 1, 1, num_columns]."
                )
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention: [B, H, C, D]
        attended = torch.matmul(attn_weights, V)

        # Concatenate heads and return to column dimensions: [B, C, E]
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_columns, self.column_dim)
        )

        # Final projection + residual + norm
        out = self.output_proj(attended)
        out = self.dropout(out)
        out = self.layer_norm(column_features + out)

        return out, attn_weights


# ========= Column Relationship Encoder =========


class ColumnRelationshipEncoder(nn.Module):
    """
    Advanced encoder for inter-column relationships with cross-attention layers, FFN, and optional TransformerEncoder for relation encoding.

    IO:
    - forward(column_features, compute_relations=False) -> Tuple[Tensor, Optional[Tensor]]
      column_features: [batch, num_columns, column_dim]
      returns:
        encoded_relations: [batch, num_columns, column_dim]
        relation_matrix: optional [batch, num_columns, num_columns] if compute_relations=True

    Args:
        num_columns: Number of columns.
        column_dim: Column representation dimension.
        hidden_dim: Hidden dimension for FFN/Transformer.
        num_layers: Number of (Attention+FFN) layers.
        num_heads: Number of attention heads per layer.
        use_transformer: Enable relation encoding via TransformerEncoder.
        dropout_p: Dropout probability.
    """

    def __init__(
        self,
        num_columns: int,
        column_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        use_transformer: bool = True,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        if num_columns <= 0 or column_dim <= 0:
            raise RelationshipError("num_columns and column_dim must be > 0.")
        if num_layers <= 0:
            raise RelationshipError("num_layers must be > 0.")

        self.num_columns = num_columns
        self.column_dim = column_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_transformer = use_transformer
        self.dropout_p = dropout_p

        # Learnable column embeddings (positional-like for columns)
        self.column_embeddings = nn.Parameter(torch.randn(1, num_columns, column_dim))

        # Cross-attention layers + FFN with norm-first architecture
        self.cross_attention_layers = nn.ModuleList(
            [
                CrossColumnAttention(
                    column_dim=column_dim, num_heads=num_heads, dropout_p=dropout_p
                )
                for _ in range(num_layers)
            ]
        )
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(column_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, column_dim),
                    nn.Dropout(dropout_p),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_attn = nn.ModuleList(
            [nn.LayerNorm(column_dim) for _ in range(num_layers)]
        )
        self.norm_ffn = nn.ModuleList(
            [nn.LayerNorm(column_dim) for _ in range(num_layers)]
        )

        # Optional TransformerEncoder for relation encoding across columns
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=column_dim,
                nhead=num_heads,
                dim_feedforward=max(hidden_dim, 128),
                dropout=dropout_p,
                activation="gelu",
                batch_first=False,
                norm_first=True,
            )
            self.relationship_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=2
            )
        else:
            self.relationship_transformer = None

        # Pairwise relation score model: takes concatenated pair [2 * column_dim] -> score
        self.relation_graph = nn.Sequential(
            nn.Linear(column_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        column_features: torch.Tensor,
        compute_relations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes relationships between columns and returns the encoded representation and optional relation matrix.
        """
        batch_size, num_columns, embed_dim = _shape3(column_features, "column_features")
        if num_columns != self.num_columns or embed_dim != self.column_dim:
            raise RelationshipError(
                f"Dimension mismatch: input ({num_columns}, {embed_dim}) does not match expected ({self.num_columns}, {self.column_dim})."
            )

        # Add column embeddings
        x = column_features + self.column_embeddings.expand(batch_size, -1, -1)

        # Cross-attention + FFN layers with residual connections
        for i in range(self.num_layers):
            # Attention (norm-first)
            attn_in = self.norm_attn[i](x)
            attn_out, _ = self.cross_attention_layers[i](attn_in)
            x = x + attn_out

            # FFN (norm-first)
            ffn_in = self.norm_ffn[i](x)
            ffn_out = self.ffn_layers[i](ffn_in)
            x = x + ffn_out

        encoded = x  # [B, C, E]

        # Relation encoding via TransformerEncoder (optional)
        if self.relationship_transformer is not None:
            seq_first = encoded.transpose(0, 1)  # [C, B, E]
            encoded_rel = self.relationship_transformer(seq_first)  # same dimensions
            encoded_rel = encoded_rel.transpose(0, 1)  # [B, C, E]
        else:
            encoded_rel = encoded

        relation_matrix: Optional[torch.Tensor] = None
        if compute_relations:
            relation_matrix = self._compute_relation_matrix(encoded_rel)

        return encoded_rel, relation_matrix

    def _compute_relation_matrix(self, column_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise relation matrix between columns using a small network.

        Args:
            column_embeddings: [batch, num_columns, column_dim]

        Returns:
            Tensor: [batch, num_columns, num_columns] values between 0 and 1.
        """
        batch_size, num_columns, embed_dim = _shape3(
            column_embeddings, "column_embeddings"
        )
        if num_columns != self.num_columns or embed_dim != self.column_dim:
            raise RelationshipError(
                "column_embeddings dimensions do not match encoder configuration."
            )

        # Expand dimensions for pairwise computation
        col_i = column_embeddings.unsqueeze(2).expand(
            -1, -1, num_columns, -1
        )  # [B, C, C, E]
        col_j = column_embeddings.unsqueeze(1).expand(
            -1, num_columns, -1, -1
        )  # [B, C, C, E]

        # Concatenate pair: [B*C*C, 2E] -> score
        pairs = torch.cat([col_i, col_j], dim=-1).contiguous().view(-1, embed_dim * 2)
        relation_scores = self.relation_graph(pairs).view(
            batch_size, num_columns, num_columns
        )

        return relation_scores


# ========= Graph-Based Relation Model =========


class GraphBasedRelationModel(nn.Module):
    """
    Graph-based model for column relationships, with learnable adjacency matrix and simple aggregation.

    IO:
    - forward(column_features) -> Tuple[Tensor, Tensor, Tensor]
      column_features: [batch, num_columns, column_dim]
      returns:
        refined_features: [batch, num_columns, column_dim]
        pooled: [batch, column_dim]
        adjacency_normalized: [num_columns, num_columns]

    Args:
        num_columns: Number of columns.
        column_dim: Column representation dimension.
        hidden_dim: Hidden dimension for transformation layers.
    """

    def __init__(
        self, num_columns: int, column_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        if num_columns <= 0 or column_dim <= 0:
            raise RelationshipError("num_columns and column_dim must be > 0.")

        # Simple transformation layers (MLP) applied after aggregating neighbor information
        self.gnn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(column_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, column_dim),
                )
                for _ in range(3)
            ]
        )

        # Learnable adjacency matrix
        self.adjacency = nn.Parameter(torch.randn(num_columns, num_columns))
        self.adjacency_norm = nn.Softmax(dim=-1)

        # Graph pooling to vector: [B, E]
        self.graph_pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, column_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs simple graph propagation via learnable adjacency matrix and aggregates results.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (refined_features, pooled, adjacency_normalized)
        """
        batch_size, num_columns, column_dim = _shape3(
            column_features, "column_features"
        )

        # Normalize adjacency matrix: [C, C]
        adj_norm = self.adjacency_norm(self.adjacency)

        x = column_features  # [B, C, E]
        for gnn in self.gnn_layers:
            # Aggregate neighbor information: [B, C, E]
            neighbor_info = torch.matmul(
                adj_norm, x
            )  # broadcasts over batch dimension automatically
            # Transform via MLP per column
            transformed = gnn(neighbor_info)
            # Residual connection
            x = x + transformed

        # Graph pooling per sample: [B, E]
        pooled = self.graph_pool(x.transpose(1, 2)).squeeze(-1)

        return x, pooled, adj_norm
