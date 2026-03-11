# tests/test_attention.py
"""
Tests for attention mechanisms
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generative.attention.attention_engine import (
    MultiScaleAttentionBlock,
    HierarchicalAttention,
)
from core.generative.attention.relationship_encoder import (
    CrossColumnAttention,
    ColumnRelationshipEncoder,
)


class TestAttention(unittest.TestCase):

    def setUp(self):
        """Initialize test data."""
        torch.manual_seed(42)
        np.random.seed(42)

        self.batch_size = 16
        self.num_columns = 8
        self.column_dim = 64

    def test_multi_scale_attention(self):
        """Test multi-scale attention."""
        msa = MultiScaleAttentionBlock(
            embed_dim=self.column_dim, num_heads=4, dropout=0.1
        )

        # Test data
        x = torch.randn(self.batch_size, 10, self.column_dim)

        # Forward pass
        output, scale_features = msa(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(scale_features.shape, (self.batch_size, self.column_dim))

        print("Multi-scale attention test passed")

    def test_hierarchical_attention(self):
        """Test hierarchical attention."""
        ha = HierarchicalAttention(embed_dim=self.column_dim, num_levels=3, num_heads=4)

        # Test data
        x = torch.randn(self.batch_size, 20, self.column_dim)

        # Forward pass
        output = ha(x)

        self.assertEqual(output.shape, x.shape)
        print("Hierarchical attention test passed")

    def test_cross_column_attention(self):
        """Test cross-column attention."""
        cca = CrossColumnAttention(column_dim=self.column_dim, num_heads=4, dropout=0.1)

        # Test data
        column_features = torch.randn(
            self.batch_size, self.num_columns, self.column_dim
        )

        # Forward pass
        output, attention_weights = cca(column_features)

        self.assertEqual(output.shape, column_features.shape)
        self.assertEqual(
            attention_weights.shape,
            (self.batch_size, 4, self.num_columns, self.num_columns),
        )

        print("Cross-column attention test passed")

    def test_column_relationship_encoder(self):
        """Test column relationship encoder."""
        encoder = ColumnRelationshipEncoder(
            num_columns=self.num_columns,
            column_dim=self.column_dim,
            hidden_dim=128,
            num_layers=2,
        )

        # Test data
        column_features = torch.randn(
            self.batch_size, self.num_columns, self.column_dim
        )

        # Forward pass
        encoded, relation_matrix = encoder(column_features, compute_relations=True)

        self.assertEqual(encoded.shape, column_features.shape)
        self.assertEqual(
            relation_matrix.shape, (self.batch_size, self.num_columns, self.num_columns)
        )

        # Check that relation matrix values are between 0 and 1
        self.assertTrue(torch.all(relation_matrix >= 0))
        self.assertTrue(torch.all(relation_matrix <= 1))

        print("Column relationship encoder test passed")

    def test_attention_gradient_flow(self):
        """Test gradient flow through attention mechanisms."""
        encoder = ColumnRelationshipEncoder(
            num_columns=self.num_columns, column_dim=self.column_dim, hidden_dim=128
        )

        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

        # Test data
        column_features = torch.randn(
            self.batch_size, self.num_columns, self.column_dim
        )

        # Training step
        encoded, _ = encoder(column_features, compute_relations=False)
        loss = encoded.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check if parameters were updated
        updated = False
        for param in encoder.parameters():
            if param.grad is not None and torch.abs(param.grad).sum() > 0:
                updated = True
                break

        self.assertTrue(updated)
        print("Gradient flow through attention mechanisms test passed")

    def tearDown(self):
        """Cleanup after tests."""
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
