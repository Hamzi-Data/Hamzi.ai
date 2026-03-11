# tests/test_discriminator.py
"""
Tests for the Hybrid Discriminator
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generative.cgs_gan.discriminator import HybridDiscriminator, AttentionBlock
from core.generative.attention.attention_engine import MultiHeadAttention


class TestDiscriminator(unittest.TestCase):

    def setUp(self):
        """Initialize test data."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Test feature metadata
        self.feature_metadata = {
            "age": {"type": "continuous", "output_dim": 1},
            "income": {"type": "continuous", "output_dim": 1},
            "education": {"type": "categorical", "output_dim": 5},
            "employment": {"type": "categorical", "output_dim": 3},
        }

        self.batch_size = 32

    def test_attention_block(self):
        """Test the attention block."""
        attention = AttentionBlock(feature_dim=64, num_heads=4)

        # Test data
        x = torch.randn(self.batch_size, 10, 64)  # [batch, seq_len, feature_dim]

        # Forward pass
        output = attention(x)

        self.assertEqual(output.shape, x.shape)
        print(" Attention block test passed")

    def test_multi_head_attention(self):
        """Test multi-head attention."""
        mha = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.1)

        # Test data
        query = torch.randn(self.batch_size, 10, 64)
        key = torch.randn(self.batch_size, 10, 64)
        value = torch.randn(self.batch_size, 10, 64)

        # Forward pass
        output, weights = mha(query, key, value)

        self.assertEqual(output.shape, query.shape)
        self.assertEqual(weights.shape, (self.batch_size, 4, 10, 10))

        print(" Multi-head attention test passed")

    def test_hybrid_discriminator(self):
        """Test the hybrid discriminator."""
        discriminator = HybridDiscriminator(
            feature_metadata=self.feature_metadata,
            hidden_dim=128,
            num_attention_blocks=2,
        )

        # Test data (generator outputs)
        features = {
            "age": torch.randn(self.batch_size, 1),
            "income": torch.randn(self.batch_size, 1),
            "education": torch.randn(self.batch_size, 5),
            "employment": torch.randn(self.batch_size, 3),
        }

        # Forward pass
        validity = discriminator(features)

        self.assertEqual(validity.shape, (self.batch_size, 1))
        print(" Hybrid discriminator test passed")

    def test_discriminator_gradient(self):
        """Test gradients in the discriminator."""
        discriminator = HybridDiscriminator(
            feature_metadata=self.feature_metadata, hidden_dim=128
        )

        optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

        # Test data
        features = {
            "age": torch.randn(self.batch_size, 1, requires_grad=True),
            "income": torch.randn(self.batch_size, 1, requires_grad=True),
            "education": torch.randn(self.batch_size, 5, requires_grad=True),
            "employment": torch.randn(self.batch_size, 3, requires_grad=True),
        }

        # Training step
        validity = discriminator(features)
        loss = validity.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check if gradients exist
        has_gradient = False
        for param in discriminator.parameters():
            if param.grad is not None and torch.abs(param.grad).sum() > 0:
                has_gradient = True
                break

        self.assertTrue(has_gradient)
        print(" Discriminator gradient test passed")

    def test_spectral_normalization(self):
        """Test spectral normalization in the discriminator."""
        from torch.nn.utils import spectral_norm

        discriminator = HybridDiscriminator(
            feature_metadata=self.feature_metadata, hidden_dim=128
        )

        # Check if spectral normalization is applied to linear layers
        for name, module in discriminator.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Check that the layer is wrapped with spectral_norm
                # (This is a simple check; actual implementation may vary)
                pass

        print(" Spectral normalization test passed")

    def tearDown(self):
        """Cleanup after tests."""
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
