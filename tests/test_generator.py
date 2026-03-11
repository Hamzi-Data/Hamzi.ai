# tests/test_generator.py
"""
Hybrid Generator Tests
"""

import unittest
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generative.cgs_gan.generator import ConditionalGenerator, GumbelSoftmaxLayer
from core.generative.cgs_gan.conditional_encoder import ConditionalEncoder


class TestGenerator(unittest.TestCase):

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

        self.noise_dim = 64
        self.batch_size = 32

    def test_gumbel_softmax_layer(self):
        """Test Gumbel-Softmax layer."""
        layer = GumbelSoftmaxLayer(temperature=0.5, hard=True)

        # Test data
        logits = torch.randn(10, 5)

        # In training mode
        layer.train()
        output_train = layer(logits)

        self.assertEqual(output_train.shape, (10, 5))
        self.assertTrue(torch.all(output_train.sum(dim=1) - 1.0 < 1e-6))

        # In inference mode
        layer.eval()
        output_eval = layer(logits)

        self.assertEqual(output_eval.shape, (10, 5))
        self.assertTrue(torch.all(output_eval.sum(dim=1) - 1.0 < 1e-6))

        print("Gumbel-Softmax test passed")

    def test_conditional_generator_forward(self):
        """Test generator forward pass."""
        generator = ConditionalGenerator(
            noise_dim=self.noise_dim,
            feature_metadata=self.feature_metadata,
            hidden_dim=128,
        )

        # Input data
        noise = torch.randn(self.batch_size, self.noise_dim)
        condition_vector = torch.randn(self.batch_size, 128)  # after conditioning

        # Forward pass
        outputs = generator(noise, condition_vector)

        # Check outputs
        self.assertIn("age", outputs)
        self.assertIn("education", outputs)

        self.assertEqual(outputs["age"].shape, (self.batch_size, 2))  # mean, log_std
        self.assertEqual(outputs["education"].shape, (self.batch_size, 5))

        print("Generator forward pass test passed")

    def test_conditional_encoder(self):
        """Test conditional encoder."""
        encoder = ConditionalEncoder(
            feature_metadata=self.feature_metadata, embedding_dim=32
        )

        # Test conditions
        conditions = {
            "education": torch.randint(0, 5, (self.batch_size,)),
            "employment": torch.randint(0, 3, (self.batch_size,)),
            "age": torch.randn(self.batch_size, 1),
            "income": torch.randn(self.batch_size, 1),
        }

        # Encoding
        encoded = encoder(conditions)

        self.assertIsNotNone(encoded)
        self.assertEqual(encoded.shape, (self.batch_size, 256))  # after fusion

        print("Conditional encoder test passed")

    def test_generator_training(self):
        """Test generator training (backward pass)."""
        generator = ConditionalGenerator(
            noise_dim=self.noise_dim,
            feature_metadata=self.feature_metadata,
            hidden_dim=128,
        )

        optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)

        # Fake training step
        noise = torch.randn(self.batch_size, self.noise_dim, requires_grad=True)
        condition_vector = torch.randn(self.batch_size, 128, requires_grad=True)

        outputs = generator(noise, condition_vector)

        # Fake loss
        loss = outputs["age"].mean() + outputs["education"].mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters were updated
        for param in generator.parameters():
            self.assertTrue(param.grad is not None)

        print("Generator training (backward pass) test passed")

    def tearDown(self):
        """Cleanup after tests."""
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
