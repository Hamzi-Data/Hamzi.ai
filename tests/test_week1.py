"""
tests/test_week1.py
Unit tests for week 1
"""

import unittest
import pandas as pd
import numpy as np
from core.integration.main_engine import CoreDataEngine


class TestWeek1(unittest.TestCase):

    def setUp(self):
        """Setup test data"""
        np.random.seed(42)

        # Fake banking data
        n_samples = 1000

        self.test_data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples),
                "balance": np.random.exponential(5000, n_samples),
                "credit_score": np.random.normal(650, 100, n_samples).clip(300, 850),
                "account_type": np.random.choice(
                    ["savings", "checking", "business"], n_samples
                ),
                "employment_status": np.random.choice(
                    ["employed", "unemployed", "self-employed", "retired"], n_samples
                ),
                "annual_income": np.random.lognormal(10, 1, n_samples),
                "total_debt": np.random.exponential(20000, n_samples),
            }
        )

        # Add some missing values
        self.test_data.loc[np.random.choice(n_samples, 50), "credit_score"] = np.nan

        # Save temporary data
        self.test_data.to_csv("test_data.csv", index=False)

    def test_deep_type_categorizer(self):
        """Test deep type classifier"""
        engine = CoreDataEngine()

        # Analyze single column
        metadata = engine.analyzer.analyze_column(self.test_data["balance"], "balance")

        self.assertIn("deep_type", metadata)
        self.assertIn("optimal_transformation", metadata)
        print(f"Column 'balance' classified as: {metadata['deep_type']}")

    def test_constraint_engine(self):
        """Test constraint engine"""
        engine = CoreDataEngine()

        # Add constraints
        engine.constraint_engine.add_constraint("min_age", "age >= 18")
        engine.constraint_engine.add_constraint("positive_balance", "balance >= 0")

        # Validate row
        test_row = {"age": 25, "balance": 5000}
        results = engine.constraint_engine.validate_row(test_row)

        self.assertTrue(results["min_age"])
        self.assertTrue(results["positive_balance"])
        print("Constraints validated successfully")

    def test_full_pipeline(self):
        """Test full pipeline"""
        engine = CoreDataEngine()

        # Run pipeline on test data
        results = engine.full_pipeline(
            raw_data_path="test_data.csv",
            constraints_file=None,  # No constraints for this test
        )

        self.assertIn("transformed_data", results)
        self.assertIn("metadata", results)

        print(f"Successfully transformed {len(results['transformed_data'])} records")
        print(f"Successfully analyzed {len(results['metadata'])} columns")

    def tearDown(self):
        """Cleanup after tests"""
        import os

        if os.path.exists("test_data.csv"):
            os.remove("test_data.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
