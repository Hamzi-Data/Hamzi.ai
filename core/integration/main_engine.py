# core/integration/main_engine.py
"""
Main engine that integrates all components
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import json
from typing import Dict
from core.feature_engineering.deep_type_categorizer import (
    DeepFeatureAnalyzer,
    AdvancedDataTransformer,
)
from core.feature_engineering.transformation_validator import TransformationValidator
from core.constraints.constraint_engine import ConstraintEngine
from core.constraints.temporal_constraints import TemporalConstraintEngine


def json_safe(obj):
    import numpy as np
    import pandas as pd
    from datetime import datetime, date

    # Do not use np.bool8 as it does not exist in modern versions
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(pd.Timestamp(obj))
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


class CoreDataEngine:
    """
    Core engine for data preparation and constraint application - Week 1
    """

    def __init__(self):
        self.analyzer = DeepFeatureAnalyzer()
        self.transformer = AdvancedDataTransformer(self.analyzer)
        self.validator = TransformationValidator()
        self.constraint_engine = ConstraintEngine()
        self.temporal_engine = TemporalConstraintEngine()

        self.feature_metadata = {}
        self.constraint_masks = {}
        self.validation_results = {}

        print("Core data preparation engine initialized")

    def load_and_prepare_data(self, data_path: str, sample_size: int = None):
        """Load raw data"""

        print(f"Loading data from {data_path}")

        if data_path.endswith(".csv"):
            df = pd.read_csv(data_path, sep=";")
        elif data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported file type")

        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)

        print(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def full_pipeline(self, raw_data_path: str, constraints_file: str = None):
        results = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_steps": {},
            "metrics": {},
        }

        # Step 1: Load raw data
        print("\n" + "=" * 60)
        print("Step 1: Loading raw data")
        print("=" * 60)

        df_raw = self.load_and_prepare_data(raw_data_path)
        results["pipeline_steps"]["data_loading"] = {
            "rows": len(df_raw),
            "columns": list(df_raw.columns),
            "memory_usage_mb": df_raw.memory_usage(deep=True).sum() / 1024**2,
        }

        # Step 2: Deep type analysis
        print("\n" + "=" * 60)
        print("Step 2: Deep type analysis")
        print("=" * 60)

        for column in df_raw.columns:
            metadata = self.analyzer.analyze_column(df_raw[column], column)
            self.feature_metadata[column] = metadata
            print(f"{column}: {metadata['deep_type']}")

        results["pipeline_steps"]["deep_typing"] = {
            "feature_types": {
                col: meta["deep_type"] for col, meta in self.feature_metadata.items()
            }
        }

        # Step 3: Data transformation
        print("\n" + "=" * 60)
        print("Step 3: Transforming data to numerical representation")
        print("=" * 60)

        df_transformed, original_dtypes = self.transformer.fit_transform(df_raw)
        results["pipeline_steps"]["transformation"] = {
            "transformed_columns": list(df_transformed.columns),
            "transformers_applied": len(self.transformer.fitted_transformers),
        }

        # Step 4: Transformation quality validation
        print("\n" + "=" * 60)
        print("Step 4: Validating transformation quality")
        print("=" * 60)

        self.validation_results = self.validator.validate_transformation(
            df_raw, df_transformed, self.feature_metadata
        )

        validation_report = self.validator.generate_validation_report(
            self.validation_results
        )
        print(validation_report)

        results["pipeline_steps"]["validation"] = {
            "overall_score": self.validation_results["overall_score"],
            "column_metrics": self.validation_results["column_wise_metrics"],
        }

        # Step 5: Load and apply constraints
        if constraints_file:
            print("\n" + "=" * 60)
            print("Step 5: Applying strict constraints")
            print("=" * 60)

            self._load_constraints(constraints_file)

            constraint_results = self.constraint_engine.validate_dataframe(df_raw)

            violations_report = self.constraint_engine.generate_violation_report()
            print(violations_report)

            results["pipeline_steps"]["constraints"] = {
                "total_constraints": len(self.constraint_engine.constraints),
                "total_violations": sum(
                    c["violations"] for c in self.constraint_engine.constraints.values()
                ),
            }

            self.constraint_masks = self.constraint_engine.generate_constraint_mask(
                df_raw
            )

            results["pipeline_steps"]["constraint_masks"] = {
                "mask_shape": self.constraint_masks.shape,
                "valid_rows": int(self.constraint_masks.sum()),
                "invalid_rows": int((self.constraint_masks).sum()),
            }

        # Step 6: Generate final report
        print("\n" + "=" * 60)
        print("Step 6: Generating final report")
        print("=" * 60)

        final_report = self._generate_final_report(results)
        print(final_report)

        # Save results
        self._save_results(results, df_transformed)

        return {
            "raw_data": df_raw,
            "transformed_data": df_transformed,
            "metadata": self.feature_metadata,
            "constraint_masks": self.constraint_masks,
            "validation_results": self.validation_results,
            "pipeline_results": results,
        }

    def _load_constraints(self, constraints_file: str):
        if constraints_file.endswith(".json"):
            with open(constraints_file, "r", encoding="utf-8") as f:
                constraints_data = json.load(f)
            for name, constraint_str in constraints_data.items():
                self.constraint_engine.add_constraint(name, constraint_str)
        elif constraints_file.endswith(".txt"):
            with open(constraints_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if ":" in line:
                            name, constraint = line.split(":", 1)
                            self.constraint_engine.add_constraint(
                                name.strip(), constraint.strip()
                            )

    def _generate_final_report(self, results: Dict) -> str:
        report = []
        report.append("=" * 80)
        report.append("Week 1 Report: Data Engineering and Logical Foundation")
        report.append("=" * 80)

        report.append(f"\nTimestamp: {results['timestamp']}")

        data_info = results["pipeline_steps"]["data_loading"]
        report.append(f"\nData Summary:")
        report.append(f"  - Number of records: {data_info['rows']:,}")
        report.append(f"  - Number of columns: {len(data_info['columns'])}")
        report.append(f"  - Memory usage: {data_info['memory_usage_mb']:.2f} MB")

        transform_info = results["pipeline_steps"]["transformation"]
        report.append(f"\nTransformation Summary:")
        report.append(
            f"  - Number of applied transformers: {transform_info['transformers_applied']}"
        )

        type_info = results["pipeline_steps"]["deep_typing"]
        type_counts = {}
        for ftype in type_info["feature_types"].values():
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

        report.append(f"\nDeep Type Distribution:")
        for ftype, count in type_counts.items():
            report.append(f"  - {ftype}: {count}")

        valid_info = results["pipeline_steps"]["validation"]
        report.append(
            f"\nOverall transformation quality: {valid_info['overall_score']:.2%}"
        )

        if "constraints" in results["pipeline_steps"]:
            const_info = results["pipeline_steps"]["constraints"]
            mask_info = results["pipeline_steps"]["constraint_masks"]

            report.append(f"\nConstraints Summary:")
            report.append(
                f"  - Number of constraints: {const_info['total_constraints']}"
            )
            report.append(f"  - Total violations: {const_info['total_violations']}")
            report.append(
                f"  - Valid records: {mask_info['valid_rows']:,} "
                f"({mask_info['valid_rows']/data_info['rows']:.2%})"
            )
            report.append(f"  - Invalid records: {mask_info['invalid_rows']:,}")

        report.append("\nRecommendations:")

        if valid_info["overall_score"] < 0.95:
            report.append("  - Improve transformations for low-quality types")

        if "constraints" in results["pipeline_steps"]:
            if const_info["total_violations"] > 0:
                report.append(
                    "  - Review violated constraints and adjust data or rules"
                )

        report.append("\nWeek 1 completed successfully!")
        report.append("Engine is ready for Week 2: Hybrid Generative Engine")

        return "\n".join(report)

    def _save_results(self, results: Dict, transformed_data: pd.DataFrame):
        """Save all results with safe type conversion for JSON"""

        import os
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/week1_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

        # 1) Save transformed data
        transformed_data.to_parquet(f"{results_dir}/transformed_data.parquet")

        # 2) Save metadata (JSON-safe)
        safe_metadata = json_safe(self.feature_metadata)
        with open(f"{results_dir}/feature_metadata.json", "w", encoding="utf-8") as f:
            json.dump(safe_metadata, f, ensure_ascii=False, indent=2)

        # 3) Save validation results (JSON-safe)
        safe_validation = json_safe(self.validation_results)
        with open(f"{results_dir}/validation_results.json", "w", encoding="utf-8") as f:
            json.dump(safe_validation, f, ensure_ascii=False, indent=2)

        # 4) Save final report as text
        final_report_text = self._generate_final_report(results)
        with open(f"{results_dir}/final_report.txt", "w", encoding="utf-8") as f:
            f.write(final_report_text)

        # 5) Save constraints and masks
        if self.constraint_engine.constraints:
            self.constraint_engine.save_constraints(f"{results_dir}/constraints.json")
            if hasattr(self, "constraint_masks") and isinstance(
                self.constraint_masks, np.ndarray
            ):
                np.save(f"{results_dir}/constraint_masks.npy", self.constraint_masks)

        print(f"\nAll results saved to: {results_dir}")
        return results_dir


# Usage example
if __name__ == "__main__":
    engine = CoreDataEngine()
    results = engine.full_pipeline(
        raw_data_path="data/raw_bank_data.csv",
        constraints_file="constraints/bank_rules.txt",
    )
    print("\nWeek 1 task completed successfully!")
    print("Deep-Type Categorizer: working efficiently")
    print("Hard-Constraints DSL: working efficiently")
    print("Data is ready for Week 2 (Hybrid Generative Engine)")
