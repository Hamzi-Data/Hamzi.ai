# scripts/evaluate_generation.py
"""
Comprehensive evaluation script for generated data
"""

import sys
import os
import argparse
import pandas as pd
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.integration.hybrid_engine import HybridGenerativeEngine
from core.generative.evaluation.metrics_calculator import MetricsCalculator
from core.generative.evaluation.privacy_tester import PrivacyTester


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of generated data"
    )
    parser.add_argument(
        "--real_data", type=str, required=True, help="Path to real data for comparison"
    )
    parser.add_argument(
        "--synthetic_data",
        type=str,
        required=True,
        help="Path to generated data for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--run_privacy_tests",
        action="store_true",
        help="Run privacy tests (Membership Inference)",
    )

    args = parser.parse_args()

    print("Starting comprehensive evaluation of generated data")
    print("=" * 60)

    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"\nLoading real data: {args.real_data}")
    real_data = pd.read_csv(args.real_data)

    print(f"Loading synthetic data: {args.synthetic_data}")
    synthetic_data = pd.read_csv(args.synthetic_data)

    print(
        f"\nReal data size: {len(real_data)} records, {len(real_data.columns)} columns"
    )
    print(
        f"Synthetic data size: {len(synthetic_data)} records, {len(synthetic_data.columns)} columns"
    )

    # 1. Statistical quality evaluation
    print("\nStarting statistical evaluation...")
    metrics_calculator = MetricsCalculator(real_data)
    quality_metrics = metrics_calculator.compute_all_metrics(synthetic_data)

    # Save quality metrics
    quality_path = os.path.join(args.output_dir, "quality_metrics.json")
    with open(quality_path, "w") as f:
        json.dump(quality_metrics, f, indent=2, default=str)

    # Quality report
    quality_report = metrics_calculator.generate_detailed_report(synthetic_data)
    quality_report_path = os.path.join(args.output_dir, "quality_report.txt")
    with open(quality_report_path, "w", encoding="utf-8") as f:
        f.write(quality_report)

    print(f"Quality report saved to: {quality_report_path}")

    # 2. Privacy tests (optional)
    if args.run_privacy_tests:
        print("\nStarting privacy tests...")

        privacy_tester = PrivacyTester()

        # Split real data
        from sklearn.model_selection import train_test_split

        real_train, real_test = train_test_split(
            real_data, test_size=0.3, random_state=42
        )

        # Run Membership Inference attacks
        privacy_results, privacy_report = privacy_tester.evaluate_privacy(
            target_model=None,
            real_train=real_train,
            real_test=real_test,
            synthetic_data=synthetic_data,
        )

        # Save privacy results
        privacy_path = os.path.join(args.output_dir, "privacy_results.json")
        with open(privacy_path, "w") as f:
            json.dump(privacy_results, f, indent=2, default=str)

        privacy_report_path = os.path.join(args.output_dir, "privacy_report.txt")
        with open(privacy_report_path, "w", encoding="utf-8") as f:
            f.write(privacy_report)

        print(f"Privacy report saved to: {privacy_report_path}")

    # 3. Summary report
    print("\nGenerating summary report...")

    summary_report = generate_summary_report(
        quality_metrics, privacy_results if args.run_privacy_tests else None
    )

    summary_path = os.path.join(args.output_dir, "summary_report.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_report)

    print(f"Summary report saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Comprehensive evaluation completed!")
    print(f"All results saved in: {args.output_dir}")


def generate_summary_report(quality_metrics, privacy_results=None):
    """Generate a concise evaluation summary report."""
    report = []
    report.append("=" * 80)
    report.append("Concise Evaluation Report - Generated Data")
    report.append("=" * 80)

    # Data quality
    report.append("\nData Quality:")
    report.append("-" * 40)

    if "quality_score" in quality_metrics:
        score = quality_metrics["quality_score"]
        report.append(f"Overall score: {score:.2%}")

        if score > 0.9:
            report.append("Excellent quality")
        elif score > 0.7:
            report.append("Good quality")
        elif score > 0.5:
            report.append("Acceptable quality")
        else:
            report.append("Low quality")

    # Privacy
    if privacy_results:
        report.append("\nData Privacy:")
        report.append("-" * 40)

        if "overall_privacy_score" in privacy_results:
            privacy_score = privacy_results["overall_privacy_score"]
            report.append(f"Privacy score: {privacy_score:.2%}")

            if privacy_score > 0.8:
                report.append("Excellent privacy")
            elif privacy_score > 0.6:
                report.append("Good privacy")
            else:
                report.append("Low privacy")

        if "membership_inference" in privacy_results:
            mi_acc = privacy_results["membership_inference"].get("accuracy", 0)
            report.append(f"Membership Inference attack accuracy: {mi_acc:.4f}")

    # Recommendations
    report.append("\nRecommendations:")
    report.append("-" * 40)

    if "quality_score" in quality_metrics:
        score = quality_metrics["quality_score"]
        if score < 0.7:
            report.append("1. Improve model training")
            report.append("2. Increase generator complexity")
            report.append("3. Tune training hyperparameters")

    if privacy_results and "overall_privacy_score" in privacy_results:
        if privacy_results["overall_privacy_score"] < 0.6:
            report.append("4. Add stronger privacy layers")
            report.append("5. Increase noise strength")

    report.append("\nThe data is ready for use in: analysis, model training, testing")

    return "\n".join(report)


if __name__ == "__main__":
    main()
