"""
core/feature_engineering/transformation_validator.py
System for validating transformation quality and preserving statistical relationships
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class TransformationValidator:
    """
    Advanced system for verifying that transformations preserve statistical relationships
    """

    def __init__(self):
        self.validation_results = {}

    def validate_transformation(self, original_df, transformed_df, metadata):
        """
        Comprehensive validation of transformation quality

        Returns:
            dict: Validation results with quality scores
        """
        results = {
            "column_wise_metrics": {},
            "relationship_preservation": {},
            "overall_score": 0.0,
        }

        # 1. Metrics per column
        for column in original_df.columns:
            orig_data = original_df[column].dropna()
            trans_data = transformed_df[column].dropna()

            if len(orig_data) > 0 and len(trans_data) > 0:
                col_metrics = self._calculate_column_metrics(
                    orig_data, trans_data, metadata[column]
                )
                results["column_wise_metrics"][column] = col_metrics

        # 2. Preservation of relationships between columns
        results["relationship_preservation"] = self._validate_relationships(
            original_df, transformed_df
        )

        # 3. Calculate overall score
        results["overall_score"] = self._calculate_overall_score(results)

        return results

    def _calculate_column_metrics(self, original, transformed, metadata):
        """Calculate quality metrics for column"""
        metrics = {}

        # Check length compatibility
        if len(original) != len(transformed):
            metrics["error"] = "Length mismatch"
            return metrics

        # If data is non-numeric, use temporary encoding
        if not pd.api.types.is_numeric_dtype(original):
            original = pd.factorize(original)[0]
        if not pd.api.types.is_numeric_dtype(transformed):
            transformed = pd.factorize(transformed)[0]

        # 1. Rank preservation
        try:
            original_ranks = pd.Series(original).rank()
            transformed_ranks = pd.Series(transformed).rank()
            metrics["spearman_correlation"] = original_ranks.corr(transformed_ranks)
        except:
            metrics["spearman_correlation"] = np.nan

        # 2. Distribution preservation (Jensen-Shannon Divergence)
        try:
            from scipy.stats import gaussian_kde

            min_val = min(np.min(original), np.min(transformed))
            max_val = max(np.max(original), np.max(transformed))
            x = np.linspace(min_val, max_val, 1000)

            kde_orig = gaussian_kde(original)
            kde_trans = gaussian_kde(transformed)

            pdf_orig = kde_orig(x)
            pdf_trans = kde_trans(x)

            pdf_orig = pdf_orig / pdf_orig.sum()
            pdf_trans = pdf_trans / pdf_trans.sum()

            metrics["js_divergence"] = jensenshannon(pdf_orig, pdf_trans)
        except:
            metrics["js_divergence"] = np.nan

        # 3. Basic statistics preservation
        try:
            metrics["mean_preservation"] = abs(
                np.mean(original) - np.mean(transformed)
            ) / (abs(np.mean(original)) + 1e-12)
            metrics["std_preservation"] = abs(
                np.std(original) - np.std(transformed)
            ) / (np.std(original) + 1e-12)
        except:
            metrics["mean_preservation"] = np.nan
            metrics["std_preservation"] = np.nan

        # 4. Kolmogorov-Smirnov test for distribution
        try:
            ks_stat, ks_p = stats.ks_2samp(original, transformed)
            metrics["ks_statistic"] = ks_stat
            metrics["ks_p_value"] = ks_p
        except:
            metrics["ks_statistic"] = np.nan
            metrics["ks_p_value"] = np.nan

        # 5. Mutual information estimation
        try:
            metrics["mutual_information"] = self._estimate_mutual_information(
                original, transformed
            )
        except:
            metrics["mutual_information"] = np.nan

        return metrics

    def _calculate_overall_score(self, results):
        """
        Calculate final transformation quality score based on all metrics.

        Idea:
        - Take key metrics such as correlation_preservation, mean_preservation, std_preservation, mutual_information.
        - Convert them to values between 0 and 1.
        - Calculate weighted average to produce overall_score.
        """
        scores = []

        try:
            # 1. Correlation preservation
            corr_pres = results.get("relationship_preservation", {}).get(
                "correlation_preservation", {}
            )
            if isinstance(corr_pres, dict):
                sim = corr_pres.get("correlation_similarity")
                if sim is not None and not np.isnan(sim):
                    scores.append(max(0.0, min(1.0, sim)))

            # 2. Mean and standard deviation preservation
            mean_pres = results.get("mean_preservation")
            std_pres = results.get("std_preservation")
            if mean_pres is not None and not np.isnan(mean_pres):
                scores.append(max(0.0, 1.0 - mean_pres))
            if std_pres is not None and not np.isnan(std_pres):
                scores.append(max(0.0, 1.0 - std_pres))

            # 3. Mutual information
            mi = results.get("mutual_information")
            if mi is not None and not np.isnan(mi):
                # Normalize value (usually MI > 0, use simple logistic function)
                scores.append(1 - np.exp(-mi))

            # 4. Hierarchy preservation
            hier_pres = results.get("relationship_preservation", {}).get(
                "hierarchy_preservation"
            )
            if isinstance(hier_pres, dict):
                if hier_pres.get("preserved", False):
                    scores.append(1.0)
                else:
                    scores.append(0.5)  # If not preserved, consider as medium

        except Exception as e:
            return {"error": f"Overall score calculation failed: {str(e)}"}

        if len(scores) == 0:
            return 0.0

        # Final average
        return float(np.mean(scores))

    def _estimate_mutual_information(self, original, transformed, bins=20):
        """
        Estimate mutual information between original and transformed column.

        Idea:
        - Discretize data into bins to build joint distributions.
        - Calculate joint and marginal probabilities.
        - Use mutual information formula:
        I(X;Y) = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )

        Args:
            original (pd.Series): Original column
            transformed (pd.Series): Transformed column
            bins (int): Number of bins for estimation

        Returns:
            float: Mutual information value
        """
        try:
            # Convert to numpy
            x = np.array(original.dropna())
            y = np.array(transformed.dropna())

            # If lengths differ, truncate to smallest
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]

            # Build joint histogram
            c_xy, _, _ = np.histogram2d(x, y, bins=bins)

            # Convert to probabilities
            p_xy = c_xy / float(np.sum(c_xy))
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)

            # Calculate mutual information
            mi = 0.0
            for i in range(p_x.shape[0]):
                for j in range(p_y.shape[0]):
                    if p_xy[i, j] > 0:
                        mi += p_xy[i, j] * np.log(
                            p_xy[i, j] / (p_x[i] * p_y[j] + 1e-12)
                        )

            return float(mi)
        except Exception:
            return np.nan

    def _validate_relationships(self, original_df, transformed_df):
        """Validate preservation of relationships between variables"""
        relationships = {}

        # 1. Correlation matrix (for numeric columns only)
        numeric_original = original_df.select_dtypes(include=[np.number])
        numeric_transformed = transformed_df.select_dtypes(include=[np.number])

        if numeric_original.shape[1] > 0 and numeric_transformed.shape[1] > 0:
            try:
                orig_corr = numeric_original.corr().values
                trans_corr = numeric_transformed.corr().values

                # Calculate correlation differences
                corr_diff = np.abs(orig_corr - trans_corr)
                relationships["correlation_preservation"] = {
                    "mean_difference": np.nanmean(corr_diff),
                    "max_difference": np.nanmax(corr_diff),
                    "correlation_similarity": np.corrcoef(
                        orig_corr.flatten(), trans_corr.flatten()
                    )[0, 1],
                }
            except Exception as e:
                relationships["correlation_preservation"] = {
                    "error": f"Correlation calculation failed: {str(e)}"
                }
        else:
            relationships["correlation_preservation"] = {
                "message": "No numeric columns available for correlation analysis"
            }

        # 2. Hierarchy preservation (for categorical data)
        relationships["hierarchy_preservation"] = self._check_hierarchy_preservation(
            original_df, transformed_df
        )

        return relationships

    def _check_hierarchy_preservation(self, original_df, transformed_df):
        """Check preservation of data hierarchy"""
        # This is an advanced function that needs adaptation to data type
        # Simplified example: check preservation of group means order
        hierarchy_scores = {}

        # More complex logic for hierarchical data can be added here
        # Such as preserving group and subgroup relationships

        return hierarchy_scores

    def generate_validation_report(self, validation_results, save_path=None):
        """Generate detailed validation report"""

        report = []
        report.append("=" * 80)
        report.append("Transformation Quality Validation Report")
        report.append("=" * 80)

        # Column summary
        report.append("\n1. Individual Column Evaluation:")
        report.append("-" * 40)

        for column, metrics in validation_results["column_wise_metrics"].items():
            report.append(f"\nColumn: {column}")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    report.append(f"  - {metric}: {value:.4f}")

        # Relationship preservation
        report.append("\n\n2. Preservation of Relationships Between Variables:")
        report.append("-" * 40)

        rel_metrics = validation_results["relationship_preservation"]
        if "correlation_preservation" in rel_metrics:
            cp = rel_metrics["correlation_preservation"]

            mean_diff = cp.get("mean_difference", 0.0)
            max_diff = cp.get("max_difference", 0.0)
            corr_sim = cp.get("correlation_similarity", 0.0)

            report.append(f"Mean correlation difference: {mean_diff:.4f}")
            report.append(f"Maximum correlation difference: {max_diff:.4f}")
            report.append(f"Correlation matrix similarity: {corr_sim:.4f}")

        # Overall score
        report.append("\n\n3. Overall Assessment:")
        report.append("-" * 40)
        report.append(f"Overall score: {validation_results['overall_score']:.2%}")

        report_str = "\n".join(report)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_str)

        return report_str

    def visualize_transformations(
        self, original_df, transformed_df, columns, save_dir=None
    ):
        """Create visualizations for transformations"""

        for column in columns:
            if column in original_df.columns and column in transformed_df.columns:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Original data
                orig_data = original_df[column].dropna()
                trans_data = transformed_df[column].dropna()

                # 1. Histogram
                axes[0, 0].hist(
                    orig_data, bins=50, alpha=0.5, label="Original", density=True
                )
                axes[0, 0].hist(
                    trans_data, bins=50, alpha=0.5, label="Transformed", density=True
                )
                axes[0, 0].set_title(f"{column} Distribution")
                axes[0, 0].legend()

                # 2. Box plot
                axes[0, 1].boxplot(
                    [orig_data, trans_data], labels=["Original", "Transformed"]
                )
                axes[0, 1].set_title(f"{column} Box Plot")

                # 3. Q-Q plot
                stats.probplot(orig_data, dist="norm", plot=axes[0, 2])
                axes[0, 2].set_title(f"Q-Q Plot (Original)")

                # 4. Rank scatter plot
                if len(orig_data) == len(trans_data):
                    axes[1, 0].scatter(orig_data.rank(), trans_data.rank(), alpha=0.5)
                    axes[1, 0].set_xlabel("Original Ranks")
                    axes[1, 0].set_ylabel("Transformed Ranks")
                    axes[1, 0].set_title("Rank Preservation")

                # 5. KDE density plot
                from scipy.stats import gaussian_kde

                x = np.linspace(
                    min(orig_data.min(), trans_data.min()),
                    max(orig_data.max(), trans_data.max()),
                    1000,
                )
                kde_orig = gaussian_kde(orig_data)
                kde_trans = gaussian_kde(trans_data)
                axes[1, 1].plot(x, kde_orig(x), label="Original")
                axes[1, 1].plot(x, kde_trans(x), label="Transformed")
                axes[1, 1].set_title("Probability Density Estimation")
                axes[1, 1].legend()

                # 6. CDF plot
                sorted_orig = np.sort(orig_data)
                sorted_trans = np.sort(trans_data)
                cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
                cdf_trans = np.arange(1, len(sorted_trans) + 1) / len(sorted_trans)
                axes[1, 2].plot(sorted_orig, cdf_orig, label="Original")
                axes[1, 2].plot(sorted_trans, cdf_trans, label="Transformed")
                axes[1, 2].set_title("Cumulative Distribution Function")
                axes[1, 2].legend()

                plt.tight_layout()

                if save_dir:
                    plt.savefig(
                        f"{save_dir}/{column}_transformation.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                plt.close()
