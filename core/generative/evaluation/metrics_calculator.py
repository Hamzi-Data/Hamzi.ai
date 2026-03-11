"""
core/generative/evaluation/metrics_calculator.py
Production-grade calculator for statistical metrics of generated data quality.

- Unified interface: Accepts real and generated data as Pandas DataFrame, with optional specification of categorical columns.
- Distribution metrics: JS/TVD for categorical columns, KS/Wasserstein/moment differences for numerical columns.
- Correlation metrics: Correlation matrix similarity (Pearson/Spearman), mean absolute difference, rank preservation.
- Privacy metrics: Nearest neighbor distance and average group overlap as preliminary indicators.
- Configurable weighted overall quality score.
- Clear error handling, protection against null values and mismatches.

Copyright: Microsoft AI (Production-grade re-engineering)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.spatial.distance import jensenshannon


# ========= Errors and Utilities =========


class MetricsError(ValueError):
    """Custom exception for metric calculator errors with precise diagnostic messages."""

    pass


def _ensure_dataframe(df: Any, name: str) -> pd.DataFrame:
    """Ensures input is a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise MetricsError(f"{name} must be a pandas.DataFrame.")
    if df.empty:
        raise MetricsError(
            f"{name} is empty; cannot compute metrics on an empty DataFrame."
        )
    return df


def _safe_mean(values: np.ndarray) -> float:
    """Safe mean ignoring NaN and returns 0.0 if no valid values exist."""
    values = np.asarray(values, dtype=np.float64)
    mask = ~np.isnan(values)
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


def _align_categories(
    real_counts: pd.Series, synth_counts: pd.Series
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Aligns categorical distribution between real and synthetic on the union of categories."""
    cats = list(set(real_counts.index) | set(synth_counts.index))
    real_probs = np.array([real_counts.get(c, 0.0) for c in cats], dtype=np.float64)
    synth_probs = np.array([synth_counts.get(c, 0.0) for c in cats], dtype=np.float64)
    # Protection against zero arrays
    if real_probs.sum() <= 0:
        real_probs = np.ones_like(real_probs, dtype=np.float64) / len(real_probs)
    if synth_probs.sum() <= 0:
        synth_probs = np.ones_like(synth_probs, dtype=np.float64) / len(synth_probs)
    return real_probs, synth_probs, cats


# ========= Metrics Calculator =========


class MetricsCalculator:
    """
    Production-grade calculator for statistical metrics between real and generated data.

    Input/Output interface:
    - compute_all_metrics(synthetic_data) -> Dict[str, Any]
      Contains keys: distribution_metrics, correlation_metrics, privacy_metrics, quality_score.

    Args:
        real_data: Real data DataFrame.
        categorical_columns: List of categorical columns (optional).
        weights: Overall quality score weights (distribution/correlation/privacy).
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.real_data: pd.DataFrame = _ensure_dataframe(real_data, "real_data")
        self.categorical_columns: List[str] = list(categorical_columns or [])
        self.weights: Dict[str, float] = dict(
            weights or {"distribution": 0.4, "correlation": 0.3, "privacy": 0.3}
        )

        # Validate categorical columns
        unknown_cats = [
            c for c in self.categorical_columns if c not in self.real_data.columns
        ]
        if unknown_cats:
            raise MetricsError(
                f"Categorical columns not found in real_data: {unknown_cats}"
            )

        # Real data statistics for future use
        self.real_stats: Dict[str, Dict[str, Any]] = self._compute_real_statistics()

    def _compute_real_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Computes basic statistics for each column in the real data.

        Returns:
            Dict[str, Dict[str, Any]]: Column type and its basic statistics.
        """
        stats_dict: Dict[str, Dict[str, Any]] = {}

        for col in self.real_data.columns:
            col_data = self.real_data[col].dropna()

            if col in self.categorical_columns:
                value_counts = col_data.value_counts(normalize=True)
                stats_dict[col] = {
                    "type": "categorical",
                    "distribution": value_counts.to_dict(),
                    "entropy": (
                        float(stats.entropy(value_counts.values))
                        if len(value_counts) > 0
                        else 0.0
                    ),
                    "unique_count": int(len(value_counts)),
                }
            else:
                # Numerical
                numeric = pd.to_numeric(col_data, errors="coerce").dropna()
                if numeric.empty:
                    stats_dict[col] = {
                        "type": "numerical",
                        "mean": 0.0,
                        "std": 0.0,
                        "skewness": 0.0,
                        "kurtosis": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }
                else:
                    stats_dict[col] = {
                        "type": "numerical",
                        "mean": float(numeric.mean()),
                        "std": float(numeric.std(ddof=1)),
                        "skewness": float(stats.skew(numeric)),
                        "kurtosis": float(stats.kurtosis(numeric)),
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                    }

        return stats_dict

    def compute_all_metrics(self, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes all statistical metrics and returns a comprehensive dictionary.

        Args:
            synthetic_data: Generated data DataFrame with same column schema as much as possible.

        Returns:
            Dict[str, Any]: All metrics and overall score.
        """
        synth_df = _ensure_dataframe(synthetic_data, "synthetic_data")

        # Basic column compatibility
        common_cols = [c for c in self.real_data.columns if c in synth_df.columns]
        if not common_cols:
            raise MetricsError("No common columns between real and generated data.")

        # 1) Distribution
        distribution_metrics = self._compute_distribution_metrics(synth_df, common_cols)

        # 2) Correlation
        correlation_metrics = self._compute_correlation_metrics(synth_df)

        # 3) Privacy
        privacy_metrics = self._compute_privacy_metrics(synth_df)

        # 4) Overall quality score
        all_metrics = {
            "distribution_metrics": distribution_metrics,
            "correlation_metrics": correlation_metrics,
            "privacy_metrics": privacy_metrics,
        }
        quality_score = self._compute_overall_quality_score(all_metrics)

        return {**all_metrics, "quality_score": quality_score}

    def _compute_distribution_metrics(
        self, synthetic_data: pd.DataFrame, common_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Computes distribution metrics for each common column between real and generated data.

        Returns:
            Dict[str, Any]: Per column, then average JS/Wasserstein.
        """
        metrics: Dict[str, Any] = {}

        for col in common_cols:
            real_series = self.real_data[col].dropna()
            synth_series = synthetic_data[col].dropna()

            if real_series.empty or synth_series.empty:
                # Skip columns empty in either side
                continue

            if col in self.categorical_columns:
                real_counts = real_series.value_counts(normalize=True)
                synth_counts = synth_series.value_counts(normalize=True)
                real_probs, synth_probs, _ = _align_categories(
                    real_counts, synth_counts
                )

                js_div = float(jensenshannon(real_probs, synth_probs))  # [0, 1]
                tv_dist = float(0.5 * np.sum(np.abs(real_probs - synth_probs)))
                coverage = float(
                    len(set(synth_series.unique()) & set(real_series.unique()))
                    / max(1, len(set(real_series.unique())))
                )

                metrics[col] = {
                    "js_divergence": js_div,
                    "tv_distance": tv_dist,
                    "category_coverage": coverage,
                }
            else:
                # Safe numerical conversion
                real_num = pd.to_numeric(real_series, errors="coerce").dropna()
                synth_num = pd.to_numeric(synth_series, errors="coerce").dropna()
                if real_num.empty or synth_num.empty:
                    continue

                ks_stat, ks_p = stats.ks_2samp(real_num.values, synth_num.values)

                wasserstein_dist = float(
                    stats.wasserstein_distance(real_num.values, synth_num.values)
                )

                real_mean, real_std = float(real_num.mean()), float(
                    real_num.std(ddof=1)
                )
                synth_mean, synth_std = float(synth_num.mean()), float(
                    synth_num.std(ddof=1)
                )

                mean_diff = float(abs(real_mean - synth_mean) / (abs(real_mean) + 1e-8))
                std_diff = float(abs(real_std - synth_std) / (real_std + 1e-8))

                metrics[col] = {
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_p),
                    "wasserstein_distance": wasserstein_dist,
                    "mean_difference": mean_diff,
                    "std_difference": std_diff,
                    "real_mean": real_mean,
                    "synth_mean": synth_mean,
                }

        metrics["average_js_divergence"] = _safe_mean(
            np.array(
                [
                    m["js_divergence"]
                    for m in metrics.values()
                    if isinstance(m, dict) and "js_divergence" in m
                ]
            )
        )
        metrics["average_wasserstein"] = _safe_mean(
            np.array(
                [
                    m["wasserstein_distance"]
                    for m in metrics.values()
                    if isinstance(m, dict) and "wasserstein_distance" in m
                ]
            )
        )

        return metrics

    def _compute_correlation_metrics(
        self, synthetic_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Computes correlation preservation metrics between matrices.

        Returns:
            Dict[str, float]: Pearson/Spearman/MeanAbsDiff/RankPreservation/MatrixSimilarity.
        """
        metrics: Dict[str, float] = {}

        real_corr = self.real_data.corr(numeric_only=True)
        synth_corr = synthetic_data.corr(numeric_only=True)

        # Common numerical columns
        common_cols = list(set(real_corr.columns) & set(synth_corr.columns))
        if not common_cols:
            return metrics

        real_corr_common = real_corr.loc[common_cols, common_cols]
        synth_corr_common = synth_corr.loc[common_cols, common_cols]

        real_vec = real_corr_common.values.flatten()
        synth_vec = synth_corr_common.values.flatten()

        mask = ~np.isnan(real_vec) & ~np.isnan(synth_vec)
        real_vec = real_vec[mask]
        synth_vec = synth_vec[mask]

        if real_vec.size == 0:
            return metrics

        # Pearson/Spearman between correlation vectors
        pearson_corr = float(np.corrcoef(real_vec, synth_vec)[0, 1])
        spearman_corr = float(stats.spearmanr(real_vec, synth_vec)[0])

        mean_abs_diff = float(np.mean(np.abs(real_vec - synth_vec)))

        # Rank preservation
        real_rank = np.argsort(np.argsort(real_vec))
        synth_rank = np.argsort(np.argsort(synth_vec))
        rank_correlation = float(stats.spearmanr(real_rank, synth_rank)[0])

        metrics = {
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "mean_absolute_difference": mean_abs_diff,
            "rank_preservation": rank_correlation,
            "correlation_matrix_similarity": pearson_corr,
        }

        return metrics

    def _compute_privacy_metrics(
        self, synthetic_data: pd.DataFrame, n_neighbors: int = 5
    ) -> Dict[str, float]:
        """
        Computes basic privacy metrics using nearest neighbors.

        Note: These are preliminary metrics; differential privacy can be integrated in other layers later.

        Args:
            synthetic_data: Generated data DataFrame.
            n_neighbors: Number of neighbors for nearest distance.

        Returns:
            Dict[str, float]: Average distance, overlap ratio, and privacy score.
        """
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as e:
            # If sklearn is not available, return empty metrics instead of failing
            return {
                "average_nearest_neighbor_distance": 0.0,
                "data_overlap_ratio": 0.0,
                "privacy_score": 0.0,
            }

        real_numeric = self.real_data.select_dtypes(include=[np.number])
        synth_numeric = synthetic_data.select_dtypes(include=[np.number])

        if real_numeric.empty or synth_numeric.empty:
            return {
                "average_nearest_neighbor_distance": 0.0,
                "data_overlap_ratio": 0.0,
                "privacy_score": 0.0,
            }

        nn = NearestNeighbors(n_neighbors=max(1, n_neighbors), metric="euclidean")
        nn.fit(real_numeric.values)

        distances, _ = nn.kneighbors(synth_numeric.values)
        avg_distance = float(distances.mean())

        # Overlap ratio via two nearest neighbors between groups
        combined = np.vstack([real_numeric.values, synth_numeric.values])
        nn_combined = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn_combined.fit(combined)

        _, indices_combined = nn_combined.kneighbors(combined)

        n_real = len(real_numeric)
        cross_group_neighbors = 0
        for i in range(len(combined)):
            # Check if at least one nearest neighbor is from the other group
            for j in indices_combined[i]:
                if (i < n_real and j >= n_real) or (i >= n_real and j < n_real):
                    cross_group_neighbors += 1
                    break

        overlap_ratio = float(cross_group_neighbors / len(combined))
        privacy_score = float(max(0.0, 1.0 - overlap_ratio))

        return {
            "average_nearest_neighbor_distance": avg_distance,
            "data_overlap_ratio": overlap_ratio,
            "privacy_score": privacy_score,
        }

    def _compute_overall_quality_score(self, all_metrics: Dict[str, Any]) -> float:
        """
        Computes overall quality score based on configurable weights.

        Returns:
            float: Quality score within [0, 1].
        """
        w_dist = float(self.weights.get("distribution", 0.0))
        w_corr = float(self.weights.get("correlation", 0.0))
        w_priv = float(self.weights.get("privacy", 0.0))

        scores: List[float] = []

        # Distribution
        dist_metrics = all_metrics.get("distribution_metrics", {})
        avg_js = float(dist_metrics.get("average_js_divergence", 1.0))
        # Convert JS to score (lower is better): with simple calibration
        js_score = max(0.0, 1.0 - avg_js * 10.0)
        scores.append(js_score * w_dist)

        # Correlation
        corr_metrics = all_metrics.get("correlation_metrics", {})
        corr_sim = float(corr_metrics.get("correlation_matrix_similarity", 0.0))
        corr_score = (corr_sim + 1.0) / 2.0  # Convert [-1,1] -> [0,1]
        scores.append(corr_score * w_corr)

        # Privacy
        priv_metrics = all_metrics.get("privacy_metrics", {})
        priv_score = float(priv_metrics.get("privacy_score", 0.0))
        scores.append(priv_score * w_priv)

        total_weight = w_dist + w_corr + w_priv
        if total_weight <= 0:
            return 0.0

        return float(sum(scores) / total_weight)

    def generate_detailed_report(self, synthetic_data: pd.DataFrame) -> str:
        """
        Generates a detailed textual report on generated data quality, with brief recommendations.

        Returns:
            str: Full text report.
        """
        metrics = self.compute_all_metrics(synthetic_data)

        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("Generated Data Quality Evaluation Report")
        lines.append("=" * 80)

        # Overall score
        lines.append(f"\nOverall Quality Score: {metrics['quality_score']:.2%}")

        # Distribution
        lines.append("\nDistribution Metrics:")
        lines.append("-" * 40)
        dist_metrics = metrics.get("distribution_metrics", {})
        if "average_js_divergence" in dist_metrics:
            lines.append(
                f"Average JS Divergence: {dist_metrics['average_js_divergence']:.4f}"
            )
        if "average_wasserstein" in dist_metrics:
            lines.append(
                f"Average Wasserstein Distance: {dist_metrics['average_wasserstein']:.4f}"
            )

        # Correlation
        lines.append("\nCorrelation Metrics:")
        lines.append("-" * 40)
        corr_metrics = metrics.get("correlation_metrics", {})
        for key in (
            "pearson_correlation",
            "spearman_correlation",
            "mean_absolute_difference",
            "rank_preservation",
        ):
            if key in corr_metrics and isinstance(corr_metrics[key], float):
                lines.append(f"{key}: {corr_metrics[key]:.4f}")

        # Privacy
        lines.append("\nPrivacy Metrics:")
        lines.append("-" * 40)
        priv_metrics = metrics.get("privacy_metrics", {})
        for key in (
            "average_nearest_neighbor_distance",
            "data_overlap_ratio",
            "privacy_score",
        ):
            if key in priv_metrics and isinstance(priv_metrics[key], float):
                lines.append(f"{key}: {priv_metrics[key]:.4f}")

        # Recommendations
        lines.append("\nRecommendations:")
        lines.append("-" * 40)
        quality_score = metrics["quality_score"]
        if quality_score > 0.9:
            lines.append(
                "Excellent quality! Generated data is very close to real data."
            )
        elif quality_score > 0.7:
            lines.append(
                "Good quality, but there is room for improvement in distribution preservation."
            )
        elif quality_score > 0.5:
            lines.append("Average quality, needs improvement in training.")
        else:
            lines.append(
                "Low quality, review the generation model, training settings, and regularization."
            )

        return "\n".join(lines)

    def summarize_contract(self) -> str:
        """
        Returns a summary of operational contract (categorical columns, weights, and real_stats status).

        Returns:
            str: Brief text summary.
        """
        lines = [
            f"categorical_columns: {self.categorical_columns}",
            f"weights: {self.weights}",
            f"real_columns: {list(self.real_data.columns)}",
            f"stats_computed_for: {list(self.real_stats.keys())}",
        ]
        return "\n".join(lines)
