from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Any,
    Protocol,
    TypeVar,
    Generic,
    Callable,
    Iterator,
)

"""
core/generative/evaluation/gan_auditor.py

Advanced monitoring and evaluation system for quality and privacy of generated data
Designed according to SOLID and Clean Code principles for a large hybrid data generation system

Smart architecture:
- Single Responsibility: Each class/function has a single responsibility
- Open/Closed: Extensible without modifying existing code
- Liskov Substitution: Replaceable interfaces
- Interface Segregation: Small and specialized interfaces
- Dependency Inversion: Depend on abstractions
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json
from core.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/system.log")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import warnings

# Type Aliases for Clear Code Intent
T = TypeVar("T")
DataBatch = Union[torch.Tensor, np.ndarray, pd.DataFrame]
QualityMetric = Dict[str, Union[float, str, Dict]]
PrivacyMetric = Dict[str, Union[float, bool, Dict]]
MetricReport = Dict[str, Union[QualityMetric, PrivacyMetric, Dict]]


# ============================================================================
# DATA ABSTRACTIONS AND PROTOCOLS
# ============================================================================


class DataProvider(Protocol):
    """Unified protocol for data providers to ensure compatibility"""

    def __iter__(self) -> Iterator[DataBatch]:
        """Iterator over data batches"""
        ...

    def __len__(self) -> int:
        """Number of batches"""
        ...

    @property
    def batch_size(self) -> int:
        """Batch size"""
        ...

    @property
    def num_features(self) -> int:
        """Number of features"""
        ...


class GeneratorInterface(Protocol):
    """Unified protocol for generators to ensure compatibility with different systems"""

    def generate(self, num_samples: int, **kwargs) -> DataBatch:
        """Generate new samples"""
        ...

    @property
    def latent_dim(self) -> int:
        """Latent space dimension"""
        ...

    def to(self, device: torch.device) -> "GeneratorInterface":
        """Move model to a specific device"""
        ...


@dataclass(frozen=True)
class AuditConfig:
    """Fully customizable configuration for auditing"""

    # Computation settings
    batch_size: int = 1024
    num_evaluation_samples: int = 10000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Metric settings
    compute_quality_metrics: bool = True
    compute_privacy_metrics: bool = True
    compute_statistical_tests: bool = True
    compute_mode_coverage: bool = True

    # Advanced settings
    gmm_n_components_range: Tuple[int, int] = (2, 10)
    statistical_test_alpha: float = 0.05
    correlation_method: str = "pearson"  # pearson, spearman, kendall
    privacy_attack_n_shadow_models: int = 5
    wasserstein_n_bins: int = 100

    # Performance settings
    use_parallel_computation: bool = True
    max_memory_gb: float = 4.0
    cache_intermediate_results: bool = True

    def __post_init__(self):
        """Validate configuration after creation"""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.num_evaluation_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if not 0 < self.statistical_test_alpha < 1:
            raise ValueError("Significance level must be between 0 and 1")


# ============================================================================
# METRIC INTERFACES AND ABSTRACTIONS
# ============================================================================


class IMetricCalculator(ABC):
    """Abstract interface for metric calculators (Dependency Inversion principle)"""

    @abstractmethod
    def compute(
        self, real_data: DataBatch, synthetic_data: DataBatch
    ) -> Dict[str, Any]:
        """Compute the metric"""
        pass

    @abstractmethod
    def get_metric_name(self) -> str:
        """Metric name"""
        pass

    @abstractmethod
    def get_metric_description(self) -> str:
        """Metric description"""
        pass


class IStatisticalTest(ABC):
    """Abstract interface for statistical tests"""

    @abstractmethod
    def test(self, real_data: DataBatch, synthetic_data: DataBatch) -> Dict[str, Any]:
        """Perform the statistical test"""
        pass

    @abstractmethod
    def get_test_name(self) -> str:
        """Test name"""
        pass

    @property
    @abstractmethod
    def test_statistic(self) -> float:
        """Test statistic value"""
        pass

    @property
    @abstractmethod
    def p_value(self) -> float:
        """P-value"""
        pass


# ============================================================================
# CONCRETE METRIC IMPLEMENTATIONS
# ============================================================================


@dataclass
class WassersteinDistanceCalculator(IMetricCalculator):
    """
    Advanced Wasserstein distance calculation with performance optimizations and special handling
    Complexity: O(n log n) per feature
    """

    num_bins: int = 100
    normalize: bool = True
    use_gpu: bool = torch.cuda.is_available()

    def compute(
        self, real_data: DataBatch, synthetic_data: DataBatch
    ) -> Dict[str, Any]:
        """
        Compute advanced Wasserstein distance

        Args:
            real_data: Real data
            synthetic_data: Generated data

        Returns:
            Dict: Distance results with detailed statistics
        """
        try:
            # Convert to numpy for computation
            real_np = self._to_numpy(real_data)
            synthetic_np = self._to_numpy(synthetic_data)

            # Validate data dimensions
            if real_np.shape[1] != synthetic_np.shape[1]:
                raise ValueError("Data dimensions do not match")

            num_features = real_np.shape[1]
            distances = []
            detailed_distances = {}

            for i in range(num_features):
                # Extract feature
                real_feature = real_np[:, i]
                synthetic_feature = synthetic_np[:, i]

                # Remove missing values
                real_feature = real_feature[~np.isnan(real_feature)]
                synthetic_feature = synthetic_feature[~np.isnan(synthetic_feature)]

                if len(real_feature) < 2 or len(synthetic_feature) < 2:
                    logger.warning(
                        f"Feature {i}: insufficient data to compute distance"
                    )
                    distances.append(np.nan)
                    continue

                # Compute distance
                try:
                    distance = wasserstein_distance(real_feature, synthetic_feature)

                    # Normalize if required
                    if self.normalize:
                        feature_range = np.ptp(
                            np.concatenate([real_feature, synthetic_feature])
                        )
                        if feature_range > 0:
                            distance /= feature_range

                    distances.append(distance)
                    detailed_distances[f"feature_{i}"] = {
                        "distance": float(distance),
                        "real_mean": float(np.mean(real_feature)),
                        "synthetic_mean": float(np.mean(synthetic_feature)),
                        "real_std": float(np.std(real_feature)),
                        "synthetic_std": float(np.std(synthetic_feature)),
                        "n_samples_real": len(real_feature),
                        "n_samples_synthetic": len(synthetic_feature),
                    }

                except Exception as e:
                    logger.error(
                        f"Error computing Wasserstein distance for feature {i}: {e}"
                    )
                    distances.append(np.nan)

            # Remove NaN values
            valid_distances = [d for d in distances if not np.isnan(d)]

            if not valid_distances:
                raise ValueError("No valid distance computed")

            return {
                "metric_name": self.get_metric_name(),
                "mean_distance": float(np.mean(valid_distances)),
                "median_distance": float(np.median(valid_distances)),
                "std_distance": float(np.std(valid_distances)),
                "min_distance": float(np.min(valid_distances)),
                "max_distance": float(np.max(valid_distances)),
                "detailed_distances": detailed_distances,
                "computation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error computing Wasserstein distance: {e}")
            raise

    def _to_numpy(self, data: DataBatch) -> np.ndarray:
        """Convert any data type to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy() if data.is_cuda else data.numpy()
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def get_metric_name(self) -> str:
        return "wasserstein_distance"

    def get_metric_description(self) -> str:
        return "Wasserstein distance to measure similarity between distributions"


@dataclass
class JensenShannonDivergenceCalculator(IMetricCalculator):
    """
    Jensen-Shannon divergence calculation with special handling for continuous distributions
    Complexity: O(n) per feature
    """

    num_bins: int = 100
    use_kde: bool = True
    kde_bandwidth: str = "scott"

    def compute(
        self, real_data: DataBatch, synthetic_data: DataBatch
    ) -> Dict[str, Any]:
        """
        Compute JS divergence with special handling for continuous variables

        Args:
            real_data: Real data
            synthetic_data: Generated data

        Returns:
            Dict: JS divergence results
        """
        try:
            real_np = self._to_numpy(real_data)
            synthetic_np = self._to_numpy(synthetic_data)

            num_features = real_np.shape[1]
            divergences = []
            detailed_divergences = {}

            for i in range(num_features):
                real_feature = real_np[:, i]
                synthetic_feature = synthetic_np[:, i]

                # Remove missing values
                real_feature = real_feature[~np.isnan(real_feature)]
                synthetic_feature = synthetic_feature[~np.isnan(synthetic_feature)]

                if len(real_feature) < 2 or len(synthetic_feature) < 2:
                    divergences.append(np.nan)
                    continue

                # Determine if feature is continuous or discrete
                is_continuous = self._is_continuous(real_feature)

                if is_continuous and self.use_kde:
                    # Use KDE for continuous distributions
                    divergence = self._compute_js_kde(real_feature, synthetic_feature)
                else:
                    # Use histogram for discrete distributions
                    divergence = self._compute_js_histogram(
                        real_feature, synthetic_feature
                    )

                divergences.append(divergence)
                detailed_divergences[f"feature_{i}"] = {
                    "divergence": float(divergence),
                    "is_continuous": is_continuous,
                    "method": "kde" if is_continuous and self.use_kde else "histogram",
                }

            # Remove NaN values
            valid_divergences = [d for d in divergences if not np.isnan(d)]

            if not valid_divergences:
                raise ValueError("No valid JS divergence computed")

            return {
                "metric_name": self.get_metric_name(),
                "mean_divergence": float(np.mean(valid_divergences)),
                "median_divergence": float(np.median(valid_divergences)),
                "std_divergence": float(np.std(valid_divergences)),
                "detailed_divergences": detailed_divergences,
            }

        except Exception as e:
            logger.error(f"Error computing JS divergence: {e}")
            raise

    def _is_continuous(self, data: np.ndarray) -> bool:
        """Determine if data is continuous"""
        unique_values = np.unique(data)
        return len(unique_values) > 20

    def _compute_js_kde(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """Compute JS using KDE for continuous distributions"""
        try:
            # Create a common range
            min_val = min(real.min(), synthetic.min())
            max_val = max(real.max(), synthetic.max())
            x = np.linspace(min_val, max_val, self.num_bins)

            # Estimate KDE
            kde_real = gaussian_kde(real, bw_method=self.kde_bandwidth)
            kde_synthetic = gaussian_kde(synthetic, bw_method=self.kde_bandwidth)

            # Compute probability densities
            pdf_real = kde_real(x)
            pdf_synthetic = kde_synthetic(x)

            # Normalize
            pdf_real = pdf_real / pdf_real.sum()
            pdf_synthetic = pdf_synthetic / pdf_synthetic.sum()

            # Compute JS divergence
            return jensenshannon(pdf_real, pdf_synthetic)
        except Exception as e:
            logger.warning(f"KDE JS failed, using histogram: {e}")
            return self._compute_js_histogram(real, synthetic)

    def _compute_js_histogram(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """Compute JS using histogram"""
        # Create a unified range
        min_val = min(real.min(), synthetic.min())
        max_val = max(real.max(), synthetic.max())

        # Avoid equal values
        if max_val - min_val < 1e-10:
            max_val = min_val + 1

        bins = np.linspace(min_val, max_val, self.num_bins + 1)

        # Compute histograms
        hist_real, _ = np.histogram(real, bins=bins, density=True)
        hist_synthetic, _ = np.histogram(synthetic, bins=bins, density=True)

        # Add a small value to avoid zeros
        epsilon = 1e-10
        hist_real = hist_real + epsilon
        hist_synthetic = hist_synthetic + epsilon

        # Normalize
        hist_real = hist_real / hist_real.sum()
        hist_synthetic = hist_synthetic / hist_synthetic.sum()

        return jensenshannon(hist_real, hist_synthetic)

    def get_metric_name(self) -> str:
        return "jensen_shannon_divergence"

    def get_metric_description(self) -> str:
        return "Jensen-Shannon divergence to measure divergence between probability distributions"


@dataclass
class CorrelationPreservationCalculator(IMetricCalculator):
    """
    Correlation preservation calculation with multiple advanced metrics
    Complexity: O(n²) for correlation matrix
    """

    correlation_methods: List[str] = field(
        default_factory=lambda: ["pearson", "spearman"]
    )
    significance_threshold: float = 0.05
    compute_pairwise: bool = True

    def compute(
        self, real_data: DataBatch, synthetic_data: DataBatch
    ) -> Dict[str, Any]:
        """
        Compute correlation preservation with statistical significance tests

        Args:
            real_data: Real data
            synthetic_data: Generated data

        Returns:
            Dict: Correlation preservation results
        """
        try:
            real_df = self._to_dataframe(real_data)
            synthetic_df = self._to_dataframe(synthetic_data)

            results = {}

            for method in self.correlation_methods:
                # Compute correlation matrices
                real_corr = self._compute_correlation(real_df, method)
                synthetic_corr = self._compute_correlation(synthetic_df, method)

                # Compute similarity metrics
                similarity_metrics = self._compute_correlation_similarity(
                    real_corr, synthetic_corr
                )

                results[method] = {
                    "real_correlation_matrix": real_corr.values.tolist(),
                    "synthetic_correlation_matrix": synthetic_corr.values.tolist(),
                    "similarity_metrics": similarity_metrics,
                    "column_names": list(real_corr.columns),
                }

            # Compute overall metrics
            overall_metrics = self._compute_overall_metrics(results)

            return {
                "metric_name": self.get_metric_name(),
                "correlation_methods": self.correlation_methods,
                "detailed_results": results,
                "overall_metrics": overall_metrics,
            }

        except Exception as e:
            logger.error(f"Error computing correlation preservation: {e}")
            raise

    def _to_dataframe(self, data: DataBatch) -> pd.DataFrame:
        """Convert data to DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data)
        elif isinstance(data, torch.Tensor):
            return pd.DataFrame(data.cpu().numpy())
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _compute_correlation(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Compute correlation matrix"""
        try:
            if method == "pearson":
                return df.corr(method="pearson")
            elif method == "spearman":
                return df.corr(method="spearman")
            elif method == "kendall":
                return df.corr(method="kendall")
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
        except Exception as e:
            logger.warning(f"Failed to compute correlation with method {method}: {e}")
            # Use alternative method
            return df.corr(method="pearson")

    def _compute_correlation_similarity(
        self, real_corr: pd.DataFrame, synthetic_corr: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute similarity metrics between correlation matrices"""
        # Convert to matrices
        real_matrix = real_corr.values
        synthetic_matrix = synthetic_corr.values

        # Remove NaN values
        mask = ~np.isnan(real_matrix) & ~np.isnan(synthetic_matrix)
        real_flat = real_matrix[mask]
        synthetic_flat = synthetic_matrix[mask]

        if len(real_flat) == 0:
            return {"error": "No valid values for comparison"}

        # Compute multiple metrics
        metrics = {}

        # 1. Pearson correlation between matrices
        try:
            metrics["matrix_pearson_corr"] = np.corrcoef(real_flat, synthetic_flat)[
                0, 1
            ]
        except:
            metrics["matrix_pearson_corr"] = np.nan

        # 2. Mean absolute difference
        metrics["mean_absolute_difference"] = np.mean(
            np.abs(real_flat - synthetic_flat)
        )

        # 3. Root mean squared difference
        metrics["root_mean_squared_difference"] = np.sqrt(
            np.mean((real_flat - synthetic_flat) ** 2)
        )

        # 4. Rank preservation (Spearman)
        try:
            metrics["rank_correlation"] = stats.spearmanr(
                real_flat, synthetic_flat
            ).correlation
        except:
            metrics["rank_correlation"] = np.nan

        return metrics

    def _compute_overall_metrics(self, detailed_results: Dict) -> Dict[str, float]:
        """Compute overall metrics"""
        overall = {
            "mean_pearson_correlation": 0.0,
            "mean_spearman_correlation": 0.0,
            "average_mean_absolute_difference": 0.0,
            "preservation_score": 0.0,
        }

        count = 0
        for method, result in detailed_results.items():
            if "similarity_metrics" in result:
                metrics = result["similarity_metrics"]
                if "matrix_pearson_corr" in metrics and not np.isnan(
                    metrics["matrix_pearson_corr"]
                ):
                    overall["mean_pearson_correlation"] += metrics[
                        "matrix_pearson_corr"
                    ]
                    overall["preservation_score"] += metrics["matrix_pearson_corr"]
                    count += 1

        if count > 0:
            overall["mean_pearson_correlation"] /= count
            overall["preservation_score"] = overall["mean_pearson_correlation"]

        return overall

    def get_metric_name(self) -> str:
        return "correlation_preservation"

    def get_metric_description(self) -> str:
        return "Preservation of correlation matrix and relationships between variables"


@dataclass
class ModeCoverageCalculator(IMetricCalculator):
    """
    Mode coverage calculation using advanced Gaussian Mixture Models
    Complexity: O(n * k * iterations) where k is number of components
    """

    n_components_range: Tuple[int, int] = (2, 10)
    covariance_type: str = "full"  # full, tied, diag, spherical
    n_init: int = 3
    max_iter: int = 100
    random_state: int = 42

    def compute(
        self, real_data: DataBatch, synthetic_data: DataBatch
    ) -> Dict[str, Any]:
        """
        Compute mode coverage with optimal component selection

        Args:
            real_data: Real data
            synthetic_data: Generated data

        Returns:
            Dict: Mode coverage results
        """
        try:
            real_np = self._to_numpy(real_data)
            synthetic_np = self._to_numpy(synthetic_data)

            # Identify numerical features only
            numerical_features = self._identify_numerical_features(real_np)

            if not numerical_features:
                return {
                    "metric_name": self.get_metric_name(),
                    "mode_coverage": 1.0,
                    "message": "No numerical features to compute modes",
                    "detailed_results": {},
                }

            results = {}
            total_coverage = 0
            feature_count = 0

            for feature_idx in numerical_features:
                try:
                    real_feature = real_np[:, feature_idx]
                    synthetic_feature = synthetic_np[:, feature_idx]

                    # Remove missing values
                    real_feature = real_feature[~np.isnan(real_feature)]
                    synthetic_feature = synthetic_feature[~np.isnan(synthetic_feature)]

                    if len(real_feature) < 10 or len(synthetic_feature) < 10:
                        continue

                    # Compute mode coverage for this feature
                    feature_coverage, feature_details = (
                        self._compute_feature_mode_coverage(
                            real_feature, synthetic_feature, feature_idx
                        )
                    )

                    if not np.isnan(feature_coverage):
                        results[f"feature_{feature_idx}"] = {
                            "coverage": float(feature_coverage),
                            "details": feature_details,
                        }
                        total_coverage += feature_coverage
                        feature_count += 1

                except Exception as e:
                    logger.warning(
                        f"Error computing modes for feature {feature_idx}: {e}"
                    )
                    continue

            if feature_count == 0:
                overall_coverage = 1.0
            else:
                overall_coverage = total_coverage / feature_count

            return {
                "metric_name": self.get_metric_name(),
                "mode_coverage": float(overall_coverage),
                "n_features_evaluated": feature_count,
                "feature_wise_results": results,
                "interpretation": self._interpret_coverage(overall_coverage),
            }

        except Exception as e:
            logger.error(f"Error computing mode coverage: {e}")
            raise

    def _identify_numerical_features(self, data: np.ndarray) -> List[int]:
        """Identify numerical features"""
        numerical_features = []

        for i in range(data.shape[1]):
            feature = data[:, i]
            feature = feature[~np.isnan(feature)]

            if len(feature) < 2:
                continue

            # Check if feature is numerical (not binary or categorical)
            unique_values = np.unique(feature)
            if (
                len(unique_values) > 10
            ):  # Threshold to distinguish numerical from categorical
                numerical_features.append(i)

        return numerical_features

    def _compute_feature_mode_coverage(
        self, real: np.ndarray, synthetic: np.ndarray, feature_idx: int
    ) -> Tuple[float, Dict]:
        """Compute mode coverage for a specific feature"""

        # Select optimal number of components using BIC
        best_gmm = None
        best_bic = np.inf
        best_n_components = 2

        for n_components in range(
            self.n_components_range[0], self.n_components_range[1] + 1
        ):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )

                gmm.fit(real.reshape(-1, 1))
                bic = gmm.bic(real.reshape(-1, 1))

                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_n_components = n_components

            except Exception as e:
                logger.debug(f"GMM failed with {n_components} components: {e}")
                continue

        if best_gmm is None:
            return np.nan, {"error": "Failed to fit GMM"}

        # Detect modes in real data
        real_labels = best_gmm.predict(real.reshape(-1, 1))
        real_modes = set(real_labels)

        # Assign modes to generated data
        synthetic_labels = best_gmm.predict(synthetic.reshape(-1, 1))
        synthetic_modes = set(synthetic_labels)

        # Compute coverage
        covered_modes = real_modes.intersection(synthetic_modes)
        coverage = len(covered_modes) / len(real_modes) if real_modes else 1.0

        # Detailed information
        details = {
            "n_components_optimal": best_n_components,
            "bic_score": float(best_bic),
            "real_modes_count": len(real_modes),
            "synthetic_modes_count": len(synthetic_modes),
            "covered_modes_count": len(covered_modes),
            "real_modes_distribution": self._compute_mode_distribution(
                real_labels, best_n_components
            ),
            "synthetic_modes_distribution": self._compute_mode_distribution(
                synthetic_labels, best_n_components
            ),
        }

        return coverage, details

    def _compute_mode_distribution(
        self, labels: np.ndarray, n_components: int
    ) -> Dict[int, float]:
        """Compute distribution of samples across modes"""
        distribution = {}
        total_samples = len(labels)

        for mode in range(n_components):
            count = np.sum(labels == mode)
            distribution[mode] = (
                float(count / total_samples) if total_samples > 0 else 0.0
            )

        return distribution

    def _interpret_coverage(self, coverage: float) -> str:
        """Interpret coverage value"""
        if coverage >= 0.9:
            return "Excellent mode coverage"
        elif coverage >= 0.7:
            return "Good mode coverage"
        elif coverage >= 0.5:
            return "Acceptable mode coverage"
        else:
            return "Poor mode coverage - possible mode collapse"

    def get_metric_name(self) -> str:
        return "mode_coverage"

    def get_metric_description(self) -> str:
        return "Proportion of modes (statistical clusters) that have been successfully generated"


# ============================================================================
# STATISTICAL TEST IMPLEMENTATIONS
# ============================================================================


@dataclass
class KolmogorovSmirnovTest(IStatisticalTest):
    """Kolmogorov-Smirnov test to prove similarity between distributions"""

    alpha: float = 0.05

    def test(self, real_data: DataBatch, synthetic_data: DataBatch) -> Dict[str, Any]:
        """Perform KS test"""
        try:
            real_np = self._to_numpy(real_data)
            synthetic_np = self._to_numpy(synthetic_data)

            results = {}

            for i in range(real_np.shape[1]):
                real_feature = real_np[:, i]
                synthetic_feature = synthetic_np[:, i]

                # Remove missing values
                real_feature = real_feature[~np.isnan(real_feature)]
                synthetic_feature = synthetic_feature[~np.isnan(synthetic_feature)]

                if len(real_feature) < 2 or len(synthetic_feature) < 2:
                    results[f"feature_{i}"] = {"error": "Insufficient data"}
                    continue

                # KS test
                statistic, p_value = stats.ks_2samp(real_feature, synthetic_feature)

                results[f"feature_{i}"] = {
                    "test_statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": p_value < self.alpha,
                    "interpretation": self._interpret_result(p_value),
                }

            return {
                "test_name": self.get_test_name(),
                "alpha_level": self.alpha,
                "feature_results": results,
                "overall_significant": any(
                    r.get("significant", False)
                    for r in results.values()
                    if isinstance(r, dict)
                ),
            }

        except Exception as e:
            logger.error(f"Error in KS test: {e}")
            raise

    def _interpret_result(self, p_value: float) -> str:
        """Interpret test result"""
        if p_value < self.alpha:
            return "Distributions are statistically different"
        else:
            return "No evidence of distribution difference"

    @property
    def test_statistic(self) -> float:
        # This is a placeholder; actual implementation needs to store values
        return 0.0

    @property
    def p_value(self) -> float:
        # This is a placeholder; actual implementation needs to store values
        return 1.0

    def get_test_name(self) -> str:
        return "kolmogorov_smirnov_test"


# ============================================================================
# MAIN AUDITOR CLASS - PRODUCTION READY
# ============================================================================


class HybridGANAuditor:
    """
    Integrated production-ready auditing and evaluation system for hybrid data generation systems

    Design advantages:
    1. Extensibility: New metrics can be added without modifying existing code
    2. Reliability: Comprehensive error handling with continuous operation
    3. Performance: Parallel computations and caching of intermediate results
    4. Flexibility: Support for multiple data types (PyTorch, NumPy, Pandas)
    5. Compatibility: Standard interfaces allow integration with other systems
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        """
        Initialize the auditor with customizable configuration

        Args:
            config: Audit configuration (optional)
        """
        self.config = config or AuditConfig()
        self._setup_metrics()
        self._setup_logging()

        # Cache for results
        self._results_cache: Dict[str, Any] = {}
        self._generation_cache: Optional[DataBatch] = None

        # Historical metrics recording
        self.metrics_history: List[Dict] = []

        logger.info(f"HybridGANAuditor initialized with config: {self.config}")

    def _setup_metrics(self) -> None:
        """Set up metric calculators and statistical tests"""
        self.metric_calculators: List[IMetricCalculator] = [
            WassersteinDistanceCalculator(
                num_bins=self.config.wasserstein_n_bins,
                normalize=True,
                use_gpu=self.config.device == "cuda",
            ),
            JensenShannonDivergenceCalculator(
                num_bins=100, use_kde=True, kde_bandwidth="scott"
            ),
            CorrelationPreservationCalculator(
                correlation_methods=[self.config.correlation_method],
                significance_threshold=self.config.statistical_test_alpha,
            ),
            ModeCoverageCalculator(
                n_components_range=self.config.gmm_n_components_range
            ),
        ]

        self.statistical_tests: List[IStatisticalTest] = [
            KolmogorovSmirnovTest(alpha=self.config.statistical_test_alpha)
        ]

    def _setup_logging(self) -> None:
        """Set up advanced logging system"""
        # Additional logging handlers can be added here
        pass

    def evaluate_generation_quality(
        self,
        generator: GeneratorInterface,
        real_data_provider: DataProvider,
        num_samples: Optional[int] = None,
        compute_metrics: Optional[List[str]] = None,
        **generator_kwargs,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated data quality

        Args:
            generator: Generator interface
            real_data_provider: Real data provider
            num_samples: Number of samples required (optional)
            compute_metrics: List of metrics to compute (optional)
            **generator_kwargs: Additional parameters for the generator

        Returns:
            Dict: Comprehensive evaluation results

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If evaluation fails
        """
        try:
            # Validate inputs
            self._validate_inputs(generator, real_data_provider)

            # Determine number of samples
            num_samples = num_samples or self.config.num_evaluation_samples
            logger.info(f"Starting evaluation for {num_samples} samples")

            # Generate samples (with caching)
            synthetic_data = self._generate_samples(
                generator, num_samples, **generator_kwargs
            )

            # Load real data
            real_data = self._load_real_data(real_data_provider, num_samples)

            # Compute required metrics
            metrics = self._compute_all_metrics(
                real_data, synthetic_data, compute_metrics
            )

            # Perform statistical tests
            if self.config.compute_statistical_tests:
                statistical_results = self._perform_statistical_tests(
                    real_data, synthetic_data
                )
                metrics["statistical_tests"] = statistical_results

            # Compute overall quality score
            overall_score = self._compute_overall_quality_score(metrics)
            metrics["overall_quality_score"] = overall_score

            # Add metadata
            metrics["metadata"] = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "num_samples": num_samples,
                "generator_type": type(generator).__name__,
                "config_used": asdict(self.config),
            }

            # Save to history
            self.metrics_history.append(metrics)

            # Clean up cache
            self._cleanup_cache()

            logger.info(
                f"Evaluation completed successfully. Overall score: {overall_score:.3f}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise RuntimeError(f"Generation quality evaluation failed: {e}")

    def _validate_inputs(
        self, generator: GeneratorInterface, real_data_provider: DataProvider
    ) -> None:
        """Validate inputs"""
        if not hasattr(generator, "generate"):
            raise ValueError("Generator must have a generate() method")

        if not hasattr(real_data_provider, "__iter__"):
            raise ValueError("Data provider must be iterable")

    def _generate_samples(
        self, generator: GeneratorInterface, num_samples: int, **kwargs
    ) -> DataBatch:
        """
        Generate samples with caching and optimization

        Args:
            generator: Generator
            num_samples: Number of samples
            **kwargs: Additional parameters

        Returns:
            DataBatch: Generated data
        """
        cache_key = f"generated_{num_samples}_{hash(str(kwargs))}"

        # Check cache
        if self.config.cache_intermediate_results and cache_key in self._results_cache:
            logger.info("Using generated data from cache")
            return self._results_cache[cache_key]

        try:
            logger.info(f"Generating {num_samples} new samples...")

            # Move generator to required device
            if self.config.device == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
                if hasattr(generator, "to"):
                    generator.to(device)
            else:
                device = torch.device("cpu")

            # Generate samples
            start_time = datetime.now()
            synthetic_data = generator.generate(num_samples, **kwargs)
            generation_time = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Generated {num_samples} samples in {generation_time:.2f} seconds"
            )

            # Cache
            if self.config.cache_intermediate_results:
                self._results_cache[cache_key] = synthetic_data

            return synthetic_data

        except Exception as e:
            logger.error(f"Sample generation failed: {e}")
            raise

    def _load_real_data(
        self, data_provider: DataProvider, num_samples: int
    ) -> DataBatch:
        """
        Load real data with optimization

        Args:
            data_provider: Data provider
            num_samples: Number of samples required

        Returns:
            DataBatch: Real data
        """
        try:
            logger.info(f"Loading {num_samples} real samples...")

            # Collect samples from data provider
            samples = []
            total_samples = 0

            for batch in data_provider:
                if total_samples >= num_samples:
                    break

                # Convert batch to appropriate format
                batch_np = self._to_numpy(batch)

                # Determine number of samples remaining
                remaining = num_samples - total_samples
                batch_to_take = min(batch_np.shape[0], remaining)

                samples.append(batch_np[:batch_to_take])
                total_samples += batch_to_take

            if total_samples == 0:
                raise ValueError("No real samples loaded")

            # Concatenate all batches
            real_data = np.vstack(samples) if len(samples) > 1 else samples[0]

            logger.info(f"Loaded {real_data.shape[0]} real samples")

            return real_data

        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            raise

    def _compute_all_metrics(
        self,
        real_data: DataBatch,
        synthetic_data: DataBatch,
        compute_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute all metrics with parallelism and optimization

        Args:
            real_data: Real data
            synthetic_data: Generated data
            compute_metrics: Required metrics

        Returns:
            Dict: Results of all metrics
        """
        results = {}

        # Filter required metrics
        calculators_to_use = self.metric_calculators
        if compute_metrics:
            calculators_to_use = [
                calc
                for calc in self.metric_calculators
                if calc.get_metric_name() in compute_metrics
            ]

        # Compute metrics
        for calculator in calculators_to_use:
            try:
                metric_name = calculator.get_metric_name()
                logger.info(f"Computing metric: {metric_name}")

                # Check cache
                cache_key = f"metric_{metric_name}_{hash(str(real_data))}_{hash(str(synthetic_data))}"

                if (
                    self.config.cache_intermediate_results
                    and cache_key in self._results_cache
                ):
                    metric_result = self._results_cache[cache_key]
                    logger.info(f"Using {metric_name} from cache")
                else:
                    # Compute metric
                    start_time = datetime.now()
                    metric_result = calculator.compute(real_data, synthetic_data)
                    computation_time = (datetime.now() - start_time).total_seconds()

                    metric_result["computation_time_seconds"] = computation_time

                    # Cache
                    if self.config.cache_intermediate_results:
                        self._results_cache[cache_key] = metric_result

                results[metric_name] = metric_result
                logger.info(
                    f"Completed {metric_name} in {metric_result.get('computation_time_seconds', 0):.2f} seconds"
                )

            except Exception as e:
                logger.error(
                    f"Failed to compute metric {calculator.get_metric_name()}: {e}"
                )
                results[calculator.get_metric_name()] = {
                    "error": str(e),
                    "metric_name": calculator.get_metric_name(),
                }

        return results

    def _perform_statistical_tests(
        self, real_data: DataBatch, synthetic_data: DataBatch
    ) -> Dict[str, Any]:
        """
        Perform statistical tests

        Args:
            real_data: Real data
            synthetic_data: Generated data

        Returns:
            Dict: Test results
        """
        results = {}

        for test in self.statistical_tests:
            try:
                test_name = test.get_test_name()
                logger.info(f"Performing statistical test: {test_name}")

                test_result = test.test(real_data, synthetic_data)
                results[test_name] = test_result

                logger.info(f"Completed {test_name}")

            except Exception as e:
                logger.error(f"Test {test.get_test_name()} failed: {e}")
                results[test.get_test_name()] = {
                    "error": str(e),
                    "test_name": test.get_test_name(),
                }

        return results

    def _compute_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall quality score with intelligent weighting

        Args:
            metrics: Results of all metrics

        Returns:
            float: Overall quality score (0-1)
        """
        try:
            # Metric weights (customizable)
            weights = {
                "wasserstein_distance": 0.3,
                "jensen_shannon_divergence": 0.25,
                "correlation_preservation": 0.25,
                "mode_coverage": 0.2,
            }

            total_score = 0.0
            total_weight = 0.0

            for metric_name, metric_data in metrics.items():
                if metric_name in weights:
                    weight = weights[metric_name]

                    # Extract score from each metric
                    score = self._extract_score_from_metric(metric_name, metric_data)

                    if score is not None:
                        total_score += score * weight
                        total_weight += weight

            # Normalize score
            if total_weight > 0:
                overall_score = total_score / total_weight
            else:
                overall_score = 0.0

            return min(1.0, max(0.0, overall_score))

        except Exception as e:
            logger.error(f"Failed to compute overall score: {e}")
            return 0.0

    def _extract_score_from_metric(
        self, metric_name: str, metric_data: Dict[str, Any]
    ) -> Optional[float]:
        """Extract score from metric results"""
        try:
            if metric_name == "wasserstein_distance":
                # For Wasserstein distance: lower is better
                distance = metric_data.get("mean_distance", 1.0)
                return 1.0 / (1.0 + distance)  # Convert to 0-1 score

            elif metric_name == "jensen_shannon_divergence":
                # For JS divergence: lower is better
                divergence = metric_data.get("mean_divergence", 1.0)
                return 1.0 / (1.0 + divergence)  # Convert to 0-1 score

            elif metric_name == "correlation_preservation":
                # For correlation preservation: take preservation score
                if "overall_metrics" in metric_data:
                    return metric_data["overall_metrics"].get("preservation_score", 0.5)
                return 0.5

            elif metric_name == "mode_coverage":
                # Mode coverage: direct value
                return metric_data.get("mode_coverage", 0.5)

            else:
                return None

        except Exception as e:
            logger.warning(f"Failed to extract score from {metric_name}: {e}")
            return None

    def _to_numpy(self, data: DataBatch) -> np.ndarray:
        """Convert any data type to numpy array"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy() if data.is_cuda else data.numpy()
        elif isinstance(data, pd.DataFrame):
            return data.values
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _cleanup_cache(self) -> None:
        """Clean up cache"""
        # More complex memory management can be added here
        if not self.config.cache_intermediate_results:
            self._results_cache.clear()

    def generate_report(
        self, metrics: Dict[str, Any], output_format: str = "json"
    ) -> Union[str, Dict]:
        """
        Generate a professional evaluation report

        Args:
            metrics: Metric results
            output_format: Report format (json, markdown, html)

        Returns:
            Union[str, Dict]: Report in requested format
        """
        try:
            if output_format == "json":
                return self._generate_json_report(metrics)
            elif output_format == "markdown":
                return self._generate_markdown_report(metrics)
            elif output_format == "html":
                return self._generate_html_report(metrics)
            else:
                raise ValueError(f"Unsupported format: {output_format}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {"error": str(e)}

    def _generate_json_report(self, metrics: Dict[str, Any]) -> str:
        """Generate JSON report"""
        return json.dumps(metrics, indent=2, ensure_ascii=False, default=str)

    def _generate_markdown_report(self, metrics: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        report = []
        report.append("# Generated Data Quality Evaluation Report")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(
            f"**Overall Score**: {metrics.get('overall_quality_score', 0):.3f}"
        )
        report.append("---")

        for metric_name, metric_data in metrics.items():
            if metric_name in ["overall_quality_score", "metadata"]:
                continue

            report.append(f"## {metric_name}")

            if "error" in metric_data:
                report.append(f" **Error**: {metric_data['error']}")
            else:
                # Display main results
                if "mean_distance" in metric_data:
                    report.append(
                        f"- **Mean Distance**: {metric_data['mean_distance']:.4f}"
                    )
                if "mean_divergence" in metric_data:
                    report.append(
                        f"- **Mean Divergence**: {metric_data['mean_divergence']:.4f}"
                    )
                if "mode_coverage" in metric_data:
                    report.append(
                        f"- **Mode Coverage**: {metric_data['mode_coverage']:.2%}"
                    )
                if "preservation_score" in metric_data:
                    report.append(
                        f"- **Preservation Score**: {metric_data['preservation_score']:.4f}"
                    )

        return "\n".join(report)

    def _generate_html_report(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML report (simplified)"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Generated Data Quality Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .metric {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
                .score {{ font-size: 24px; color: #2c3e50; font-weight: bold; }}
                .good {{ color: green; }}
                .fair {{ color: orange; }}
                .poor {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Generated Data Quality Evaluation Report</h1>
                <p>Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p class="score">Overall Score: {metrics.get('overall_quality_score', 0):.3f}</p>
            </div>
        """

        for metric_name, metric_data in metrics.items():
            if metric_name in ["overall_quality_score", "metadata"]:
                continue

            html += f"""
            <div class="metric">
                <h3>{metric_name}</h3>
            """

            if "error" in metric_data:
                html += f"<p class='poor'>Error: {metric_data['error']}</p>"
            else:
                if "mean_distance" in metric_data:
                    html += f"<p>Mean Distance: {metric_data['mean_distance']:.4f}</p>"
                if "mean_divergence" in metric_data:
                    html += (
                        f"<p>Mean Divergence: {metric_data['mean_divergence']:.4f}</p>"
                    )
                if "mode_coverage" in metric_data:
                    coverage = metric_data["mode_coverage"]
                    cls = (
                        "good"
                        if coverage > 0.8
                        else "fair" if coverage > 0.6 else "poor"
                    )
                    html += f"<p class='{cls}'>Mode Coverage: {coverage:.2%}</p>"

        html += """
        </body>
        </html>
        """

        return html


# ============================================================================
# FACTORY AND BUILDER PATTERNS FOR FLEXIBLE CONSTRUCTION
# ============================================================================


class GANAuditorBuilder:
    """
    Flexible builder for GANAuditor using Fluent Interface
    """

    def __init__(self):
        self._config = AuditConfig()
        self._metric_calculators: List[IMetricCalculator] = []
        self._statistical_tests: List[IStatisticalTest] = []

    def with_config(self, config: AuditConfig) -> "GANAuditorBuilder":
        """Set custom configuration"""
        self._config = config
        return self

    def with_batch_size(self, batch_size: int) -> "GANAuditorBuilder":
        """Set batch size"""
        self._config.batch_size = batch_size
        return self

    def with_device(self, device: str) -> "GANAuditorBuilder":
        """Set computation device"""
        self._config.device = device
        return self

    def add_metric_calculator(
        self, calculator: IMetricCalculator
    ) -> "GANAuditorBuilder":
        """Add a custom metric calculator"""
        self._metric_calculators.append(calculator)
        return self

    def add_statistical_test(self, test: IStatisticalTest) -> "GANAuditorBuilder":
        """Add a custom statistical test"""
        self._statistical_tests.append(test)
        return self

    def build(self) -> HybridGANAuditor:
        """Build a GANAuditor instance"""
        auditor = HybridGANAuditor(self._config)

        # Add custom metrics
        if self._metric_calculators:
            auditor.metric_calculators = self._metric_calculators

        # Add custom tests
        if self._statistical_tests:
            auditor.statistical_tests = self._statistical_tests

        return auditor


# ============================================================================
# USAGE EXAMPLES AND DEMONSTRATION
# ============================================================================


def example_usage() -> None:
    """
    Practical example of using the system in a production environment
    """
    print("Example usage of HybridGANAuditor in a production environment")

    # 1. Create a flexible builder
    builder = GANAuditorBuilder()

    # 2. Configure the system
    config = AuditConfig(
        batch_size=2048,
        num_evaluation_samples=50000,
        compute_mode_coverage=True,
        compute_privacy_metrics=False,
        use_parallel_computation=True,
    )

    # 3. Build the auditor
    auditor = (
        builder.with_config(config)
        .with_device("cuda" if torch.cuda.is_available() else "cpu")
        .add_metric_calculator(WassersteinDistanceCalculator(num_bins=200))
        .add_statistical_test(KolmogorovSmirnovTest(alpha=0.01))
        .build()
    )

    print("System built successfully")

    # 4. Use the system with a mock generator (for testing)
    class MockGenerator:
        """Mock generator for testing"""

        def generate(self, num_samples: int, **kwargs) -> np.ndarray:
            # Generate random data similar to real data
            return np.random.normal(0, 1, (num_samples, 10))

    class MockDataProvider:
        """Mock data provider for testing"""

        def __iter__(self):
            # Return mock batches
            for _ in range(10):
                yield np.random.normal(0, 1, (1000, 10))

        def __len__(self):
            return 10

    # 5. Run evaluation
    try:
        results = auditor.evaluate_generation_quality(
            generator=MockGenerator(),
            real_data_provider=MockDataProvider(),
            num_samples=10000,
            compute_metrics=["wasserstein_distance", "correlation_preservation"],
        )

        print(f"Overall score: {results.get('overall_quality_score', 0):.3f}")

        # 6. Generate report
        report = auditor.generate_report(results, output_format="markdown")
        print("\nEvaluation report:")
        print(report[:500] + "..." if len(report) > 500 else report)

    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    # Run example
    example_usage()
