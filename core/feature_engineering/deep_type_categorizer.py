"""
core/feature_engineering/deep_type_categorizer.py
Deep Feature Categorization System - Version 1.0
Discovers complex types and transforms them into optimal mathematical representation for neural networks
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
import warnings

warnings.filterwarnings("ignore")


class DeepFeatureAnalyzer:
    """
    Advanced feature analysis and categorization system
    """

    # Advanced mathematical type definitions
    FEATURE_TYPES = {
        "CONTINUOUS_GAUSSIAN": "Continuous Gaussian distribution",
        "CONTINUOUS_MULTIMODAL": "Continuous multimodal",
        "CONTINUOUS_HEAVY_TAILED": "Heavy-tailed continuous",
        "DISCRETE_UNIFORM": "Discrete uniform",
        "DISCRETE_POWER_LAW": "Discrete power law",
        "CATEGORICAL_LOW_CARDINALITY": "Categorical with low cardinality (<10)",
        "CATEGORICAL_HIGH_CARDINALITY": "Categorical with high cardinality (>=10)",
        "ORDINAL": "Ordinal",
        "CYCLIC": "Cyclic (e.g., time, angles)",
        "COMPOSITE": "Composite (requires multi-dimensional analysis)",
    }

    def __init__(self, significance_level=0.05, multimodality_threshold=0.3):
        """
        Initialize the advanced analyzer

        Args:
            significance_level: Statistical significance level for tests
            multimodality_threshold: Multimodality detection threshold
        """
        self.significance_level = significance_level
        self.multimodality_threshold = multimodality_threshold
        self.feature_metadata = {}
        self.transformers = {}

    def analyze_column(self, column_data, column_name):
        """
        Deep column analysis using multiple statistical tests

        Returns:
            dict: All mathematical properties of the column
        """
        # Remove missing values for analysis
        clean_data = column_data.dropna()

        if len(clean_data) == 0:
            return {"type": "UNDEFINED", "message": "No valid data"}

        # Check basic type
        is_numeric = pd.api.types.is_numeric_dtype(clean_data)

        metadata = {
            "name": column_name,
            "original_dtype": str(column_data.dtype),
            "sample_count": len(clean_data),
            "missing_rate": column_data.isna().mean(),
            "is_numeric": is_numeric,
        }

        if is_numeric:
            metadata.update(self._analyze_numeric_column(clean_data))
        else:
            metadata.update(self._analyze_categorical_column(clean_data))

        # Determine final type using advanced decision tree
        metadata["deep_type"] = self._determine_deep_type(metadata)

        # Calculate optimal transformation distribution
        metadata["optimal_transformation"] = self._determine_optimal_transformation(
            metadata
        )

        self.feature_metadata[column_name] = metadata
        return metadata

    def _analyze_numeric_column(self, data):
        """Deep analysis of numeric column"""
        results = {}

        # Basic statistics
        results["mean"] = float(data.mean())
        results["std"] = float(data.std())
        results["skewness"] = float(stats.skew(data))
        results["kurtosis"] = float(stats.kurtosis(data))
        results["min"] = float(data.min())
        results["max"] = float(data.max())

        results["unique_count"] = len(data.unique())
        # Special case tests
        results["is_integer"] = all(data.apply(lambda x: x == int(x)))
        results["is_positive"] = all(data > 0)
        results["is_bounded"] = results["min"] > -np.inf and results["max"] < np.inf

        # Statistical distribution tests
        results.update(self._run_distribution_tests(data))

        # Multimodality analysis using Kernel Density Estimation
        results["multimodality_score"] = self._calculate_multimodality_score(data)
        results["is_multimodal"] = (
            results["multimodality_score"] > self.multimodality_threshold
        )

        # Tail analysis
        results["tail_behavior"] = self._analyze_tail_behavior(data)

        return results

    def _run_distribution_tests(self, data):
        """Run a suite of distribution tests"""
        tests = {}

        # Shapiro-Wilk test for normality (for small samples)
        if len(data) <= 5000:
            _, shapiro_p = stats.shapiro(data)
            tests["shapiro_p"] = shapiro_p
            tests["is_gaussian"] = shapiro_p > self.significance_level

        # Anderson-Darling test
        anderson_result = stats.anderson(data)
        tests["anderson_statistic"] = anderson_result.statistic

        # Kolmogorov-Smirnov test for normal distribution
        ks_stat, ks_p = stats.kstest(data, "norm", args=(data.mean(), data.std()))
        tests["ks_statistic"] = ks_stat
        tests["ks_p_value"] = ks_p

        return tests

    def _calculate_multimodality_score(self, data):
        """Calculate multimodality score using Hartigan's Dip Test"""
        try:
            from scipy.stats import diptest

            dip_stat, _ = diptest(data)
            return dip_stat
        except:
            # Alternative approach - Silverman's bandwidth test
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x = np.linspace(data.min(), data.max(), 1000)
            density = kde(x)

            # Find local peaks
            from scipy.signal import find_peaks

            peaks, _ = find_peaks(density, height=0.01)

            # Multimodality score depends on number and clarity of peaks
            return min(1.0, len(peaks) * 0.3)

    def _analyze_tail_behavior(self, data):
        """Analyze tail behavior using extreme value theory"""
        from scipy.stats import genpareto

        # Estimate tail coefficient
        threshold = np.percentile(data, 90)
        excess = data[data > threshold] - threshold

        if len(excess) > 10:
            try:
                params = genpareto.fit(excess)
                tail_index = params[0]  # Tail coefficient

                if tail_index < 0.3:
                    return "LIGHT_TAILED"
                elif tail_index < 1:
                    return "MODERATE_TAILED"
                else:
                    return "HEAVY_TAILED"
            except:
                return "UNKNOWN"
        return "INSUFFICIENT_DATA"

    def _calculate_categorical_entropy(self, value_counts):
        """Calculate entropy for categorical distribution"""
        probs = value_counts / value_counts.sum()
        entropy = -np.sum(
            probs * np.log2(probs + 1e-12)
        )  # Small addition to avoid log(0)
        return float(entropy)

    def _detect_hierarchy(self, data):
        """
        Analyze presence of hierarchy in categorical data.

        Idea:
        - If values contain delimiters (e.g., 'Finance>Banking' or 'Region/Subregion') this indicates levels.
        - If there are recurring patterns suggesting internal subgroups.
        - If frequency distribution shows tree-like structure (some categories contain many sub-values).

        Returns:
            bool: Whether hierarchy exists
            dict: Details of hierarchical structure if found
        """
        hierarchy_info = {
            "has_levels": False,
            "levels_detected": 0,
            "example_pattern": None,
        }

        try:
            unique_vals = data.dropna().unique()
            str_vals = [str(v) for v in unique_vals]

            # 1. Look for common delimiters indicating levels
            delimiters = [">", "/", "|", ":"]
            for delim in delimiters:
                if any(delim in val for val in str_vals):
                    hierarchy_info["has_levels"] = True
                    # Calculate number of levels based on first value
                    hierarchy_info["levels_detected"] = max(
                        len(val.split(delim)) for val in str_vals
                    )
                    hierarchy_info["example_pattern"] = next(
                        val for val in str_vals if delim in val
                    )
                    return hierarchy_info

            # 2. If no delimiters, look for recurring patterns (e.g., "HR_1", "HR_2")
            import re

            pattern_detected = False
            for val in str_vals:
                if re.search(r"[A-Za-z]+[_\-]\d+", val):
                    pattern_detected = True
                    hierarchy_info["has_levels"] = True
                    hierarchy_info["levels_detected"] = 2
                    hierarchy_info["example_pattern"] = val
                    break

            if pattern_detected:
                return hierarchy_info

            # 3. Analyze frequency distribution: if some categories have very high count compared to others
            value_counts = data.value_counts()
            if value_counts.max() > 5 * value_counts.median():
                hierarchy_info["has_levels"] = True
                hierarchy_info["levels_detected"] = 2
                hierarchy_info["example_pattern"] = value_counts.index[0]

            return hierarchy_info

        except Exception:
            return hierarchy_info

    def _analyze_categorical_column(self, data):
        """Analyze categorical column"""
        results = {}

        unique_values = data.unique()
        value_counts = data.value_counts()

        results["unique_count"] = len(unique_values)
        results["cardinality"] = len(unique_values) / len(data)
        results["most_frequent"] = value_counts.index[0]
        results["frequency_of_most_frequent"] = value_counts.iloc[0] / len(data)

        # Test if ordinal
        results["is_ordinal"] = self._test_ordinality(data)

        # Frequency distribution analysis
        results["entropy"] = self._calculate_categorical_entropy(value_counts)
        results["gini_impurity"] = 1 - ((value_counts / len(data)) ** 2).sum()

        # Hierarchy analysis
        results["has_hierarchy"] = self._detect_hierarchy(data)

        return results

    def _test_cyclicity(self, metadata: dict) -> bool:
        """
        Professional cyclicity test for a column.
        Consider column cyclic if:
        - Its name or metadata suggests time cycle (month, day_of_week, season, hour).
        - Or if numeric values fall within known cycle ranges (7 days, 12 months, 24 hours).
        """
        col_name = metadata.get("column_name", "").lower()
        unique_count = metadata.get("unique_count", 0)
        values = metadata.get("unique_values", [])

        # 1. Common cyclic keywords
        cyclic_keywords = [
            "month",
            "day",
            "day_of_week",
            "weekday",
            "season",
            "hour",
            "minute",
        ]
        if any(kw in col_name for kw in cyclic_keywords):
            return True

        # 2. Known numeric ranges
        if metadata.get("is_integer", False):
            if unique_count <= 12 and all(
                1 <= v <= 12 for v in values if isinstance(v, (int, float))
            ):
                return True  # Months
            if unique_count <= 7 and all(
                0 <= v <= 6 for v in values if isinstance(v, (int, float))
            ):
                return True  # Days of week
            if unique_count <= 24 and all(
                0 <= v <= 23 for v in values if isinstance(v, (int, float))
            ):
                return True  # Hours of day

        # 3. fallback: not cyclic
        return False

    def _test_ordinality(self, data):
        """Test if data is ordinal"""
        try:
            # Attempt to convert to numbers
            numeric_data = pd.to_numeric(data, errors="coerce")
            if numeric_data.notna().all():
                # Test if values are approximately equally spaced
                unique_values = np.sort(numeric_data.unique())
                diffs = np.diff(unique_values)
                if np.std(diffs) / np.mean(diffs) < 0.3:
                    return True
        except:
            pass
        return False

    # Inside DeepFeatureAnalyzer

    def _test_power_law(self, metadata: dict, config: dict = None) -> dict:
        """
        Professional power-law distribution test based on frequency distribution.
        Uses Clauset–Shalizi–Newman methodology for selecting x_min and MLE estimation of α,
        calculates KS, with optional lightweight Bootstrap for approximate p-value.

        Returns:
            dict: {
                'is_power_law': bool,
                'alpha': float or None,
                'x_min': int or None,
                'ks': float or None,
                'p_value': float or None,
                'tail_fraction': float,
                'support_size': int,
                'n_tail': int,
                'diagnostics': { ... messages and details ... }
            }
        """
        import numpy as np
        import pandas as pd

        # Default adjustable settings
        cfg = {
            "min_support_size": 50,  # Minimum number of categories before considering Power Law
            "min_tail_fraction": 0.05,  # Minimum proportion of data in acceptable tail
            "min_n_tail": 100,  # Minimum number of values in tail
            "max_candidates": 1000,  # Upper limit for number of x_min candidates
            "bootstrap": True,  # Enable bootstrap for approximate p-value calculation
            "bootstrap_runs": 200,  # Number of bootstrap replicates (lightweight to avoid cost)
            "seed": 42,
            "alpha_bounds": (1.2, 5.0),  # Realistic α range for real-world data
            "ks_threshold_good": 0.1,  # KS below this is typically good
            "ks_threshold_strict": 0.2,  # KS above this is often poor
        }
        if config and isinstance(config, dict):
            cfg.update(config)

        rng = np.random.default_rng(cfg["seed"])

        # Attempt to get frequency distribution from metadata
        # Expect 'value_counts' as a series containing frequencies for each value/category
        counts_series = metadata.get("value_counts", None)

        # If not present, try to infer from raw data if available
        raw_series = metadata.get("raw_series", None)
        if counts_series is None and raw_series is not None:
            try:
                counts_series = pd.Series(raw_series).value_counts(dropna=False)
            except Exception:
                counts_series = None

        # Failed to obtain frequencies
        if counts_series is None or len(counts_series) == 0:
            return {
                "is_power_law": False,
                "alpha": None,
                "x_min": None,
                "ks": None,
                "p_value": None,
                "tail_fraction": 0.0,
                "support_size": 0,
                "n_tail": 0,
                "diagnostics": {"reason": "no_counts_available"},
            }

        # Work on frequency distribution: we want to analyze the distribution of frequencies themselves
        # Example: if values appear 1, 2, 10 times... analyze distribution of these frequencies
        freqs = counts_series.astype(int).values
        freqs = freqs[freqs > 0]

        support_size = int(len(freqs))

        if support_size < cfg["min_support_size"]:
            return {
                "is_power_law": False,
                "alpha": None,
                "x_min": None,
                "ks": None,
                "p_value": None,
                "tail_fraction": 0.0,
                "support_size": support_size,
                "n_tail": 0,
                "diagnostics": {"reason": "insufficient_support"},
            }

        # Candidates for x_min: try unique frequency values with upper limit
        candidates = np.unique(freqs)
        candidates.sort()
        if len(candidates) > cfg["max_candidates"]:
            candidates = candidates[
                -cfg["max_candidates"] :
            ]  # Prioritize larger values

        # MLE α estimation function (continuous approximation) for data > x_min
        def mle_alpha_continuous(x_tail, x_min):
            # α_hat = 1 + n / sum(log(x_i / x_min))
            n = len(x_tail)
            if n == 0:
                return None
            s = np.sum(np.log(x_tail / float(x_min)))
            if s <= 0:
                return None
            return 1.0 + n / s

        # Empirical CDF:
        def empirical_cdf(x_tail):
            x_sorted = np.sort(x_tail)
            n = len(x_sorted)
            # cdf value at each support point
            uniq = np.unique(x_sorted)
            cdf_vals = np.array([np.mean(x_sorted <= u) for u in uniq])
            return uniq, cdf_vals

        # Power-law model CDF (continuous): F(x) = 1 - (x / x_min)^{1-α}, x >= x_min
        def model_cdf_continuous(x_vals, alpha, x_min):
            # If α <= 1 model is undefined, return None
            if alpha is None or alpha <= 1.0:
                return None
            return 1.0 - (x_vals.astype(float) / float(x_min)) ** (1.0 - alpha)

        # Scan candidates to select x_min that minimizes KS
        best = {"x_min": None, "alpha": None, "ks": np.inf, "n_tail": 0}

        total_n = len(freqs)
        for x_min in candidates:
            x_tail = freqs[freqs >= x_min]
            n_tail = len(x_tail)
            tail_fraction = n_tail / total_n

            if n_tail < cfg["min_n_tail"] or tail_fraction < cfg["min_tail_fraction"]:
                continue

            alpha_hat = mle_alpha_continuous(x_tail, x_min)
            if alpha_hat is None or not np.isfinite(alpha_hat):
                continue

            # Calculate KS between empirical and model CDF
            support, ecdf = empirical_cdf(x_tail)
            mcdf = model_cdf_continuous(support, alpha_hat, x_min)
            if mcdf is None:
                continue

            ks = float(np.max(np.abs(ecdf - mcdf)))

            if ks < best["ks"]:
                best = {
                    "x_min": int(x_min),
                    "alpha": float(alpha_hat),
                    "ks": ks,
                    "n_tail": int(n_tail),
                }

        if best["x_min"] is None:
            return {
                "is_power_law": False,
                "alpha": None,
                "x_min": None,
                "ks": None,
                "p_value": None,
                "tail_fraction": 0.0,
                "support_size": support_size,
                "n_tail": 0,
                "diagnostics": {"reason": "no_valid_xmin_candidate"},
            }

        # Approximate p-value estimation via bootstrap (optional)
        p_val = None
        if cfg["bootstrap"]:
            try:
                # Generate synthetic samples from estimated distribution and calculate KS for each
                ks_boot = []
                n_tail = best["n_tail"]
                alpha = best["alpha"]
                x_min = best["x_min"]

                # Generate samples from continuous power-law: x = x_min * U^{1/(1-α)}, where U ∈ (0,1)
                for _ in range(cfg["bootstrap_runs"]):
                    U = rng.uniform(0.0, 1.0, size=n_tail)
                    x_sim = x_min * (U ** (1.0 / (1.0 - alpha)))

                    # Round to integer values since frequencies are naturally integers
                    x_sim = np.clip(np.round(x_sim).astype(int), x_min, None)

                    # KS for synthetic sample
                    sup_sim, ecdf_sim = empirical_cdf(x_sim)
                    mcdf_sim = model_cdf_continuous(sup_sim, alpha, x_min)
                    if mcdf_sim is None:
                        continue
                    ks_sim = float(np.max(np.abs(ecdf_sim - mcdf_sim)))
                    ks_boot.append(ks_sim)

                if ks_boot:
                    ks_boot = np.array(ks_boot, dtype=float)
                    # p-value ~ proportion of samples with KS greater than observed KS
                    p_val = float(np.mean(ks_boot >= best["ks"]))
            except Exception as e:
                p_val = None

        tail_fraction = best["n_tail"] / float(total_n)
        alpha_bounds_ok = (
            best["alpha"] >= cfg["alpha_bounds"][0]
            and best["alpha"] <= cfg["alpha_bounds"][1]
        )
        ks_quality = (
            "good"
            if best["ks"] <= cfg["ks_threshold_good"]
            else ("borderline" if best["ks"] <= cfg["ks_threshold_strict"] else "poor")
        )

        # Final decision: consider power-law if:
        # - Tail is sufficient (size and proportion),
        # - α is within logical range,
        # - KS quality not too poor,
        # - If p-value available, not too low (e.g., < 0.05)
        is_pl = (
            tail_fraction >= cfg["min_tail_fraction"]
            and best["n_tail"] >= cfg["min_n_tail"]
            and alpha_bounds_ok
            and (ks_quality != "poor")
            and (p_val is None or p_val >= 0.05)
        )

        return {
            "is_power_law": bool(is_pl),
            "alpha": float(best["alpha"]),
            "x_min": int(best["x_min"]),
            "ks": float(best["ks"]),
            "p_value": None if p_val is None else float(p_val),
            "tail_fraction": float(tail_fraction),
            "support_size": support_size,
            "n_tail": int(best["n_tail"]),
            "diagnostics": {
                "alpha_bounds_ok": alpha_bounds_ok,
                "ks_quality": ks_quality,
                "total_n": int(total_n),
            },
        }

    def _determine_deep_type(self, metadata):
        # Non-numeric columns
        if not metadata["is_numeric"]:
            if metadata.get("is_ordinal", False):
                return "ORDINAL"
            elif metadata["unique_count"] < 10:
                return "CATEGORICAL_LOW_CARDINALITY"
            else:
                # Test power law for high cardinality categories
                pl_result = self._test_power_law(metadata)
                metadata["power_law_test"] = pl_result  # Save diagnosis in metadata
                if pl_result.get("is_power_law"):
                    return "CATEGORICAL_POWER_LAW"
                return "CATEGORICAL_HIGH_CARDINALITY"

        # Numeric data (Discrete)
        if metadata.get("is_integer", False):
            if metadata["unique_count"] < 20:
                return "DISCRETE_UNIFORM"
            else:
                # Test power law for integers
                pl_result = self._test_power_law(metadata)
                metadata["power_law_test"] = pl_result
                if pl_result.get("is_power_law"):
                    return "DISCRETE_POWER_LAW"
                return "DISCRETE_UNIFORM"

        # Continuous data
        if metadata.get("is_multimodal", False):
            return "CONTINUOUS_MULTIMODAL"

        if (
            metadata.get("is_gaussian", False)
            or metadata["ks_p_value"] > self.significance_level
        ):
            return "CONTINUOUS_GAUSSIAN"

        if metadata["tail_behavior"] == "HEAVY_TAILED":
            # If tail is very heavy, test power law as well
            pl_result = self._test_power_law(metadata)
            metadata["power_law_test"] = pl_result
            if pl_result.get("is_power_law"):
                return "CONTINUOUS_POWER_LAW_HEAVY_TAIL"
            return "CONTINUOUS_HEAVY_TAILED"

        # Test cyclicity
        if self._test_cyclicity(metadata):
            return "CYCLIC"

        # Default type
        return "CONTINUOUS_MULTIMODAL"

    def _determine_optimal_transformation(self, metadata):
        """Determine optimal transformation for each type"""

        transformations = {
            "CONTINUOUS_GAUSSIAN": {"type": "identity", "parameters": {}},
            "CONTINUOUS_MULTIMODAL": {
                "type": "quantile_gaussian",
                "parameters": {"n_quantiles": 1000, "output_distribution": "normal"},
            },
            "CONTINUOUS_HEAVY_TAILED": {
                "type": "box_cox",
                "parameters": {"method": "box-cox"},
            },
            "DISCRETE_UNIFORM": {"type": "ordinal_encoding", "parameters": {}},
            "DISCRETE_POWER_LAW": {
                "type": "log_transformation",
                "parameters": {"offset": 1},
            },
            "CATEGORICAL_LOW_CARDINALITY": {"type": "one_hot", "parameters": {}},
            "CATEGORICAL_HIGH_CARDINALITY": {
                "type": "target_encoding",
                "parameters": {"smoothing": 10},
            },
            "ORDINAL": {
                "type": "ordinal_encoding",
                "parameters": {"mapping_strategy": "frequency"},
            },
            "CYCLIC": {
                "type": "sincos_encoding",
                "parameters": {"period": metadata.get("period", 24)},
            },
        }

        return transformations.get(
            metadata["deep_type"], transformations["CONTINUOUS_MULTIMODAL"]
        )


class AdvancedDataTransformer:
    """
    Advanced transformation system preserving statistical relationships
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.fitted_transformers = {}
        self.inverse_transformers = {}

    def fit_transform(self, df):
        """Fit and transform all columns"""

        transformed_data = {}
        original_dtypes = {}

        for column in df.columns:
            print(f"Analyzing and transforming column: {column}")

            # Analyze column
            metadata = self.analyzer.analyze_column(df[column], column)

            # Save original data type
            original_dtypes[column] = df[column].dtype

            # Apply appropriate transformation
            transformed_column, transformer = self._apply_transformation(
                df[column], metadata
            )

            transformed_data[column] = transformed_column
            self.fitted_transformers[column] = {
                "transformer": transformer,
                "metadata": metadata,
                "original_dtype": original_dtypes[column],
            }

        return pd.DataFrame(transformed_data), original_dtypes

    def _apply_transformation(self, column_data, metadata):
        """Apply transformation based on deep type"""

        transformation = metadata["optimal_transformation"]
        clean_data = column_data.dropna()

        if transformation["type"] == "quantile_gaussian":
            transformer = QuantileTransformer(
                n_quantiles=min(1000, len(clean_data)),
                output_distribution="normal",
                random_state=42,
            )

            if len(clean_data) > 0:
                transformer.fit(clean_data.values.reshape(-1, 1))
                transformed = column_data.copy()
                transformed.loc[clean_data.index] = transformer.transform(
                    clean_data.values.reshape(-1, 1)
                ).flatten()
                return transformed, transformer

        elif transformation["type"] == "box_cox":
            from scipy import stats

            # Find best lambda for Box-Cox transformation
            transformed_data, lambda_opt = stats.boxcox(clean_data + 1)
            transformer = {"type": "box_cox", "lambda": lambda_opt}

            transformed = column_data.copy()
            transformed.loc[clean_data.index] = transformed_data
            return transformed, transformer

        elif transformation["type"] == "sincos_encoding":
            # Transform cyclic values to sin and cos
            period = transformation["parameters"]["period"]

            sin_component = np.sin(2 * np.pi * column_data / period)
            cos_component = np.cos(2 * np.pi * column_data / period)

            # Return both components
            transformer = {"type": "cyclic", "period": period}
            return (sin_component, cos_component), transformer

        # Other transformations...

        # Default transformation (no change)
        return column_data, None

    def inverse_transform(self, df_transformed, column_info):
        """Inverse transform to restore data to original form"""
        df_original = pd.DataFrame(index=df_transformed.index)

        for column in df_transformed.columns:
            if column in self.fitted_transformers:
                transformer_info = self.fitted_transformers[column]
                transformer = transformer_info["transformer"]

                if transformer is None:
                    df_original[column] = df_transformed[column]

                elif hasattr(transformer, "inverse_transform"):
                    transformed_vals = df_transformed[column].values.reshape(-1, 1)
                    original_vals = transformer.inverse_transform(transformed_vals)
                    df_original[column] = original_vals.flatten()

                # Restore original data type
                df_original[column] = df_original[column].astype(
                    transformer_info["original_dtype"]
                )

        return df_original


# ==========================================================
# Canonical Feature Projection Layer (CRITICAL)
# ==========================================================

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class CanonicalFeatureProjector:
    """
    Transforms advanced analytical deep_type to Canonical Types
    understood by all generative models without any failure
    """

    # =============================================
    # Deep → Canonical Projection Map
    # =============================================
    CANONICAL_MAP = {
        # ===== CATEGORICAL =====
        "CATEGORICAL_LOW_CARDINALITY": "categorical",
        "CATEGORICAL_HIGH_CARDINALITY": "categorical",
        "CATEGORICAL_POWER_LAW": "categorical",
        "CATEGORICAL": "categorical",
        "ORDINAL": "categorical",
        "NOMINAL": "categorical",
        "BOOLEAN": "categorical",
        "BINARY": "categorical",
        "DISCRETE": "categorical",
        # ===== DISCRETE / NUMERIC =====
        "DISCRETE_UNIFORM": "continuous",
        "DISCRETE_POWER_LAW": "continuous",
        # ===== CONTINUOUS =====
        "CONTINUOUS": "continuous",
        "NUMERIC": "continuous",
        "FLOAT": "continuous",
        "INTEGER": "continuous",
        "REAL": "continuous",
        "CONTINUOUS_GAUSSIAN": "continuous",
        "CONTINUOUS_HEAVY_TAILED": "continuous",
        "CONTINUOUS_POWER_LAW_HEAVY_TAIL": "continuous",
        # ===== MULTIMODAL =====
        "CONTINUOUS_MULTIMODAL": "multimodal",
        # ===== TIME =====
        "CYCLIC": "datetime",
        "DATETIME": "datetime",
        "DATE": "datetime",
        "TIME": "datetime",
        "TIMESTAMP": "datetime",
    }

    SUPPORTED = {"categorical", "continuous", "multimodal", "datetime"}

    @classmethod
    def project(
        cls, feature_metadata: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Safe + intelligent + crash-proof projection
        """

        if not isinstance(feature_metadata, dict):
            raise TypeError("feature_metadata must be a dict")

        projected: Dict[str, Dict[str, Any]] = {}

        for col, meta in feature_metadata.items():
            if not isinstance(meta, dict):
                raise ValueError(f"Metadata for column {col} must be a dict")

            meta = meta.copy()

            # ---------------------------------------------
            # 1) Extract types
            # ---------------------------------------------
            deep_type_raw = meta.get("deep_type")
            base_type_raw = meta.get("type")

            deep_type = (
                str(deep_type_raw).upper().strip()
                if deep_type_raw not in [None, "", "NONE"]
                else None
            )

            base_type = (
                str(base_type_raw).upper().strip()
                if base_type_raw not in [None, "", "NONE"]
                else None
            )

            # ---------------------------------------------
            # 2) Determine Canonical Type
            # ---------------------------------------------
            canonical_type = None

            if deep_type and deep_type in cls.CANONICAL_MAP:
                canonical_type = cls.CANONICAL_MAP[deep_type]

            elif base_type and base_type in cls.CANONICAL_MAP:
                canonical_type = cls.CANONICAL_MAP[base_type]

            else:
                # Safe intelligent fallback
                canonical_type = "continuous"
                logger.warning(
                    f"[CanonicalProjector] Fallback applied for column '{col}' "
                    f"(deep_type={deep_type_raw}, type={base_type_raw})"
                )

            # ---------------------------------------------
            # 3) Verify support
            # ---------------------------------------------
            if canonical_type not in cls.SUPPORTED:
                raise ValueError(
                    f"[CanonicalProjector] Unsupported canonical type '{canonical_type}' "
                    f"for column '{col}'"
                )

            # ---------------------------------------------
            # 4) Finalize type
            # ---------------------------------------------
            meta["type"] = canonical_type
            meta["deep_type"] = canonical_type

            # ---------------------------------------------
            # 5) Special handling per type
            # ---------------------------------------------
            # =====================================================
            # MULTIMODAL FIX — n_modes Injection
            # =====================================================
            if (
                meta.get("type") == "multimodal"
                or meta.get("deep_type") == "CONTINUOUS_MULTIMODAL"
            ):

                # If number of modes not specified
                if (
                    "n_modes" not in meta
                    or not isinstance(meta["n_modes"], int)
                    or meta["n_modes"] < 2
                ):
                    # Intelligent conservative fallback
                    meta["n_modes"] = 2

                # output_dim must equal number of modes
                meta["output_dim"] = meta["n_modes"]

                # Standardize type
                meta["type"] = "multimodal"
            if canonical_type == "categorical":
                # Priority 1: explicit output_dim
                if isinstance(meta.get("output_dim"), int) and meta["output_dim"] > 1:
                    pass

                # Priority 2: classes
                elif (
                    isinstance(meta.get("classes"), (list, tuple))
                    and len(meta["classes"]) > 1
                ):
                    meta["output_dim"] = len(meta["classes"])

                # Priority 3: cardinality (very important for job)
                elif (
                    isinstance(meta.get("cardinality"), int) and meta["cardinality"] > 1
                ):
                    meta["output_dim"] = meta["cardinality"]

                else:
                    # Safe fallback that won't crash
                    meta["output_dim"] = 2

                # Final protection
                if meta["output_dim"] <= 1:
                    raise ValueError(
                        f"[CanonicalProjector] Invalid output_dim={meta['output_dim']} "
                        f"for categorical column '{col}'"
                    )
            elif canonical_type == "datetime":
                meta.setdefault("output_dim", 1)

            # ---------------------------------------------
            # 6) Save
            # ---------------------------------------------
            projected[col] = meta

        logger.info(
            f"[CanonicalProjector] Successfully projected {len(projected)} features"
        )

        return projected
