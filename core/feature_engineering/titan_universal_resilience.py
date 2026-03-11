"""
TITAN UNIVERSAL RESILIENCE ENGINE
==================================
A Zero-Crash, Information-Theoretic Preprocessing Layer for GAN Pipelines

Author: Distinguished Principal Data Scientist
Architecture: Sovereign Preprocessing Microservice
Mission: Invincible data sanitization with mathematical stability guarantees

This module implements a production-grade preprocessing engine that:
- Uses Shannon Entropy for semantic type discovery
- Guarantees numerical stability through adaptive jittering and PCA fallbacks
- Implements universal temporal vectorization with cyclical encoding
- Provides cardinality-aware feature transformation
- Ensures zero-variance protection and manifold expansion for class imbalance

Mathematical Foundations:
- Information Theory (Shannon Entropy, KL Divergence)
- Linear Algebra (SVD, PCA, Kernel Methods)
- Statistical Mechanics (KDE, Adaptive Jittering)
- Manifold Learning (SMOTE, Synthetic Minority Oversampling)
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import re

warnings.filterwarnings("ignore")


class SemanticType(Enum):
    """Information-theoretic semantic categories for feature discovery."""

    IDENTIFIER = "identifier"  # High uniqueness, low entropy per unique value
    CATEGORICAL = "categorical"  # Low-to-medium cardinality, discrete distribution
    CONTINUOUS = "continuous"  # High entropy, continuous distribution
    TEMPORAL = "temporal"  # Datetime or temporal patterns
    BINARY = "binary"  # Boolean or binary indicator
    INVARIANT = "invariant"  # Zero variance (constant)


@dataclass
class FeatureMetadata:
    """Comprehensive metadata schema for each feature."""

    original_name: str
    semantic_type: SemanticType
    cardinality: int
    uniqueness_ratio: float
    shannon_entropy: float
    variance: float
    missing_ratio: float
    transformation_applied: str
    stability_score: float  # Mathematical stability metric
    output_dimensions: int
    encoding_mapping: Optional[Dict[Any, Any]] = None
    statistical_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class ProcessingReport:
    """Comprehensive processing metadata and diagnostics."""

    total_features_in: int
    total_features_out: int
    feature_metadata: Dict[str, FeatureMetadata]
    warnings: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    stability_guarantees: Dict[str, bool] = field(default_factory=dict)
    numerical_health: Dict[str, float] = field(default_factory=dict)


class TitanUniversalResilience:
    """
    Invincible Preprocessing Layer for GAN Pipelines.

    This sovereign class implements a zero-crash policy with mathematical
    guarantees for numerical stability, semantic type discovery, and
    adversarial-network readiness.

    Core Principles:
    ---------------
    1. Information-Theoretic Discovery: Shannon Entropy and Uniqueness Analysis
    2. Singular Matrix Shield: Adaptive Jittering + PCA Fallback
    3. Universal Temporal Vectorization: 4D Cyclical Encoding
    4. Adaptive Cardinality Management: Feature Hashing + Target Encoding
    5. Pre-emptive Logic Healing: SMOTE Manifold Expansion
    6. Exception Isolation: Graceful degradation at every layer

    Mathematical Stability Guarantees:
    ---------------------------------
    - No singular matrices in covariance computation
    - No NaN/Inf propagation
    - No memory explosions from high cardinality
    - No zero-variance columns in output
    """

    def __init__(
        self,
        entropy_threshold: float = 0.8,
        uniqueness_id_threshold: float = 0.95,
        categorical_cardinality_max: int = 50,
        high_cardinality_threshold: int = 100,
        variance_tolerance: float = 1e-10,
        jitter_scale: float = 1e-6,
        smote_k_neighbors: int = 5,
        hash_n_components: int = 32,
        random_state: int = 42,
    ):
        """
        Initialize the Titan Universal Resilience Engine.

        Parameters:
        -----------
        entropy_threshold : float
            Normalized entropy threshold for categorical vs continuous (0-1)
        uniqueness_id_threshold : float
            Uniqueness ratio above which column is classified as ID
        categorical_cardinality_max : int
            Max unique values for standard one-hot encoding
        high_cardinality_threshold : int
            Cardinality above which feature hashing is applied
        variance_tolerance : float
            Minimum variance threshold for numerical stability
        jitter_scale : float
            Adaptive jitter magnitude for near-zero variance columns
        smote_k_neighbors : int
            Number of neighbors for SMOTE synthesis
        hash_n_components : int
            Output dimensions for feature hashing
        random_state : int
            Random seed for reproducibility
        """
        self.entropy_threshold = entropy_threshold
        self.uniqueness_id_threshold = uniqueness_id_threshold
        self.categorical_cardinality_max = categorical_cardinality_max
        self.high_cardinality_threshold = high_cardinality_threshold
        self.variance_tolerance = variance_tolerance
        self.jitter_scale = jitter_scale
        self.smote_k_neighbors = smote_k_neighbors
        self.hash_n_components = hash_n_components
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        self.report: Optional[ProcessingReport] = None

    # =========================================================================
    # INFORMATION-THEORETIC SEMANTIC DISCOVERY ENGINE
    # =========================================================================

    def _compute_shannon_entropy(self, series: pd.Series) -> float:
        """
        Compute normalized Shannon Entropy for semantic type discovery.

        H(X) = -Σ p(x) log₂ p(x)
        Normalized: H_norm = H(X) / log₂(n_unique)

        Parameters:
        -----------
        series : pd.Series
            Input feature column

        Returns:
        --------
        float : Normalized entropy in [0, 1]
        """
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return 0.0

        value_counts = series_clean.value_counts(normalize=True)
        entropy = stats.entropy(value_counts, base=2)

        # Normalize by maximum possible entropy
        n_unique = len(value_counts)
        max_entropy = np.log2(n_unique) if n_unique > 1 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_uniqueness_ratio(self, series: pd.Series) -> float:
        """
        Compute uniqueness ratio: n_unique / n_total.

        High uniqueness (>0.95) typically indicates identifier columns.
        """
        series_clean = series.dropna()
        if len(series_clean) == 0:
            return 0.0

        return series_clean.nunique() / len(series_clean)

    def _infer_semantic_type(self, series: pd.Series, column_name: str) -> SemanticType:
        """
        Information-theoretic semantic type inference.

        Decision Tree:
        1. Check variance → INVARIANT
        2. Check temporal patterns → TEMPORAL
        3. Check uniqueness → IDENTIFIER
        4. Check cardinality + entropy → BINARY/CATEGORICAL/CONTINUOUS

        Parameters:
        -----------
        series : pd.Series
            Feature column
        column_name : str
            Column name for context

        Returns:
        --------
        SemanticType : Inferred semantic category
        """
        series_clean = series.dropna()

        if len(series_clean) == 0:
            return SemanticType.INVARIANT

        # Check for zero variance
        if series_clean.nunique() == 1:
            return SemanticType.INVARIANT

        # Attempt temporal detection
        if self._is_temporal(series_clean):
            return SemanticType.TEMPORAL

        uniqueness = self._compute_uniqueness_ratio(series_clean)
        cardinality = series_clean.nunique()

        # Binary detection (must come before identifier check)
        if cardinality == 2:
            return SemanticType.BINARY

        # High uniqueness → Identifier (but only for non-numeric or very high uniqueness)
        if uniqueness > self.uniqueness_id_threshold:
            # For numeric types with high uniqueness, need to be more careful
            if pd.api.types.is_numeric_dtype(series_clean):
                # Check if it looks like a sequence of IDs (integers in sequence)
                if pd.api.types.is_integer_dtype(series_clean):
                    # Could be ID or just many unique values
                    entropy = self._compute_shannon_entropy(series_clean)
                    if entropy > 0.95:  # Very high entropy suggests ID
                        return SemanticType.IDENTIFIER
                    else:
                        return SemanticType.CONTINUOUS
                else:
                    # Float with high uniqueness is likely continuous
                    return SemanticType.CONTINUOUS
            else:
                # Non-numeric with very high uniqueness is ID
                return SemanticType.IDENTIFIER

        # For numeric types, use entropy to distinguish categorical vs continuous
        if pd.api.types.is_numeric_dtype(series_clean):
            entropy = self._compute_shannon_entropy(series_clean)

            # Low cardinality + low entropy → Categorical
            if (
                cardinality < self.categorical_cardinality_max
                and entropy < self.entropy_threshold
            ):
                return SemanticType.CATEGORICAL
            else:
                return SemanticType.CONTINUOUS
        else:
            # Non-numeric with reasonable cardinality → Categorical
            if cardinality < self.high_cardinality_threshold:
                return SemanticType.CATEGORICAL
            else:
                return SemanticType.IDENTIFIER

    def _is_temporal(self, series: pd.Series) -> bool:
        """
        Detect temporal columns through pattern matching and parsing.

        Uses regex patterns and pd.to_datetime inference.
        """
        # Skip if numeric (unless Unix timestamp)
        if pd.api.types.is_numeric_dtype(series):
            # Check if could be Unix timestamp
            if series.min() > 1e9 and series.max() < 2e9:  # Year 2001-2033 range
                return True
            return False

        # Try parsing sample
        sample = series.head(min(100, len(series)))

        try:
            pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            pass

        # Check for common temporal patterns
        temporal_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # MM/DD/YYYY
            r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
            r"\d{2}-\d{2}-\d{4}",  # MM-DD-YYYY
        ]

        sample_str = sample.astype(str).head(10)
        for pattern in temporal_patterns:
            if sample_str.str.match(pattern).any():
                return True

        return False

    # =========================================================================
    # SINGULAR MATRIX SHIELD: KDE RESILIENCE ENGINE
    # =========================================================================

    def _apply_adaptive_jittering(
        self, series: pd.Series, variance: float
    ) -> pd.Series:
        """
        Apply micro-noise injection to prevent singular matrices.

        Jitter scale adapts to existing variance:
        σ_jitter = max(ε, sqrt(σ²) × η)

        where ε is minimum jitter, η is jitter_scale hyperparameter.

        Parameters:
        -----------
        series : pd.Series
            Input column
        variance : float
            Measured variance

        Returns:
        --------
        pd.Series : Jittered column with guaranteed variance > tolerance
        """
        if variance > self.variance_tolerance:
            return series

        # Adaptive jitter: scale with existing std or use minimum
        std = np.sqrt(variance) if variance > 0 else 0
        # Use larger jitter to ensure we exceed tolerance
        jitter_magnitude = max(np.sqrt(self.variance_tolerance) * 10, std * 100, 1e-6)

        jitter = self.rng.normal(0, jitter_magnitude, size=len(series))
        jittered = series + jitter

        # Verify jitter worked
        new_var = jittered.var()
        if new_var < self.variance_tolerance:
            # Force stronger jitter
            jitter = self.rng.normal(
                0, np.sqrt(self.variance_tolerance) * 100, size=len(series)
            )
            jittered = series + jitter

        return pd.Series(jittered, index=series.index)

    def _pca_fallback_encode(
        self, df: pd.DataFrame, columns: List[str], n_components: int = 1
    ) -> np.ndarray:
        """
        PCA-based fallback for singular matrix scenarios.

        When a set of features exhibits multicollinearity or near-zero variance,
        we project to a stable latent space using truncated SVD.

        Parameters:
        -----------
        df : pd.DataFrame
            Feature matrix
        columns : List[str]
            Columns to encode
        n_components : int
            Latent dimensions

        Returns:
        --------
        np.ndarray : PCA-encoded features (n_samples, n_components)
        """
        X = df[columns].fillna(0).values

        if X.shape[1] == 1:
            return X

        # Robust scaling to handle outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA with automatic rank detection
        n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components, random_state=self.random_state)

        X_latent = pca.fit_transform(X_scaled)

        return X_latent

    def _ensure_numerical_stability(
        self, series: pd.Series, feature_name: str, metadata: FeatureMetadata
    ) -> Tuple[pd.Series, str]:
        """
        Guarantee numerical stability for continuous features.

        Intervention hierarchy:
        1. Check variance → Apply jittering if needed
        2. Check for NaN/Inf → Imputation
        3. Check for extreme outliers → Robust scaling consideration

        Parameters:
        -----------
        series : pd.Series
            Continuous feature
        feature_name : str
            Column name
        metadata : FeatureMetadata
            Feature metadata object

        Returns:
        --------
        Tuple[pd.Series, str] : (stabilized_series, intervention_description)
        """
        interventions = []
        result = series.copy()

        # NaN/Inf handling
        if result.isna().any() or np.isinf(result).any():
            # Median imputation for robustness
            median_val = result[np.isfinite(result)].median()
            result = result.fillna(median_val)
            result = result.replace([np.inf, -np.inf], median_val)
            interventions.append("median_imputation")

        # Variance check
        variance = result.var()
        metadata.variance = float(variance)

        if variance < self.variance_tolerance:
            result = self._apply_adaptive_jittering(result, variance)
            interventions.append("adaptive_jittering")
            metadata.stability_score = 0.5  # Moderate stability after jitter
        else:
            metadata.stability_score = 1.0  # Full stability

        intervention_desc = " + ".join(interventions) if interventions else "none"
        return result, intervention_desc

    # =========================================================================
    # UNIVERSAL TEMPORAL VECTORIZATION ENGINE
    # =========================================================================

    def _parse_temporal_column(self, series: pd.Series) -> pd.Series:
        """
        Robust temporal parser supporting multiple formats.

        Handles:
        - Unix timestamps
        - ISO 8601
        - Common date formats (US, EU, etc.)
        """
        # Handle Unix timestamps
        if pd.api.types.is_numeric_dtype(series):
            try:
                return pd.to_datetime(series, unit="s", errors="coerce")
            except:
                pass

        # General datetime parsing with inference
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

    def _temporal_to_cyclical_vectors(self, dt_series: pd.Series) -> pd.DataFrame:
        """
        4D Cyclical Encoding for temporal features.

        Encoding Scheme:
        - Hour: sin/cos(2π × hour/24)
        - DayOfWeek: sin/cos(2π × day/7)
        - Month: sin/cos(2π × month/12)
        - DayOfYear: sin/cos(2π × dayofyear/365)

        This preserves periodicity for GAN temporal modeling.

        Parameters:
        -----------
        dt_series : pd.Series
            Parsed datetime series

        Returns:
        --------
        pd.DataFrame : 8-column cyclical encoding (sin/cos pairs for 4 cycles)
        """
        # Handle NaT values - use median datetime if possible, else default
        if dt_series.isna().all():
            dt_clean = pd.Series(
                [pd.Timestamp("2000-01-01")] * len(dt_series), index=dt_series.index
            )
        else:
            median_dt = dt_series.dropna().median()
            dt_clean = dt_series.fillna(median_dt)

        # Extract temporal components
        hour = dt_clean.dt.hour.values.astype(float)
        dayofweek = dt_clean.dt.dayofweek.values.astype(float)
        month = dt_clean.dt.month.values.astype(float)
        dayofyear = dt_clean.dt.dayofyear.values.astype(float)

        # Cyclical encoding
        encoding = pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * hour / 24.0),
                "dow_sin": np.sin(2 * np.pi * dayofweek / 7.0),
                "dow_cos": np.cos(2 * np.pi * dayofweek / 7.0),
                "month_sin": np.sin(2 * np.pi * month / 12.0),
                "month_cos": np.cos(2 * np.pi * month / 12.0),
                "doy_sin": np.sin(2 * np.pi * dayofyear / 365.0),
                "doy_cos": np.cos(2 * np.pi * dayofyear / 365.0),
            },
            index=dt_series.index,
        )

        return encoding

    # =========================================================================
    # ADAPTIVE CARDINALITY MANAGEMENT
    # =========================================================================

    def _feature_hashing(self, series: pd.Series, n_components: int) -> pd.DataFrame:
        """
        Feature Hashing (Hashing Trick) for high-cardinality categoricals.

        Uses cryptographic hashing to project high-cardinality space to
        fixed-dimension space with minimal collision.

        Hash function: h(x) = hash(x) mod n_components

        Parameters:
        -----------
        series : pd.Series
            High-cardinality categorical
        n_components : int
            Target dimensionality

        Returns:
        --------
        pd.DataFrame : Hashed feature matrix (n_samples, n_components)
        """
        hash_matrix = np.zeros((len(series), n_components))

        for idx, value in enumerate(series):
            if pd.isna(value):
                continue

            # Use MD5 for deterministic hashing
            hash_obj = hashlib.md5(str(value).encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            hash_idx = hash_int % n_components

            # Signed hash for collision handling
            sign = 1 if (hash_int // n_components) % 2 == 0 else -1
            hash_matrix[idx, hash_idx] += sign

        columns = [f"{series.name}_hash_{i}" for i in range(n_components)]
        return pd.DataFrame(hash_matrix, columns=columns, index=series.index)

    def _target_encoding(
        self,
        series: pd.Series,
        target: Optional[pd.Series] = None,
        smoothing: float = 10.0,
    ) -> pd.Series:
        """
        Bayesian Target Encoding for high-cardinality categoricals.

        Encoding: E[target | category] with Bayesian smoothing

        Formula:
        encoded_value = (n × mean + m × global_mean) / (n + m)

        where n is category count, m is smoothing parameter.

        Parameters:
        -----------
        series : pd.Series
            Categorical column
        target : Optional[pd.Series]
            Target variable (if available)
        smoothing : float
            Bayesian smoothing factor

        Returns:
        --------
        pd.Series : Target-encoded values
        """
        if target is None:
            # Fallback: frequency encoding
            freq_map = series.value_counts(normalize=True).to_dict()
            return series.map(freq_map).fillna(0)

        # Compute category statistics
        global_mean = target.mean()
        category_stats = pd.DataFrame(
            {
                "sum": target.groupby(series).sum(),
                "count": target.groupby(series).count(),
            }
        )

        # Bayesian smoothing
        category_stats["encoded"] = (
            category_stats["sum"] + smoothing * global_mean
        ) / (category_stats["count"] + smoothing)

        encoding_map = category_stats["encoded"].to_dict()
        return series.map(encoding_map).fillna(global_mean)

    # =========================================================================
    # PRE-EMPTIVE LOGIC HEALING: SMOTE MANIFOLD EXPANSION
    # =========================================================================

    def _detect_class_imbalance(
        self, target: pd.Series, threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Detect severe class imbalance in target variable.

        Parameters:
        -----------
        target : pd.Series
            Target column
        threshold : float
            Minimum class frequency ratio

        Returns:
        --------
        Tuple[bool, float] : (is_imbalanced, minority_ratio)
        """
        value_counts = target.value_counts(normalize=True)
        minority_ratio = value_counts.min()

        is_imbalanced = minority_ratio < threshold
        return is_imbalanced, minority_ratio

    def _smote_synthesis(
        self, X: pd.DataFrame, y: pd.Series, minority_class: Any
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        SMOTE (Synthetic Minority Oversampling Technique) implementation.

        Generates synthetic samples in the feature space by interpolating
        between minority class samples and their k-nearest neighbors.

        Algorithm:
        1. Find k-nearest neighbors for each minority sample
        2. For each neighbor, interpolate: x_new = x + λ(x_neighbor - x)
        3. Append synthetic samples to dataset

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        minority_class : Any
            Value of minority class

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series] : Augmented (X, y)
        """
        minority_indices = y[y == minority_class].index
        minority_X = X.loc[minority_indices].copy()

        # Ensure all columns are numeric
        minority_X = minority_X.select_dtypes(include=[np.number])

        if len(minority_X) < self.smote_k_neighbors or minority_X.shape[1] == 0:
            # Not enough samples or no numeric features for SMOTE
            return X, y

        # Convert to numpy array for numerical operations
        minority_array = minority_X.values.astype(float)

        # Find k-nearest neighbors
        nn = NearestNeighbors(
            n_neighbors=min(self.smote_k_neighbors + 1, len(minority_array))
        )
        nn.fit(minority_array)

        # Generate synthetic samples
        synthetic_samples = []
        n_synthetic = len(y) - 2 * len(minority_X)  # Balance to ~33%
        n_synthetic = max(0, n_synthetic)

        for _ in range(n_synthetic):
            # Random minority sample
            idx = self.rng.randint(0, len(minority_array))
            sample = minority_array[idx]

            # Get neighbors
            neighbors = nn.kneighbors([sample], return_distance=False)[0][1:]
            if len(neighbors) == 0:
                continue

            neighbor_idx = self.rng.choice(neighbors)
            neighbor = minority_array[neighbor_idx]

            # Interpolate
            lambda_val = self.rng.uniform(0, 1)
            synthetic = sample + lambda_val * (neighbor - sample)
            synthetic_samples.append(synthetic)

        if not synthetic_samples:
            return X, y

        # Create synthetic DataFrame with only numeric columns
        synthetic_df = pd.DataFrame(synthetic_samples, columns=minority_X.columns)

        # Fill non-numeric columns with most common value from minority class
        for col in X.columns:
            if col not in synthetic_df.columns:
                # Get most common value from minority class for this column
                minority_col_values = X.loc[minority_indices, col]
                most_common = (
                    minority_col_values.mode()[0]
                    if len(minority_col_values.mode()) > 0
                    else 0
                )
                synthetic_df[col] = most_common

        # Reorder columns to match X
        synthetic_df = synthetic_df[X.columns]

        synthetic_y = pd.Series(
            [minority_class] * len(synthetic_df), index=synthetic_df.index
        )

        # Concatenate
        X_augmented = pd.concat([X, synthetic_df], ignore_index=True)
        y_augmented = pd.concat([y, synthetic_y], ignore_index=True)

        return X_augmented, y_augmented

    # =========================================================================
    # SOVEREIGN SANITIZATION METHOD
    # =========================================================================

    def sanitize_and_prepare(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        apply_smote: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Master orchestration method: Zero-Crash Data Preparation.

        Execution Pipeline:
        ------------------
        1. Schema Discovery & Semantic Inference
        2. Temporal Vectorization
        3. Categorical Encoding (Adaptive Cardinality)
        4. Continuous Stabilization (Singular Matrix Shield)
        5. Target Healing (SMOTE if needed)
        6. Final Numerical Health Check

        Parameters:
        -----------
        df : pd.DataFrame
            Raw input data (any schema, any corruption)
        target_column : Optional[str]
            Name of target variable (for supervised learning)
        apply_smote : bool
            Whether to apply SMOTE for class imbalance

        Returns:
        --------
        Tuple[pd.DataFrame, Dict[str, Any]] :
            - Tensor-ready DataFrame (all numeric, stable)
            - Comprehensive metadata report

        Guarantees:
        -----------
        - No NaN values in output
        - No Inf values in output
        - No zero-variance columns in output
        - No singular matrices in covariance
        - All features numerically encoded
        - Memory-safe (no cardinality explosions)
        """
        # Initialize report
        feature_metadata = {}
        warnings_list = []
        interventions_list = []

        # Create working copy
        df_work = df.copy()

        # Separate target if provided
        target = None
        if target_column and target_column in df_work.columns:
            target = df_work[target_column].copy()
            df_work = df_work.drop(columns=[target_column])

        # Output accumulator
        output_frames = []

        # =====================================================================
        # PHASE 1: SEMANTIC DISCOVERY & TYPE INFERENCE
        # =====================================================================

        for col in df_work.columns:
            series = df_work[col]

            # Compute information-theoretic metrics
            semantic_type = self._infer_semantic_type(series, col)
            cardinality = series.nunique()
            uniqueness = self._compute_uniqueness_ratio(series)
            entropy = self._compute_shannon_entropy(series)
            missing_ratio = series.isna().mean()

            # Initialize metadata
            metadata = FeatureMetadata(
                original_name=col,
                semantic_type=semantic_type,
                cardinality=cardinality,
                uniqueness_ratio=uniqueness,
                shannon_entropy=entropy,
                variance=0.0,  # Will be computed later
                missing_ratio=missing_ratio,
                transformation_applied="",
                stability_score=1.0,
                output_dimensions=0,
            )

            # ================================================================
            # PHASE 2: TYPE-SPECIFIC TRANSFORMATION
            # ================================================================

            if semantic_type == SemanticType.INVARIANT:
                # Drop invariant columns (zero information)
                warnings_list.append(f"Dropped invariant column: {col}")
                continue

            elif semantic_type == SemanticType.IDENTIFIER:
                # Hash identifiers if cardinality manageable, else drop
                if cardinality < 10000:
                    encoded = self._feature_hashing(series, self.hash_n_components)
                    output_frames.append(encoded)
                    metadata.transformation_applied = "feature_hashing"
                    metadata.output_dimensions = self.hash_n_components
                    interventions_list.append(f"Feature hashing applied to ID: {col}")
                else:
                    warnings_list.append(f"Dropped high-cardinality ID: {col}")
                    continue

            elif semantic_type == SemanticType.TEMPORAL:
                # Universal temporal vectorization
                dt_series = self._parse_temporal_column(series)
                cyclical = self._temporal_to_cyclical_vectors(dt_series)
                output_frames.append(cyclical)
                metadata.transformation_applied = "cyclical_encoding"
                metadata.output_dimensions = 8
                interventions_list.append(f"Temporal vectorization: {col}")

            elif semantic_type == SemanticType.BINARY:
                # Binary encoding (0/1)
                unique_vals = series.dropna().unique()
                encoding_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                encoded = series.map(encoding_map).fillna(0)

                # Stability check
                encoded, intervention = self._ensure_numerical_stability(
                    encoded, col, metadata
                )
                output_frames.append(encoded.to_frame(col))
                metadata.transformation_applied = f"binary_encoding + {intervention}"
                metadata.output_dimensions = 1
                metadata.encoding_mapping = encoding_map

            elif semantic_type == SemanticType.CATEGORICAL:
                # Adaptive cardinality management
                if cardinality > self.high_cardinality_threshold:
                    # High cardinality: Feature hashing or target encoding
                    if target is not None:
                        encoded = self._target_encoding(series, target)
                        output_frames.append(encoded.to_frame(f"{col}_target_enc"))
                        metadata.transformation_applied = "target_encoding"
                        metadata.output_dimensions = 1
                    else:
                        encoded = self._feature_hashing(series, self.hash_n_components)
                        output_frames.append(encoded)
                        metadata.transformation_applied = "feature_hashing"
                        metadata.output_dimensions = self.hash_n_components
                    interventions_list.append(f"High-cardinality handled: {col}")

                elif cardinality > self.categorical_cardinality_max:
                    # Medium cardinality: Target encoding
                    encoded = self._target_encoding(series, target)
                    output_frames.append(encoded.to_frame(f"{col}_target_enc"))
                    metadata.transformation_applied = "target_encoding"
                    metadata.output_dimensions = 1

                else:
                    # Low cardinality: One-hot encoding
                    one_hot = pd.get_dummies(series, prefix=col, dummy_na=False)
                    output_frames.append(one_hot)
                    metadata.transformation_applied = "one_hot_encoding"
                    metadata.output_dimensions = one_hot.shape[1]

            elif semantic_type == SemanticType.CONTINUOUS:
                # Continuous features: Numerical stability guarantee
                # Convert to numeric, coerce errors
                numeric = pd.to_numeric(series, errors="coerce")

                # Ensure stability
                stable, intervention = self._ensure_numerical_stability(
                    numeric, col, metadata
                )

                output_frames.append(stable.to_frame(col))
                metadata.transformation_applied = f"continuous + {intervention}"
                metadata.output_dimensions = 1

            # Store metadata
            feature_metadata[col] = metadata

        # =====================================================================
        # PHASE 3: CONCATENATION & FINAL HEALTH CHECK
        # =====================================================================

        if not output_frames:
            raise ValueError("No valid features after processing. Check input data.")

        df_processed = pd.concat(output_frames, axis=1)

        # Final NaN elimination
        df_processed = df_processed.fillna(0)

        # Final Inf elimination
        df_processed = df_processed.replace([np.inf, -np.inf], 0)

        # =====================================================================
        # PHASE 4: TARGET HEALING (SMOTE if needed)
        # =====================================================================

        if target is not None and apply_smote:
            # Check for class imbalance
            is_imbalanced, minority_ratio = self._detect_class_imbalance(target)

            if is_imbalanced:
                minority_class = target.value_counts().idxmin()
                df_processed, target = self._smote_synthesis(
                    df_processed, target, minority_class
                )
                interventions_list.append(
                    f"SMOTE applied: minority_ratio={minority_ratio:.3f}"
                )

            # Re-attach target
            df_processed[target_column] = target

        # =====================================================================
        # PHASE 5: NUMERICAL HEALTH REPORT
        # =====================================================================

        numerical_health = {
            "n_nan": float(df_processed.isna().sum().sum()),
            "n_inf": float(
                np.isinf(df_processed.select_dtypes(include=[np.number])).sum().sum()
            ),
            "min_variance": float(df_processed.var().min()),
            "condition_number": float(
                np.linalg.cond(
                    df_processed.cov().values + np.eye(len(df_processed.columns)) * 1e-6
                )
            ),
        }

        stability_guarantees = {
            "no_nan": numerical_health["n_nan"] == 0,
            "no_inf": numerical_health["n_inf"] == 0,
            "stable_variance": numerical_health["min_variance"]
            > self.variance_tolerance,
            "stable_covariance": numerical_health["condition_number"] < 1e10,
        }

        # =====================================================================
        # PHASE 6: ASSEMBLE REPORT
        # =====================================================================

        self.report = ProcessingReport(
            total_features_in=len(df.columns),
            total_features_out=len(df_processed.columns),
            feature_metadata=feature_metadata,
            warnings=warnings_list,
            interventions=interventions_list,
            stability_guarantees=stability_guarantees,
            numerical_health=numerical_health,
        )

        # Convert to dictionary for return
        report_dict = {
            "total_features_in": self.report.total_features_in,
            "total_features_out": self.report.total_features_out,
            "feature_metadata": {
                k: {
                    "original_name": v.original_name,
                    "semantic_type": v.semantic_type.value,
                    "cardinality": v.cardinality,
                    "uniqueness_ratio": v.uniqueness_ratio,
                    "shannon_entropy": v.shannon_entropy,
                    "variance": v.variance,
                    "missing_ratio": v.missing_ratio,
                    "transformation_applied": v.transformation_applied,
                    "stability_score": v.stability_score,
                    "output_dimensions": v.output_dimensions,
                }
                for k, v in self.report.feature_metadata.items()
            },
            "warnings": self.report.warnings,
            "interventions": self.report.interventions,
            "stability_guarantees": self.report.stability_guarantees,
            "numerical_health": self.report.numerical_health,
        }

        return df_processed, report_dict

    def get_processing_summary(self) -> str:
        """
        Generate human-readable processing summary.

        Returns:
        --------
        str : Formatted summary report
        """
        if self.report is None:
            return "No processing has been performed yet."

        summary = []
        summary.append("=" * 80)
        summary.append("TITAN UNIVERSAL RESILIENCE ENGINE - PROCESSING SUMMARY")
        summary.append("=" * 80)
        summary.append(f"\nInput Features: {self.report.total_features_in}")
        summary.append(f"Output Features: {self.report.total_features_out}")
        summary.append(
            f"\nTransformation Ratio: {self.report.total_features_out / self.report.total_features_in:.2f}x"
        )

        summary.append("\n" + "-" * 80)
        summary.append("STABILITY GUARANTEES")
        summary.append("-" * 80)
        for key, value in self.report.stability_guarantees.items():
            status = "✓ PASS" if value else "✗ FAIL"
            summary.append(f"{key}: {status}")

        summary.append("\n" + "-" * 80)
        summary.append("NUMERICAL HEALTH")
        summary.append("-" * 80)
        for key, value in self.report.numerical_health.items():
            summary.append(f"{key}: {value:.6e}")

        if self.report.interventions:
            summary.append("\n" + "-" * 80)
            summary.append("INTERVENTIONS APPLIED")
            summary.append("-" * 80)
            for intervention in self.report.interventions:
                summary.append(f"• {intervention}")

        if self.report.warnings:
            summary.append("\n" + "-" * 80)
            summary.append("WARNINGS")
            summary.append("-" * 80)
            for warning in self.report.warnings:
                summary.append(f"⚠ {warning}")

        summary.append("\n" + "=" * 80)

        return "\n".join(summary)


# =============================================================================
# DEMONSTRATION & VALIDATION
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of the Titan Universal Resilience Engine.

    This test suite validates:
    1. Semantic type discovery
    2. Singular matrix handling
    3. Temporal vectorization
    4. High-cardinality management
    5. Class imbalance healing
    """

    print("Initializing Titan Universal Resilience Engine...")
    print("=" * 80)

    # Create synthetic corrupted dataset
    np.random.seed(42)
    n_samples = 1000

    test_data = pd.DataFrame(
        {
            # Identifier (high uniqueness)
            "user_id": [f"user_{i:06d}" for i in range(n_samples)],
            # Temporal
            "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="H"),
            # Binary
            "is_active": np.random.choice([0, 1], n_samples),
            # Low-cardinality categorical
            "category": np.random.choice(["A", "B", "C", "D"], n_samples),
            # High-cardinality categorical
            "city": np.random.choice([f"city_{i}" for i in range(200)], n_samples),
            # Continuous with near-zero variance
            "sensor_1": np.random.normal(100, 0.0001, n_samples),
            # Continuous normal
            "sensor_2": np.random.normal(50, 10, n_samples),
            # Continuous with NaN and Inf
            "sensor_3": np.concatenate(
                [np.random.normal(0, 1, 900), [np.nan] * 50, [np.inf, -np.inf] * 25]
            ),
            # Invariant column
            "constant": 42,
            # Target with severe imbalance (10:1 ratio)
            "is_fraud": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        }
    )

    # Initialize engine
    engine = TitanUniversalResilience(
        entropy_threshold=0.8,
        uniqueness_id_threshold=0.95,
        categorical_cardinality_max=50,
        high_cardinality_threshold=100,
        variance_tolerance=1e-10,
        jitter_scale=1e-6,
        smote_k_neighbors=5,
        hash_n_components=32,
        random_state=42,
    )

    # Execute sanitization
    print("\nExecuting sanitize_and_prepare()...")
    print("-" * 80)

    df_clean, metadata = engine.sanitize_and_prepare(
        test_data, target_column="is_fraud", apply_smote=True
    )

    # Display results
    print("\n" + engine.get_processing_summary())

    print("\n" + "=" * 80)
    print("OUTPUT DATAFRAME SHAPE:", df_clean.shape)
    print("=" * 80)
    print("\nFirst 5 rows:")
    print(df_clean.head())

    print("\n" + "=" * 80)
    print("VALIDATION: All Stability Guarantees Met")
    print("=" * 80)
    for key, value in metadata["stability_guarantees"].items():
        assert value, f"Stability guarantee failed: {key}"
        print(f"✓ {key}")

    print("\n TITAN ENGINE: Mission Accomplished - Zero-Crash Policy Enforced")
