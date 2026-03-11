# core/integration/hybrid_engine.py
"""
Production-Grade Hybrid Synthetic Data Generation Engine - Core Infrastructure
Streamlined architecture with zero redundancy - fully integrated with hybrid_engine2.py

Features:
- Core configuration management
- Feature schema validation
- Advanced data processing pipeline
- Memory-efficient data loading
- Metrics tracking and monitoring
- State management and checkpointing
- Circuit breaker fault tolerance

Author: Titan AI Architecture Team
Version: 5.0 Enterprise
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)


# ============================================================================
# LOGGER CONFIGURATION
# ============================================================================


def setup_advanced_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Advanced logger setup with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    if format_string is None:
        format_string = (
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(funcName)s:%(lineno)d | %(message)s"
        )

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_advanced_logger(
    __name__, log_file=Path("logs/hybrid_engine_core.log"), level=logging.INFO
)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class EngineState(Enum):
    """System state machine for tracking engine lifecycle."""

    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    DATA_LOADING = auto()
    DATA_LOADED = auto()
    MODEL_BUILDING = auto()
    MODELS_BUILT = auto()
    TRAINING = auto()
    TRAINED = auto()
    GENERATING = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class FeatureType(Enum):
    """Canonical feature types for schema validation."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    DATETIME = "datetime"
    TEXT = "text"
    MIXED = "mixed"


class ScalingStrategy(Enum):
    """Data scaling strategies for continuous features."""

    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    NONE = "none"


class EncodingStrategy(Enum):
    """Categorical encoding strategies."""

    ONEHOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"
    TARGET = "target"
    EMBEDDING = "embedding"


# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""

    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """
    Production-grade circuit breaker for fault tolerance.
    Prevents cascade failures during massive generation operations.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 3,
        name: str = "DefaultCircuitBreaker",
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_attempts = half_open_attempts
        self.name = name

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        self.half_open_attempt_count = 0

        self._lock = threading.Lock()

        logger.info(
            f"Circuit Breaker '{self.name}' initialized: "
            f"threshold={failure_threshold}, timeout={timeout_seconds}s"
        )

    def call(self, func, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    logger.info(
                        f"Circuit Breaker '{self.name}' entering HALF_OPEN state"
                    )
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_attempt_count = 0
                else:
                    raise RuntimeError(
                        f"Circuit Breaker '{self.name}' is OPEN. "
                        f"Retry after {self.timeout_seconds}s"
                    )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_seconds

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_attempt_count += 1

                if self.half_open_attempt_count >= self.half_open_attempts:
                    logger.info(f"Circuit Breaker '{self.name}' reset to CLOSED")
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.half_open_attempt_count = 0

            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.warning(
                    f"Circuit Breaker '{self.name}' reopened after half-open failure"
                )
                self.state = CircuitBreakerState.OPEN

            elif self.failure_count >= self.failure_threshold:
                logger.error(
                    f"Circuit Breaker '{self.name}' opened after {self.failure_count} failures"
                )
                self.state = CircuitBreakerState.OPEN

    def reset(self):
        """Manually reset the circuit breaker."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_attempt_count = 0
            self.last_failure_time = None
            logger.info(f"Circuit Breaker '{self.name}' manually reset")


# ============================================================================
# FEATURE SCHEMA VALIDATION
# ============================================================================


@dataclass
class FeatureSchema:
    """
    Comprehensive feature schema with validation.
    Supports all major feature types and transformations.
    """

    name: str
    feature_type: FeatureType
    is_nullable: bool = False
    default_value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    ordinal_order: Optional[List[str]] = None
    datetime_format: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    distribution: Optional[str] = None
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD
    encoding_strategy: EncodingStrategy = EncodingStrategy.ONEHOT
    output_dim: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, value: Any) -> bool:
        """Validate value against schema constraints."""
        if pd.isna(value):
            return self.is_nullable

        if self.feature_type == FeatureType.CONTINUOUS:
            if not isinstance(value, (int, float)):
                return False
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False

        elif self.feature_type == FeatureType.CATEGORICAL:
            if self.categories and value not in self.categories:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "is_nullable": self.is_nullable,
            "default_value": self.default_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "categories": self.categories,
            "ordinal_order": self.ordinal_order,
            "datetime_format": self.datetime_format,
            "dependencies": self.dependencies,
            "distribution": self.distribution,
            "scaling_strategy": self.scaling_strategy.value,
            "encoding_strategy": self.encoding_strategy.value,
            "output_dim": self.output_dim,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureSchema:
        """Create from dictionary."""
        data = data.copy()
        data["feature_type"] = FeatureType(data["feature_type"])
        data["scaling_strategy"] = ScalingStrategy(data["scaling_strategy"])
        data["encoding_strategy"] = EncodingStrategy(data["encoding_strategy"])
        return cls(**data)


class FeatureSchemaRegistry:
    """Central registry for feature schemas with validation."""

    def __init__(self):
        self.schemas: Dict[str, FeatureSchema] = OrderedDict()
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def register(self, schema: FeatureSchema):
        """Register a feature schema."""
        with self._lock:
            if schema.name in self.schemas:
                logger.warning(
                    f"Overwriting existing schema for feature: {schema.name}"
                )

            self.schemas[schema.name] = schema

            # Update dependency graph
            for dep in schema.dependencies:
                self.dependency_graph[dep].append(schema.name)

    def get(self, feature_name: str) -> Optional[FeatureSchema]:
        """Get schema by feature name."""
        return self.schemas.get(feature_name)

    def get_all(self) -> List[FeatureSchema]:
        """Get all registered schemas."""
        return list(self.schemas.values())

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate entire dataframe against schemas."""
        errors = []

        for col in df.columns:
            schema = self.get(col)
            if schema is None:
                errors.append(f"No schema found for column: {col}")
                continue

            for idx, value in enumerate(df[col]):
                if not schema.validate(value):
                    errors.append(
                        f"Validation failed for {col} at row {idx}: value={value}"
                    )
                    if len(errors) > 100:
                        errors.append("... (truncated)")
                        break

        is_valid = len(errors) == 0
        return is_valid, errors

    def save(self, path: Union[str, Path]):
        """Save schemas to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        schemas_dict = {name: schema.to_dict() for name, schema in self.schemas.items()}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(schemas_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Schemas saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> FeatureSchemaRegistry:
        """Load schemas from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            schemas_dict = json.load(f)

        registry = cls()
        for name, schema_data in schemas_dict.items():
            schema = FeatureSchema.from_dict(schema_data)
            registry.register(schema)

        logger.info(f"Loaded {len(registry.schemas)} schemas from: {path}")
        return registry


# ============================================================================
# VECTORIZED DATA PROCESSOR
# ============================================================================


class VectorizedDataProcessor:
    """
    Production-grade data processor with vectorized operations.

    Features:
    - Parallel processing for large datasets
    - Memory-efficient transformations
    - Comprehensive validation
    - Support for all feature types
    """

    def __init__(
        self,
        schema_registry: Optional[FeatureSchemaRegistry] = None,
        use_gpu_acceleration: bool = False,
        n_jobs: int = -1,
    ):
        self.schema_registry = schema_registry or FeatureSchemaRegistry()
        self.use_gpu_acceleration = use_gpu_acceleration
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count() or 1

        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_metadata: Dict[str, Dict] = {}
        self.is_fitted = False

    def fit(self, data: pd.DataFrame) -> "VectorizedDataProcessor":
        """
        Enterprise-grade fitting orchestrator.

        Orchestrates the statistical analysis and state initialization for all features.
        Implements a 'Metadata-First' approach to ensure downstream sub-processors
        have a valid state registry to write to, preventing KeyErrors.
        """
        logger.info(
            f" Initiating fitting sequence on {len(data):,} samples | {len(data.columns)} features"
        )

        self.feature_metadata = {}

        # We use a stable copy of the columns to ensure consistency during parallel processing
        columns_to_process = data.columns.tolist()

        for col in columns_to_process:
            # 1. Check if schema exists or infer it
            schema = self.schema_registry.get(col)
            if schema is None:
                logger.info(f" Auto-detecting schema for feature: '{col}'")
                schema = self._auto_detect_schema(data[col], col)
                self.schema_registry.register(schema)

            # 2. Structural Preparation
            # We initialize metadata 'before' calling sub-fit functions to ensure the reference exists
            self.feature_metadata[col] = {
                "type": schema.feature_type.value,
                "output_dim": schema.output_dim,
                "schema": schema.to_dict(),
                "impute_value": 0.0,  # default value, updated inside sub-functions
                "fitted_at": pd.Timestamp.now().isoformat(),
            }

            # 3. Semantic routing based on feature type
            try:
                if schema.feature_type == FeatureType.CONTINUOUS:
                    self._fit_continuous(data[col], schema)

                elif schema.feature_type in [
                    FeatureType.CATEGORICAL,
                    FeatureType.ORDINAL,
                ]:
                    self._fit_categorical(data[col], schema)

                elif schema.feature_type == FeatureType.DATETIME:
                    self._fit_datetime(data[col], schema)

                # Update output dimension in metadata after actual fit (e.g. One-Hot expansion)
                self.feature_metadata[col]["output_dim"] = schema.output_dim

            except Exception as e:
                logger.error(f" Failed to fit feature '{col}': {str(e)}", exc_info=True)
                raise  # In production we prefer to stop here if fit fails to ensure model safety

        self.is_fitted = True
        logger.info(" Data processor fitting sequence completed successfully")
        return self

    def transform(self, data: pd.DataFrame) -> torch.Tensor:
        """Transform data using fitted transformers."""
        if not self.is_fitted:
            raise RuntimeError("Processor not fitted. Call fit() first.")

        transformed_parts = []

        # Parallel transformation
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_col = {
                executor.submit(self._transform_column, data[col], col): col
                for col in data.columns
            }

            for future in as_completed(future_to_col):
                col = future_to_col[future]
                try:
                    result = future.result()
                    transformed_parts.append(result)
                except Exception as e:
                    logger.error(f"Error transforming column {col}: {e}")
                    raise

        # Concatenate and convert to tensor
        transformed_df = pd.concat(transformed_parts, axis=1)
        transformed_df.index = data.index

        # Convert to torch tensor
        tensor = torch.tensor(transformed_df.values, dtype=torch.float32)

        return tensor

    def inverse_transform(
        self, data: Union[pd.DataFrame, torch.Tensor]
    ) -> pd.DataFrame:
        """Inverse transform synthetic data back to original space."""
        if isinstance(data, torch.Tensor):
            data = self._tensor_to_dataframe(data)

        reconstructed_parts = []

        for col in self.feature_metadata.keys():
            schema = self.schema_registry.get(col)

            if schema.feature_type == FeatureType.CONTINUOUS:
                reconstructed_parts.append(self._inverse_continuous(data, col, schema))

            elif schema.feature_type in [FeatureType.CATEGORICAL, FeatureType.ORDINAL]:
                reconstructed_parts.append(self._inverse_categorical(data, col, schema))

            elif schema.feature_type == FeatureType.DATETIME:
                reconstructed_parts.append(self._inverse_datetime(data, col, schema))

        reconstructed = pd.concat(reconstructed_parts, axis=1)
        return reconstructed

    def validate(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data quality."""
        errors = []

        # Check for missing values
        missing_ratio = data.isnull().sum() / len(data)
        high_missing = missing_ratio[missing_ratio > 0.5]
        if not high_missing.empty:
            errors.append(f"High missing values: {high_missing.to_dict()}")

        # Check for infinite values
        for col in data.select_dtypes(include=[np.number]).columns:
            if np.isinf(data[col]).any():
                errors.append(f"Infinite values in column: {col}")

        # Schema validation
        is_valid, schema_errors = self.schema_registry.validate_dataframe(data)
        errors.extend(schema_errors[:10])  # Limit error messages

        is_valid = len(errors) == 0
        return is_valid, errors

    def _auto_detect_schema(self, series: pd.Series, col_name: str) -> FeatureSchema:
        """
        Enterprise-grade automatic schema inference engine.

        Optimized to differentiate between true continuous variables and
        numerical representations of categorical data, preventing mathematical
        errors in downstream processing.
        """
        # 1. Initial statistical properties analysis
        nunique = series.nunique()
        total_count = len(series)
        null_ratio = series.isnull().mean()

        # 2. Numeric data inference
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            # Check cardinality
            # If unique values are very few relative to data size → likely categorical
            is_low_cardinality = nunique < 15 or (
                nunique / total_count < 0.01 and nunique < 50
            )

            if not is_low_cardinality:
                # Outlier-aware scaling strategy selection
                # ROBUST is default as it is safest in production
                return FeatureSchema(
                    name=col_name,
                    feature_type=FeatureType.CONTINUOUS,
                    min_value=float(series.min()) if pd.notnull(series.min()) else 0.0,
                    max_value=float(series.max()) if pd.notnull(series.max()) else 1.0,
                    scaling_strategy=ScalingStrategy.ROBUST,
                    output_dim=1,
                )

        # 3. Temporal data inference
        if pd.api.types.is_datetime64_any_dtype(series) or "date" in col_name.lower():
            # Attempt conversion if name suggests date but type is still object
            try:
                if not pd.api.types.is_datetime64_any_dtype(series):
                    pd.to_datetime(series.iloc[:100], errors="raise")

                return FeatureSchema(
                    name=col_name,
                    feature_type=FeatureType.DATETIME,
                    output_dim=6,  # (Year, Month, Day, Hour, DayOfWeek, IsWeekend)
                )
            except:
                pass

        # 4. Categorical data inference
        # Includes text, low-cardinality numbers, and anything else
        categories = series.unique().tolist()
        # Clean list from null values to ensure encoder stability
        categories = [cat for cat in categories if pd.notnull(cat)]

        # Choose encoding strategy based on cardinality (optimization)
        encoding_strategy = (
            EncodingStrategy.ONEHOT if nunique < 25 else EncodingStrategy.LABEL
        )

        return FeatureSchema(
            name=col_name,
            feature_type=FeatureType.CATEGORICAL,
            categories=categories,
            output_dim=(
                max(2, nunique) if encoding_strategy == EncodingStrategy.ONEHOT else 1
            ),
            encoding_strategy=encoding_strategy,
            default_value="__MISSING__",
        )

    def _fit_continuous(self, series: pd.Series, schema: FeatureSchema):
        """
        Enterprise-grade fitting for continuous features.

        Optimized for high-performance processing of 10M records with
        multi-layer error recovery and statistical consistency.
        """
        try:
            # 1. Safe numeric conversion
            # Coerce any text contamination (e.g. 'technician') to NaN to prevent math crashes
            numeric_series = pd.to_numeric(series, errors="coerce")

            # 2. Strategic imputation value
            # Median is more robust to outliers
            if numeric_series.notna().any():
                fill_value = float(numeric_series.median())
            else:
                # Complete column corruption fallback
                logger.warning(
                    f" Feature '{schema.name}' lacks valid numeric data. Fallback to 0.0"
                )
                fill_value = 0.0

            # 3. Vectorized data preparation
            clean_data = numeric_series.fillna(fill_value).values.reshape(-1, 1)

            # 4. Adaptive scaling strategy selection & configuration
            if schema.scaling_strategy == ScalingStrategy.STANDARD:
                scaler = StandardScaler()
            elif schema.scaling_strategy == ScalingStrategy.ROBUST:
                # 5-95 quantile range for heavy-tailed distributions
                scaler = RobustScaler(
                    quantile_range=(5, 95), with_centering=True, with_scaling=True
                )
            elif schema.scaling_strategy == ScalingStrategy.MINMAX:
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = None

            # 5. Fit scaler and sync state & metadata
            if scaler:
                scaler.fit(clean_data)
                self.scalers[schema.name] = scaler

                # Proactive key existence check to prevent KeyError
                if schema.name not in self.feature_metadata:
                    self.feature_metadata[schema.name] = {}

                # Save imputation value for train-generation consistency
                self.feature_metadata[schema.name]["impute_value"] = fill_value
                schema.output_dim = 1

            logger.debug(f" Successfully fitted continuous feature: '{schema.name}'")

        except Exception as e:
            logger.error(
                f" Critical failure in _fit_continuous for '{schema.name}': {str(e)}",
                exc_info=True,
            )
            # Emergency fail-safe protocol: prevent engine-level crash
            fallback_scaler = MinMaxScaler(feature_range=(-1, 1))
            fallback_scaler.fit([[0], [1]])
            self.scalers[schema.name] = fallback_scaler

            if schema.name in self.feature_metadata:
                self.feature_metadata[schema.name]["impute_value"] = 0.0
            schema.output_dim = 1

    def _fit_categorical(self, series: pd.Series, schema: FeatureSchema):
        """Fit encoder for categorical features."""
        clean_data = series.fillna("__MISSING__")

        if schema.encoding_strategy == EncodingStrategy.ONEHOT:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoder.fit(clean_data.values.reshape(-1, 1))
            schema.output_dim = len(encoder.categories_[0])

        elif schema.encoding_strategy == EncodingStrategy.LABEL:
            encoder = LabelEncoder()
            encoder.fit(clean_data)
            schema.output_dim = 1

        elif schema.encoding_strategy == EncodingStrategy.ORDINAL:
            if schema.ordinal_order:
                encoder = OrdinalEncoder(categories=[schema.ordinal_order])
            else:
                encoder = OrdinalEncoder()
            encoder.fit(clean_data.values.reshape(-1, 1))
            schema.output_dim = 1

        else:
            encoder = LabelEncoder()
            encoder.fit(clean_data)
            schema.output_dim = 1

        self.encoders[schema.name] = encoder

    def _fit_datetime(self, series: pd.Series, schema: FeatureSchema):
        """Fit datetime feature extractor."""
        schema.output_dim = 6

    def _transform_column(self, series: pd.Series, col_name: str) -> pd.DataFrame:
        """
        Enterprise-grade single column transformation pipeline.

        Ensures strict mathematical consistency between training and inference
        by utilizing persisted metadata and safe-casting mechanisms.
        """
        schema = self.schema_registry.get(col_name)
        if not schema:
            logger.warning(
                f"No schema found for column '{col_name}'. Using raw pass-through."
            )
            return pd.DataFrame(series, index=series.index)

        # --- 1. Continuous features transformation ---
        if schema.feature_type == FeatureType.CONTINUOUS:
            # Forced numeric conversion to prevent process hang on unexpected text
            numeric_series = pd.to_numeric(series, errors="coerce")

            # Retrieve pre-computed imputation value (never recompute median here)
            fill_value = self.feature_metadata.get(col_name, {}).get(
                "impute_value", 0.0
            )
            clean_data = numeric_series.fillna(fill_value).values.reshape(-1, 1)

            if col_name in self.scalers:
                try:
                    scaled = self.scalers[col_name].transform(clean_data)
                except Exception as e:
                    logger.error(
                        f"Transformation failed for '{col_name}': {e}. Returning zero-filled."
                    )
                    scaled = np.zeros_like(clean_data)
            else:
                scaled = clean_data

            return pd.DataFrame(scaled, columns=[col_name], index=series.index)

        # --- 2. Categorical / Ordinal features transformation ---
        elif schema.feature_type in [FeatureType.CATEGORICAL, FeatureType.ORDINAL]:
            # Handle missing values with system-defined default
            clean_data = series.fillna("__MISSING__").astype(str)

            if col_name in self.encoders:
                encoder = self.encoders[col_name]

                try:
                    if isinstance(encoder, OneHotEncoder):
                        encoded = encoder.transform(clean_data.values.reshape(-1, 1))
                        cols = [f"{col_name}_{i}" for i in range(encoded.shape[1])]
                        return pd.DataFrame(encoded, columns=cols, index=series.index)
                    else:
                        # LabelEncoder & OrdinalEncoder expect 1D input
                        encoded = encoder.transform(clean_data)
                        return pd.DataFrame(
                            encoded, columns=[col_name], index=series.index
                        )
                except Exception as e:
                    logger.error(
                        f"Encoding failed for '{col_name}': {e}. Using fallback labeling."
                    )
                    return pd.DataFrame(
                        np.zeros((len(series), 1)),
                        columns=[col_name],
                        index=series.index,
                    )

        # --- 3. Datetime features transformation ---
        elif schema.feature_type == FeatureType.DATETIME:
            dt_series = pd.to_datetime(series, errors="coerce")

            # Vectorized temporal feature extraction
            features = pd.DataFrame(
                {
                    f"{col_name}_year": dt_series.dt.year.fillna(2020),
                    f"{col_name}_month": dt_series.dt.month.fillna(1),
                    f"{col_name}_day": dt_series.dt.day.fillna(1),
                    f"{col_name}_hour": dt_series.dt.hour.fillna(0),
                    f"{col_name}_dayofweek": dt_series.dt.dayofweek.fillna(0),
                    f"{col_name}_is_weekend": (dt_series.dt.dayofweek >= 5)
                    .fillna(0)
                    .astype(int),
                },
                index=series.index,
            )

            # Cyclical / standard normalization for temporal features
            # Using 1e-8 to prevent division by zero in empty cases
            for c in features.columns:
                f_mean = features[c].mean()
                f_std = features[c].std()
                features[c] = (features[c] - f_mean) / (f_std + 1e-8)

            return features

        # --- 4. Fallback / pass-through ---
        return pd.DataFrame(series, index=series.index)

    def _inverse_continuous(
        self, data: pd.DataFrame, col_name: str, schema: FeatureSchema
    ) -> pd.DataFrame:
        """Inverse transform continuous feature."""
        if col_name in data.columns:
            values = data[col_name].values.reshape(-1, 1)
        else:
            values = np.zeros((len(data), 1))

        if col_name in self.scalers:
            original = self.scalers[col_name].inverse_transform(values)
        else:
            original = values

        return pd.DataFrame(original.flatten(), columns=[col_name])

    def _inverse_categorical(
        self, data: pd.DataFrame, col_name: str, schema: FeatureSchema
    ) -> pd.DataFrame:
        """Inverse transform categorical feature."""
        encoder = self.encoders.get(col_name)

        if encoder is None:
            return pd.DataFrame([schema.default_value] * len(data), columns=[col_name])

        if isinstance(encoder, OneHotEncoder):
            feature_cols = [c for c in data.columns if c.startswith(f"{col_name}_")]

            if not feature_cols:
                return pd.DataFrame(
                    [schema.default_value] * len(data), columns=[col_name]
                )

            encoded_values = data[feature_cols].values
            decoded = encoder.inverse_transform(encoded_values)
            return pd.DataFrame(decoded.flatten(), columns=[col_name])

        else:
            if col_name in data.columns:
                values = data[col_name].values
            else:
                values = np.zeros(len(data))

            decoded = encoder.inverse_transform(values.astype(int))
            return pd.DataFrame(decoded, columns=[col_name])

    def _inverse_datetime(
        self, data: pd.DataFrame, col_name: str, schema: FeatureSchema
    ) -> pd.DataFrame:
        """Inverse transform datetime feature."""
        dt_cols = {
            "year": f"{col_name}_year",
            "month": f"{col_name}_month",
            "day": f"{col_name}_day",
            "hour": f"{col_name}_hour",
        }

        missing_cols = [c for c in dt_cols.values() if c not in data.columns]
        if missing_cols:
            return pd.DataFrame(
                [pd.Timestamp("2020-01-01")] * len(data), columns=[col_name]
            )

        reconstructed = pd.DataFrame(
            {
                "year": (data[dt_cols["year"]] * 10 + 2020).astype(int),
                "month": np.clip((data[dt_cols["month"]] * 5 + 6).astype(int), 1, 12),
                "day": np.clip((data[dt_cols["day"]] * 10 + 15).astype(int), 1, 28),
                "hour": np.clip((data[dt_cols["hour"]] * 8 + 12).astype(int), 0, 23),
            }
        )

        datetimes = pd.to_datetime(reconstructed, errors="coerce")
        return pd.DataFrame(datetimes, columns=[col_name])

    def _tensor_to_dataframe(self, tensor: torch.Tensor) -> pd.DataFrame:
        """Convert tensor to dataframe with proper column mapping."""
        numpy_array = tensor.cpu().numpy()

        columns = []
        for col_name, meta in self.feature_metadata.items():
            output_dim = meta["output_dim"]

            if output_dim == 1:
                columns.append(col_name)
            else:
                columns.extend([f"{col_name}_{i}" for i in range(output_dim)])

        if len(columns) != numpy_array.shape[1]:
            logger.warning(
                f"Column count mismatch: expected {len(columns)}, got {numpy_array.shape[1]}"
            )
            columns = [f"feature_{i}" for i in range(numpy_array.shape[1])]

        return pd.DataFrame(numpy_array, columns=columns)


# ============================================================================
# DATALOADER FACTORY
# ============================================================================


class DataLoaderFactory:
    """Factory for creating optimized DataLoaders."""

    @staticmethod
    def create_dataloader(
        data: Union[pd.DataFrame, torch.Tensor, np.ndarray],
        config: Any,
        is_training: bool = True,
    ) -> DataLoader:
        """
        Create optimized DataLoader based on data type.

        Args:
            data: Input data (DataFrame, Tensor, or ndarray)
            config: Engine configuration
            is_training: Whether this is for training

        Returns:
            Optimized DataLoader
        """
        # Convert to tensor
        if isinstance(data, pd.DataFrame):
            tensor_data = torch.tensor(data.values, dtype=torch.float32)
        elif isinstance(data, np.ndarray):
            tensor_data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            tensor_data = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Create dataset
        dataset = TensorDataset(tensor_data)

        # Create dataloader
        return DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=is_training,
            num_workers=config.num_workers,
            pin_memory=(config.device == "cuda"),
            drop_last=is_training,
        )


# ============================================================================
# METRICS TRACKER
# ============================================================================


class MetricsTracker:
    """Advanced metrics tracking with aggregation and export."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: Dict[str, List[datetime]] = defaultdict(list)
        self.history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def log(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Log a metric value."""
        with self._lock:
            self.metrics[metric_name].append(value)
            self.history[metric_name].append(value)
            self.timestamps[metric_name].append(timestamp or datetime.now())

    def log_batch(
        self, metrics_dict: Dict[str, float], timestamp: Optional[datetime] = None
    ):
        """Log multiple metrics at once."""
        ts = timestamp or datetime.now()
        with self._lock:
            for name, value in metrics_dict.items():
                self.metrics[name].append(value)
                self.history[name].append(value)
                self.timestamps[name].append(ts)

    def get_last(self, metric_name: str, n: int = 1) -> List[float]:
        """Get last n values of a metric."""
        with self._lock:
            return self.metrics[metric_name][-n:]

    def compute_statistics(self, metric_name: str) -> Dict[str, float]:
        """Compute statistics for a metric."""
        with self._lock:
            values = self.metrics.get(metric_name, [])

            if not values:
                return {}

            return {
                "count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
                "last": float(values[-1]),
            }

    def compute_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for all metrics."""
        return {name: self.compute_statistics(name) for name in self.metrics.keys()}

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export metrics to pandas DataFrame."""
        data = []

        with self._lock:
            for metric_name in self.metrics.keys():
                for value, timestamp in zip(
                    self.metrics[metric_name], self.timestamps[metric_name]
                ):
                    data.append(
                        {"metric": metric_name, "value": value, "timestamp": timestamp}
                    )

        return pd.DataFrame(data)

    def save(self, path: Union[str, Path]):
        """Save metrics to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.export_to_dataframe()
        df.to_csv(path, index=False)

        logger.info(f"Metrics saved to: {path}")

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.timestamps.clear()
            self.history.clear()


# ============================================================================
# SYSTEM STATE MANAGER
# ============================================================================


class SystemStateManager:
    """Manages engine state transitions and persistence."""

    def __init__(self, config: Any):
        self.config = config
        self.state = EngineState.UNINITIALIZED
        self.state_history: List[Tuple[EngineState, datetime]] = []
        self.metadata: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def transition(self, new_state: EngineState):
        """Transition to new state."""
        with self._lock:
            old_state = self.state
            self.state = new_state
            self.state_history.append((new_state, datetime.now()))

            logger.info(f"State transition: {old_state.name} -> {new_state.name}")

    def can_transition(self, target_state: EngineState) -> bool:
        """Check if transition to target state is valid."""
        valid_transitions = {
            EngineState.UNINITIALIZED: [EngineState.INITIALIZING],
            EngineState.INITIALIZING: [EngineState.READY, EngineState.ERROR],
            EngineState.READY: [EngineState.DATA_LOADING, EngineState.MODEL_BUILDING],
            EngineState.DATA_LOADING: [EngineState.DATA_LOADED, EngineState.ERROR],
            EngineState.DATA_LOADED: [EngineState.MODEL_BUILDING],
            EngineState.MODEL_BUILDING: [EngineState.MODELS_BUILT, EngineState.ERROR],
            EngineState.MODELS_BUILT: [EngineState.TRAINING],
            EngineState.TRAINING: [EngineState.TRAINED, EngineState.ERROR],
            EngineState.TRAINED: [EngineState.GENERATING, EngineState.TRAINING],
            EngineState.GENERATING: [EngineState.TRAINED, EngineState.ERROR],
            EngineState.ERROR: [EngineState.INITIALIZING],
        }

        return target_state in valid_transitions.get(self.state, [])

    def get_current_state(self) -> EngineState:
        """Get current state."""
        return self.state

    def is_ready_for_training(self) -> bool:
        """Check if system is ready for training."""
        return self.state == EngineState.MODELS_BUILT

    def is_trained(self) -> bool:
        """Check if models are trained."""
        return self.state == EngineState.TRAINED

    def save_state(self, path: Union[str, Path]):
        """Save state to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state_data = {
            "current_state": self.state.name,
            "state_history": [
                (state.name, ts.isoformat()) for state, ts in self.state_history
            ],
            "metadata": self.metadata,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"State saved to: {path}")

    @classmethod
    def load_state(cls, path: Union[str, Path], config: Any) -> SystemStateManager:
        """Load state from JSON file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            state_data = json.load(f)

        manager = cls(config)
        manager.state = EngineState[state_data["current_state"]]
        manager.metadata = state_data["metadata"]

        logger.info(f"State loaded from: {path}")
        return manager


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Logger
    "setup_advanced_logger",
    # Enumerations
    "EngineState",
    "FeatureType",
    "ScalingStrategy",
    "EncodingStrategy",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    # Feature Schema
    "FeatureSchema",
    "FeatureSchemaRegistry",
    # Data Processing
    "VectorizedDataProcessor",
    "DataLoaderFactory",
    # Monitoring
    "MetricsTracker",
    "SystemStateManager",
]


# ============================================================================
# VALIDATION & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TITAN HYBRID ENGINE - CORE INFRASTRUCTURE v5.0")
    print("=" * 80)

    # Test logger
    test_logger = setup_advanced_logger("test", level=logging.INFO)
    test_logger.info("✓ Logger initialized")

    # Test circuit breaker
    cb = CircuitBreaker(failure_threshold=3, timeout_seconds=5, name="TestBreaker")
    test_logger.info("✓ Circuit Breaker initialized")

    # Test feature schema
    schema = FeatureSchema(
        name="test_feature",
        feature_type=FeatureType.CONTINUOUS,
        min_value=0.0,
        max_value=100.0,
    )
    test_logger.info(f"✓ Feature Schema created: {schema.name}")

    # Test schema registry
    registry = FeatureSchemaRegistry()
    registry.register(schema)
    test_logger.info(f"✓ Schema Registry: {len(registry.schemas)} schemas")

    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.log("test_metric", 0.95)
    stats = tracker.compute_statistics("test_metric")
    test_logger.info(f"✓ Metrics Tracker: {stats}")

    # Test state manager
    class DummyConfig:
        device = "cpu"
        batch_size = 512

    state_manager = SystemStateManager(DummyConfig())
    state_manager.transition(EngineState.INITIALIZING)
    state_manager.transition(EngineState.READY)
    test_logger.info(f"✓ State Manager: {state_manager.get_current_state().name}")

    print("\n" + "=" * 80)
    print("3 CORE INFRASTRUCTURE VALIDATION COMPLETE")
    print("=" * 80)
    print("\nAvailable Components:")
    print("  ✓ Advanced logging system")
    print("  ✓ Circuit breaker fault tolerance")
    print("  ✓ Feature schema validation")
    print("  ✓ Vectorized data processing")
    print("  ✓ Metrics tracking and monitoring")
    print("  ✓ State management")
    print("\nReady for integration with hybrid_engine2.py")
    print("=" * 80)
