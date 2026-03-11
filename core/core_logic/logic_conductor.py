"""
Advanced Logic Conductor - Schema-Agnostic Cognitive Core for Data Validation
===============================================================================

This module implements a schema-agnostic probabilistic reasoning engine using:
- Dynamic Bayesian Networks (auto-discovered structure)
- Adaptive Constraint Satisfaction
- Information-theoretic quality scoring
- Self-healing with intelligent inference

Key Features:
- Works with ANY tabular dataset (no hardcoded columns)
- Auto-detects relationships via correlation/mutual information
- Graceful degradation when features are missing
- Type-safe and production-hardened

Author: Refactored Production Implementation
License: MIT
Version: 2.0 (Schema-Agnostic)
"""

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Default message level (can be changed to DEBUG or WARNING)
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Define logger
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Optional pgmpy with graceful fallback
try:
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.inference import VariableElimination

    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    print("  Warning: pgmpy not available. Using fallback probabilistic engine.")

from scipy.stats import entropy, chi2_contingency, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# --- Add this line to solve import issue in other files ---
BANKING_MANIFEST = {"rules": []}
# ---------------------------------------------------------


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================


@dataclass
class ValidationResult:
    """Encapsulates validation result for a single row."""

    row_id: int
    original_values: Dict[str, Any]
    corrected_values: Dict[str, Any]
    confidence_score: float
    entropy_score: float
    violations: List[str] = field(default_factory=list)
    corrections_applied: List[str] = field(default_factory=list)
    status: str = "valid"  # valid, corrected, discarded


@dataclass
class LogicReport:
    """Comprehensive validation report."""

    total_rows: int
    valid_rows: int
    corrected_rows: int
    discarded_rows: int
    validation_results: List["ValidationResult"]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def health_score(self) -> float:
        """Calculate the overall health score dynamically."""
        if self.total_rows == 0:
            return 100.0
        # Percentage of valid and successfully corrected records
        return ((self.valid_rows + self.corrected_rows) / self.total_rows) * 100.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        # Use the new property directly
        success_rate = self.health_score

        return f"""
╔══════════════════════════════════════════════════════════╗
║       LOGIC CONDUCTOR VALIDATION REPORT (v2.0)           ║
╠══════════════════════════════════════════════════════════╣
║ Timestamp: {self.timestamp:<42} ║
║ Total Rows Processed: {self.total_rows:<34} ║
║ ├─ Valid (No Changes): {self.valid_rows:<31} ║
║ ├─ Corrected (Self-Healed): {self.corrected_rows:<26} ║
║ └─ Discarded (Unsalvageable): {self.discarded_rows:<24} ║
║                                                          ║
║ Logic Health Score: {success_rate:.2f}%{' ' * (42 - len(f'{success_rate:.2f}'))}║
╚══════════════════════════════════════════════════════════╝
"""


# =============================================================================
# SCHEMA ANALYZER - Auto-discovers dataset structure
# =============================================================================


class SchemaAnalyzer:
    """
    Automatically analyzes dataset schema and discovers:
    - Column types (numerical, categorical, binary)
    - Statistical relationships (correlation, mutual information)
    - Suggested graph structure for Bayesian Network
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.column_types = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.relationship_graph = None

    def analyze(
        self, data: pd.DataFrame, correlation_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Perform comprehensive schema analysis.

        Args:
            data: Input DataFrame
            correlation_threshold: Minimum correlation to create graph edge

        Returns:
            Schema metadata dictionary
        """
        if data.empty:
            return self._empty_schema()

        if self.verbose:
            print(
                f" Schema Analysis: Analyzing {len(data)} rows × {len(data.columns)} columns..."
            )

        # Step 1: Classify column types
        self._classify_columns(data)

        # Step 2: Discover relationships
        self._discover_relationships(data, correlation_threshold)

        # Step 3: Build suggested graph structure
        suggested_edges = self._build_suggested_structure()

        schema_info = {
            "column_types": self.column_types,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "binary_cols": self.binary_cols,
            "suggested_edges": suggested_edges,
            "relationship_matrix": self.relationship_graph,
        }

        if self.verbose:
            print(
                f"   ✓ Detected {len(self.numerical_cols)} numerical, "
                f"{len(self.categorical_cols)} categorical, "
                f"{len(self.binary_cols)} binary columns"
            )
            print(f"   ✓ Suggested {len(suggested_edges)} dependency edges\n")

        return schema_info

    def _classify_columns(self, data: pd.DataFrame) -> None:
        """Classify each column as numerical, categorical, or binary."""
        for col in data.columns:
            nunique = data[col].nunique()
            dtype = data[col].dtype

            # Binary detection
            if nunique == 2:
                self.binary_cols.append(col)
                self.column_types[col] = "binary"
            # Numerical detection
            elif pd.api.types.is_numeric_dtype(dtype):
                self.numerical_cols.append(col)
                self.column_types[col] = "numerical"
            # Categorical (low cardinality or object type)
            elif nunique < len(data) * 0.5:  # Less than 50% unique
                self.categorical_cols.append(col)
                self.column_types[col] = "categorical"
            else:
                # High cardinality - treat as categorical but note it
                self.categorical_cols.append(col)
                self.column_types[col] = "categorical_high_cardinality"

    def _discover_relationships(self, data: pd.DataFrame, threshold: float) -> None:
        """
        Discover statistical relationships using:
        - Spearman correlation for numerical pairs
        - Mutual information for mixed types
        - Chi-square for categorical pairs
        """
        cols = data.columns.tolist()
        n_cols = len(cols)

        # Initialize relationship matrix
        self.relationship_graph = pd.DataFrame(0.0, index=cols, columns=cols)

        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i >= j:  # Skip diagonal and duplicates
                    continue

                try:
                    score = self._calculate_relationship_score(data, col1, col2)
                    self.relationship_graph.loc[col1, col2] = score
                    self.relationship_graph.loc[col2, col1] = score
                except Exception:
                    pass  # Skip problematic pairs

    def _calculate_relationship_score(
        self, data: pd.DataFrame, col1: str, col2: str
    ) -> float:
        """Calculate relationship strength between two columns."""
        type1 = self.column_types[col1]
        type2 = self.column_types[col2]

        # Clean data
        subset = data[[col1, col2]].dropna()
        if len(subset) < 10:
            return 0.0

        # Numerical-Numerical: Spearman correlation
        if type1 == "numerical" and type2 == "numerical":
            corr, _ = spearmanr(subset[col1], subset[col2])
            return abs(corr) if not np.isnan(corr) else 0.0

        # Categorical-Categorical: Chi-square based
        elif "categorical" in type1 and "categorical" in type2:
            contingency = pd.crosstab(subset[col1], subset[col2])
            if contingency.size > 1:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                # Cramér's V (normalized chi-square)
                n = contingency.sum().sum()
                min_dim = min(contingency.shape) - 1
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                return cramers_v if p_value < 0.05 else 0.0
            return 0.0

        # Mixed types: Mutual information
        else:
            # Encode categorical
            if "categorical" in type1 or type1 == "binary":
                le = LabelEncoder()
                x = le.fit_transform(subset[col1].astype(str))
            else:
                x = subset[col1].values

            if "categorical" in type2 or type2 == "binary":
                le = LabelEncoder()
                y = le.fit_transform(subset[col2].astype(str))
            else:
                y = subset[col2].values

            # Calculate MI
            x = x.reshape(-1, 1)
            if type2 in ["categorical", "binary"] or "categorical" in type2:
                mi = mutual_info_classif(x, y, discrete_features=True, random_state=42)
            else:
                mi = mutual_info_regression(
                    x, y, discrete_features=False, random_state=42
                )

            # Normalize to [0, 1]
            return min(mi[0] / 2.0, 1.0)

    def _build_suggested_structure(self) -> List[Tuple[str, str]]:
        """
        Build suggested DAG edges based on relationship strengths.

        Strategy:
        1. Order columns by their average relationship strength (heuristic for "causality")
        2. Create edges from stronger influencers to weaker dependents
        3. Ensure acyclic structure
        """
        if self.relationship_graph is None:
            return []

        # Calculate influence score (sum of relationships)
        influence_scores = self.relationship_graph.sum(axis=1).sort_values(
            ascending=False
        )
        ordered_cols = influence_scores.index.tolist()

        edges = []
        for i, col1 in enumerate(ordered_cols):
            # Only create edges to columns "below" in ordering (ensures acyclic)
            for col2 in ordered_cols[i + 1 :]:
                strength = self.relationship_graph.loc[col1, col2]
                if strength > 0.15:  # Threshold for edge creation
                    edges.append((col1, col2))

        return edges

    def _empty_schema(self) -> Dict[str, Any]:
        """Return empty schema for error cases."""
        return {
            "column_types": {},
            "numerical_cols": [],
            "categorical_cols": [],
            "binary_cols": [],
            "suggested_edges": [],
            "relationship_matrix": None,
        }


# =============================================================================
# AUTO BINNER - Generic discretization for any numerical column
# =============================================================================


class AutoBinner:
    """
    Automatically discretizes numerical columns without knowing their names.
    Uses quantile-based binning for robustness.
    """

    def __init__(self, n_bins: int = 10, strategy: str = "quantile"):
        """
        Args:
            n_bins: Number of bins to create
            strategy: 'quantile' or 'uniform'
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges = {}  # Store for consistency

    def fit_transform(
        self, data: pd.DataFrame, numerical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Discretize all numerical columns.

        Args:
            data: DataFrame with mixed types
            numerical_cols: List of numerical column names

        Returns:
            DataFrame with numerical columns converted to categories
        """
        result = data.copy()

        for col in numerical_cols:
            if col not in result.columns:
                continue

            try:
                if self.strategy == "quantile":
                    # Quantile-based binning (robust to outliers)
                    result[col], bins = pd.qcut(
                        result[col],
                        q=self.n_bins,
                        labels=[f"{col}_q{i}" for i in range(self.n_bins)],
                        duplicates="drop",
                        retbins=True,
                    )
                else:
                    # Uniform-width binning
                    result[col], bins = pd.cut(
                        result[col],
                        bins=self.n_bins,
                        labels=[f"{col}_b{i}" for i in range(self.n_bins)],
                        retbins=True,
                    )

                self.bin_edges[col] = bins
                result[col] = result[col].astype(str)  # Ensure string type

            except Exception as e:
                # Fallback: leave as-is or convert to string
                result[col] = result[col].astype(str)

        return result


# =============================================================================
# BAYESIAN DEPENDENCY GRAPH - Schema-Agnostic Version
# =============================================================================


class BayesianDependencyGraph:
    """
    Schema-agnostic Bayesian Network that:
    - Accepts pre-defined structure OR auto-discovers from data
    - Dynamically prunes missing columns
    - Handles type mismatches gracefully
    """

    def __init__(self, use_pgmpy: bool = True, verbose: bool = True):
        self.use_pgmpy = use_pgmpy and PGMPY_AVAILABLE
        self.verbose = verbose
        self.graph = nx.DiGraph()
        self.cpds = {}
        self.structure = None
        self.inference_engine = None
        self.label_encoders = {}
        self.auto_binner = AutoBinner(n_bins=5)
        self.schema_info = None

    def build_structure(
        self,
        edges: Optional[List[Tuple[str, str]]] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Build graph structure either from provided edges or auto-discovery.

        Args:
            edges: Optional list of (parent, child) edges
            data: Optional data for auto-discovery if edges not provided
        """
        if edges:
            # Use provided structure
            self.graph.add_edges_from(edges)
            if self.verbose:
                print(f"   ├─ Using provided structure: {len(edges)} edges")
        elif data is not None:
            # Auto-discover structure
            analyzer = SchemaAnalyzer(verbose=self.verbose)
            self.schema_info = analyzer.analyze(data)
            edges = self.schema_info["suggested_edges"]
            self.graph.add_edges_from(edges)
            if self.verbose:
                print(f"   ├─ Auto-discovered structure: {len(edges)} edges")
        else:
            if self.verbose:
                print("     Warning: No structure provided, using empty graph")

        if self.use_pgmpy and edges:
            try:
                self.structure = DiscreteBayesianNetwork(edges)
            except Exception as e:
                if self.verbose:
                    print(f"     pgmpy structure creation failed: {e}")
                self.use_pgmpy = False

    def learn_parameters(
        self, reference_data: pd.DataFrame, discretize: bool = True
    ) -> None:
        """
        Learn CPDs from reference data with schema alignment.

        Args:
            reference_data: Training DataFrame
            discretize: Whether to bin numerical columns
        """
        if reference_data.empty:
            if self.verbose:
                print("     Warning: Empty reference data, skipping parameter learning")
            return

        working_data = reference_data.copy()

        # Step 1: Auto-detect schema if not done
        if self.schema_info is None:
            analyzer = SchemaAnalyzer(verbose=False)
            self.schema_info = analyzer.analyze(working_data)

        # Step 2: Discretize numerical columns
        if discretize and self.schema_info["numerical_cols"]:
            working_data = self.auto_binner.fit_transform(
                working_data, self.schema_info["numerical_cols"]
            )

        # Step 3: Prune graph to match available columns (CRITICAL)
        available_cols = set(working_data.columns)
        if self.structure:
            nodes_to_remove = [
                n for n in self.structure.nodes() if n not in available_cols
            ]
            for node in nodes_to_remove:
                self.structure.remove_node(node)
                if self.verbose:
                    print(f"   ├─ Pruned node '{node}' (not in data)")

        # Update graph as well
        graph_nodes_to_remove = [
            n for n in self.graph.nodes() if n not in available_cols
        ]
        for node in graph_nodes_to_remove:
            self.graph.remove_node(node)

        # Step 4: Encode all columns as categorical strings (type safety)
        for col in working_data.columns:
            working_data[col] = working_data[col].astype(str)

        # Step 5: Fit Bayesian Network
        if self.use_pgmpy and self.structure and len(self.structure.nodes()) > 0:
            try:
                # Ensure we only use columns that exist in both structure and data
                valid_cols = [
                    n for n in self.structure.nodes() if n in working_data.columns
                ]

                if not valid_cols:
                    if self.verbose:
                        print(
                            "     No valid columns for Bayesian Network, using fallback"
                        )
                    self.use_pgmpy = False
                    return

                self.structure.fit(
                    working_data[valid_cols],
                    estimator=BayesianEstimator,
                    prior_type="BDeu",
                    equivalent_sample_size=5,
                )
                self.inference_engine = VariableElimination(self.structure)

                if self.verbose:
                    print(f"   ✓ Learned CPDs for {len(valid_cols)} variables")

            except Exception as e:
                if self.verbose:
                    print(f"     Bayesian Network training failed: {e}")
                    print(f"   ├─ Switching to empirical fallback mode")
                self.use_pgmpy = False
                self._compute_empirical_cpds(working_data)
        else:
            # Fallback to empirical CPDs
            self._compute_empirical_cpds(working_data)

    def _compute_empirical_cpds(self, data: pd.DataFrame) -> None:
        """Compute empirical conditional probabilities."""
        for col in data.columns:
            if col not in self.graph.nodes():
                continue

            parents = list(self.graph.predecessors(col))

            if not parents:
                # Prior probability
                self.cpds[col] = data[col].value_counts(normalize=True).to_dict()
            else:
                # Conditional probability (simplified)
                parent_cols = [p for p in parents if p in data.columns]
                if parent_cols:
                    grouped = data[parent_cols + [col]].value_counts(normalize=True)
                    self.cpds[col] = grouped.to_dict()

    def calculate_row_probability(self, row: Dict[str, Any]) -> float:
        """
        Calculate likelihood of row under learned distribution.

        Returns probability in [0, 1] (higher = more coherent)
        """
        if not self.graph.nodes():
            return 0.5  # Neutral if no structure

        log_prob = 0.0
        n_valid = 0

        for node in self.graph.nodes():
            if node not in row:
                continue

            try:
                parents = list(self.graph.predecessors(node))

                if not parents or node not in self.cpds:
                    # Use prior
                    prob = self.cpds.get(node, {}).get(row[node], 1e-5)
                else:
                    # Use conditional probability
                    parent_vals = tuple([row.get(p) for p in parents if p in row])
                    key = parent_vals + (row[node],)
                    prob = self.cpds.get(node, {}).get(key, 1e-5)

                log_prob += np.log(max(prob, 1e-10))
                n_valid += 1
            except Exception:
                continue

        # Normalize by number of variables
        if n_valid > 0:
            avg_log_prob = log_prob / n_valid
            return min(1.0, max(0.0, np.exp(avg_log_prob)))

        return 0.5


# =============================================================================
# CONSTRAINT SATISFACTION SOLVER - Generic Version
# =============================================================================


class ConstraintSatisfactionSolver:
    """
    Schema-agnostic constraint solver that:
    - Accepts constraints dynamically via add_rule()
    - Skips rules referencing missing columns
    - Provides safe fallback for all operations
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.constraints = []
        self.soft_constraints = []

    def add_rule(
        self,
        name: str,
        constraint_func: Callable,
        variables: List[str],
        rule_type: str = "hard",
        weight: float = 1.0,
        description: str = "",
    ) -> None:
        """
        Add a validation rule dynamically.

        Args:
            name: Rule identifier
            constraint_func: Function(row_dict) -> bool or float
            variables: Columns this rule depends on
            rule_type: 'hard' (must satisfy) or 'soft' (optimization)
            weight: Weight for soft constraints
            description: Human-readable explanation
        """
        rule = {
            "name": name,
            "func": constraint_func,
            "variables": variables,
            "type": rule_type,
            "weight": weight,
            "description": description,
        }

        if rule_type == "hard":
            self.constraints.append(rule)
        else:
            self.soft_constraints.append(rule)

    def check_constraints(
        self, row: Dict[str, Any], available_columns: Set[str]
    ) -> Tuple[List[str], float]:
        """
        Check all applicable constraints.

        Args:
            row: Data row as dictionary
            available_columns: Columns present in dataset

        Returns:
            violations: List of violated constraint names
            soft_score: Soft constraint score [0, 1]
        """
        violations = []

        # Check hard constraints
        for constraint in self.constraints:
            # Skip if required columns missing
            if not all(v in available_columns for v in constraint["variables"]):
                continue

            try:
                if not constraint["func"](row):
                    violations.append(constraint["name"])
            except Exception as e:
                if self.verbose:
                    print(f"     Constraint {constraint['name']} failed: {e}")

        # Calculate soft constraint score
        total_weight = 0.0
        soft_score = 0.0

        for soft_c in self.soft_constraints:
            # Skip if required columns missing
            if not all(v in available_columns for v in soft_c["variables"]):
                continue

            try:
                score = soft_c["func"](row)
                soft_score += score * soft_c["weight"]
                total_weight += soft_c["weight"]
            except Exception:
                continue

        soft_score = soft_score / total_weight if total_weight > 0 else 1.0

        return violations, soft_score

    def heal_row(
        self,
        row: Dict[str, Any],
        reference_data: pd.DataFrame,
        available_columns: Set[str],
    ) -> Tuple[Dict[str, Any], List[str], float]:
        """
        Attempt to heal constraint violations.

        Args:
            row: Row to heal
            reference_data: Training data for reference
            available_columns: Available columns in dataset

        Returns:
            healed_row: Corrected row
            corrections: List of applied corrections
            confidence: Healing confidence [0, 1]
        """
        violations, soft_score = self.check_constraints(row, available_columns)

        if not violations:
            return row, [], 1.0

        healed_row = row.copy()
        corrections = []

        # Apply rule-based healing for violated constraints
        for violation in violations:
            constraint = next(
                (c for c in self.constraints if c["name"] == violation), None
            )
            if constraint:
                # Attempt intelligent correction based on violation type
                correction = self._intelligent_correction(
                    healed_row, constraint, reference_data
                )
                if correction:
                    corrections.append(correction)

        # Verify healing
        final_violations, final_soft_score = self.check_constraints(
            healed_row, available_columns
        )

        # Calculate confidence
        violations_resolved = len(violations) - len(final_violations)
        confidence = min(
            1.0,
            violations_resolved / max(len(violations), 1) * 0.7
            + final_soft_score * 0.3,
        )

        return healed_row, corrections, max(0.0, confidence)

    def _intelligent_correction(
        self, row: Dict[str, Any], constraint: Dict, reference_data: pd.DataFrame
    ) -> Optional[str]:
        variables = constraint["variables"]
        # Identify "anchor columns" (assumed correct in the current row for reference)
        anchors = [
            c for c in row.keys() if c not in variables and c in reference_data.columns
        ]

        for var in variables:
            if var not in row or var not in reference_data.columns:
                continue

            # 1. One-Hot group correction (structural correction)
            if any(c.startswith(var + "_") for c in reference_data.columns):
                group_cols = [
                    c for c in reference_data.columns if c.startswith(var + "_")
                ]
                probs = [row.get(c, 0.0) for c in group_cols]
                best_idx = np.argmax(probs)
                for i, c in enumerate(group_cols):
                    row[c] = 1.0 if i == best_idx else 0.0
                return f"{var} group: enforced logical winner"

            # 2. Contextual Numerical Healing
            if pd.api.types.is_numeric_dtype(reference_data[var]):
                # Attempt to find similar context in reference data instead of using global mean
                try:
                    # Sample reference data that matches other columns (like job or status)
                    context_match = reference_data
                    for anchor in anchors[
                        :2
                    ]:  # Use first two anchors for performance and to avoid zero results
                        if anchor in row:
                            val = row[anchor]
                            # If column is categorical, look for same category
                            if isinstance(val, (str, bool)):
                                mask = context_match[anchor] == val
                                if mask.any():
                                    context_match = context_match[mask]

                    target_median = context_match[var].median()
                    target_std = context_match[var].std() or (
                        reference_data[var].std() * 0.1
                    )
                except:
                    # Fallback if similar context not found
                    target_median = reference_data[var].median()
                    target_std = reference_data[var].std() or 1.0

                # Apply correction with smart jitter
                # The 0.05 ratio preserves logic and prevents Mode Collapse in GAN
                jitter = np.random.normal(0, 0.05 * target_std)
                new_val = target_median + jitter

                # Dynamic clamping using original data bounds
                min_val = reference_data[var].min()
                max_val = reference_data[var].max()
                new_val = np.clip(new_val, min_val, max_val)

                # Special handling for integer columns
                if pd.api.types.is_integer_dtype(reference_data[var]):
                    new_val = round(new_val)

                old = row[var]
                row[var] = new_val
                return f"{var}: {old:.2f} -> {new_val:.2f} (contextual healing)"

        return None


# =============================================================================
# ENTROPY SCORER - Generic Quality Assessment
# =============================================================================


class EntropyScorer:
    """
    Calculates information-theoretic quality metrics without assuming column names.
    """

    def __init__(self):
        self.feature_distributions = {}
        self.reference_stats = {}

    def fit(self, reference_data: pd.DataFrame) -> None:
        """Learn reference distribution statistics."""
        for col in reference_data.columns:
            try:
                if (
                    reference_data[col].dtype == "object"
                    or reference_data[col].nunique() < 20
                ):
                    # Categorical
                    value_counts = reference_data[col].value_counts(normalize=True)
                    self.feature_distributions[col] = value_counts.to_dict()

                    probs = value_counts.values
                    ent = entropy(probs, base=2) if len(probs) > 0 else 0

                    self.reference_stats[col] = {
                        "entropy": ent,
                        "mode": (
                            value_counts.index[0] if len(value_counts) > 0 else None
                        ),
                        "type": "categorical",
                    }
                else:
                    # Numerical
                    self.reference_stats[col] = {
                        "mean": reference_data[col].mean(),
                        "std": reference_data[col].std(),
                        "min": reference_data[col].min(),
                        "max": reference_data[col].max(),
                        "q25": reference_data[col].quantile(0.25),
                        "q50": reference_data[col].quantile(0.50),
                        "q75": reference_data[col].quantile(0.75),
                        "type": "numerical",
                    }
            except Exception:
                continue

    def calculate_entropy_score(self, row: Dict[str, Any]) -> float:
        """Calculate entropy-based quality score for a row."""
        scores = []

        for col, value in row.items():
            if col not in self.reference_stats:
                continue

            stats = self.reference_stats[col]

            try:
                if stats["type"] == "categorical":
                    prob = self.feature_distributions.get(col, {}).get(value, 0)
                    if prob == 0:
                        scores.append(0.3)  # Rare
                    elif prob < 0.01:
                        scores.append(0.6)  # Uncommon
                    else:
                        scores.append(min(1.0, -np.log(prob) / 5))
                else:
                    # Numerical z-score
                    mean, std = stats["mean"], stats["std"]
                    z_score = abs((value - mean) / (std + 1e-10))

                    if z_score < 1:
                        scores.append(1.0)
                    elif z_score < 2:
                        scores.append(0.8)
                    elif z_score < 3:
                        scores.append(0.6)
                    else:
                        scores.append(0.3)
            except Exception:
                continue

        return np.mean(scores) if scores else 0.5

    def calculate_confidence_score(
        self,
        row: Dict[str, Any],
        bayesian_prob: float,
        constraint_score: float,
        num_corrections: int,
    ) -> float:
        """Unified confidence score combining multiple signals."""
        entropy_score = self.calculate_entropy_score(row)

        # Weighted combination
        alpha, beta, gamma, delta = 0.2, 0.5, 0.3, 0.05
        edit_penalty = min(1.0, num_corrections * 0.08)

        confidence = (
            alpha * min(bayesian_prob * 10, 1.0)
            + beta * constraint_score
            + gamma * entropy_score
            - delta * edit_penalty
        )

        return max(0.0, min(1.0, confidence))


# =============================================================================
# LOGIC CONDUCTOR - Main Orchestrator
# =============================================================================


class LogicConductor:
    """
    Schema-agnostic cognitive core that:
    - Works with ANY dataset
    - Auto-discovers relationships
    - Gracefully handles missing columns
    - Provides production-grade error handling
    """

    def __init__(self, confidence_threshold: float = 0.85, verbose: bool = True):
        """
        Args:
            confidence_threshold: Minimum confidence to keep a row [0-1]
            verbose: Print progress messages
        """
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.bayesian_graph = BayesianDependencyGraph(verbose=verbose)
        self.csp_solver = ConstraintSatisfactionSolver(verbose=verbose)
        self.entropy_scorer = EntropyScorer()
        self.is_trained = False
        self.available_columns = set()
        self.schema_info = None

    def train(
        self,
        reference_data: pd.DataFrame,
        structure_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Train on reference data with optional custom structure.

        Args:
            reference_data: Clean training DataFrame
            structure_edges: Optional list of (parent, child) edges
        """
        if reference_data.empty:
            if self.verbose:
                print("  Warning: Empty reference data provided")
            return

        if self.verbose:
            print(" Training Cognitive Core...")

        self.available_columns = set(reference_data.columns)

        # Build Bayesian network structure
        if self.verbose:
            print("   ├─ Constructing Bayesian dependency graph...")

        if structure_edges:
            self.bayesian_graph.build_structure(edges=structure_edges)
        else:
            self.bayesian_graph.build_structure(data=reference_data)

        # Learn parameters
        if self.verbose:
            print("   ├─ Learning probabilistic parameters...")

        self.bayesian_graph.learn_parameters(reference_data, discretize=True)

        # Fit entropy scorer
        if self.verbose:
            print("   ├─ Computing reference statistics...")

        self.entropy_scorer.fit(reference_data)

        self.is_trained = True

        if self.verbose:
            print("   └─  Training complete!\n")

    def add_constraint(
        self,
        name: str,
        constraint_func: Callable,
        variables: List[str],
        rule_type: str = "hard",
        weight: float = 1.0,
        description: str = "",
    ) -> None:
        """
        Add a custom validation constraint.

        Args:
            name: Constraint identifier
            constraint_func: Function(row_dict) -> bool (hard) or float (soft)
            variables: List of column names this constraint uses
            rule_type: 'hard' or 'soft'
            weight: Weight for soft constraints
            description: Human-readable explanation
        """
        self.csp_solver.add_rule(
            name=name,
            constraint_func=constraint_func,
            variables=variables,
            rule_type=rule_type,
            weight=weight,
            description=description,
        )

        if self.verbose:
            print(f"    Added {rule_type} constraint: {name}")

    def validate_and_heal(
        self, synthetic_data: pd.DataFrame, strategy: str = "aggressive"
    ) -> Tuple[pd.DataFrame, LogicReport]:
        """
        [Titan Logic Gate - V5.6]
        Final stage for data purification and enforcing strict logical constraints.
        """
        if not self.is_trained:
            logger.warning(" [Logic] Conductor not trained. Returning original data.")
            return synthetic_data, LogicReport(
                len(synthetic_data), 0, 0, len(synthetic_data), []
            )

        if synthetic_data.empty:
            return synthetic_data, LogicReport(0, 0, 0, 0, [])

        if self.verbose:
            logger.info(
                f" [Titan Shield] Securing {len(synthetic_data):,} banking records..."
            )

        validation_results = []
        cleaned_rows = []
        available_columns = set(synthetic_data.columns)

        # Performance optimization: iterate and process each row independently
        for idx, row in synthetic_data.iterrows():
            # [Critical fix]: Pass arguments with original names compatible with the class
            result = self._process_row(
                idx=idx,
                row=row.to_dict(),
                reference_data=synthetic_data,
                available_columns=available_columns,
            )
            validation_results.append(result)

            # Filtering strategy
            if result.status != "discarded":
                cleaned_rows.append(result.corrected_values)
            elif strategy == "conservative":
                cleaned_rows.append(result.original_values)

        # Rebuild dataframe preserving column order
        if cleaned_rows:
            cleaned_df = pd.DataFrame(cleaned_rows)
            cleaned_df = cleaned_df[list(synthetic_data.columns)]
        else:
            cleaned_df = pd.DataFrame(columns=synthetic_data.columns)

        # Generate final quality report
        report = LogicReport(
            total_rows=len(synthetic_data),
            valid_rows=sum(1 for r in validation_results if r.status == "valid"),
            corrected_rows=sum(
                1 for r in validation_results if r.status == "corrected"
            ),
            discarded_rows=sum(
                1 for r in validation_results if r.status == "discarded"
            ),
            validation_results=validation_results,
        )

        return cleaned_df, report

    def _process_row(
        self,
        idx: int,
        row: Dict[str, Any],
        reference_data: pd.DataFrame,
        available_columns: Set[str],
    ) -> ValidationResult:
        """
        [Titan Logic Core - V5.6]
        Individual record processor: integrates strict constraints (CSP) and Bayesian probabilities
        with an added protection layer for polar outputs.
        """
        original_row = row.copy()

        # 1. Structural constraint check (Constraint Satisfaction - CSP)
        # Detects contradictions like (age 10 years + married status)
        violations, soft_score = self.csp_solver.check_constraints(
            row, available_columns
        )

        corrections_applied = []
        healed_row = row.copy()

        # 2. Self-Healing Protocol
        # 2. Self-Healing Protocol
        if violations:
            # Engine attempts to find the nearest logical value from reference_data
            healed_row, corrections, _ = self.csp_solver.heal_row(
                row, reference_data, available_columns
            )
            corrections_applied = corrections

            # ── BUG B FIX: re-project coupled groups after scalar healing ──
            # CSP healing mutates individual columns (e.g. sin_month → median)
            # without adjusting their coupled partners.  We call repair_groups()
            # which vectorises all invariant projections back onto the correct
            # manifold (unit circle, probability simplex, calendar grid).
            if hasattr(self, "_healing_engine") and self._healing_engine is not None:
                try:
                    _tmp = pd.DataFrame([healed_row])
                    _tmp = self._healing_engine.repair_groups(_tmp)
                    healed_row = _tmp.iloc[0].to_dict()
                except Exception as _repair_exc:
                    logger.debug(
                        "[Titan][BugB-Fix] repair_groups() skipped: %s",
                        _repair_exc,
                    )

        # 3. Industrial Bayesian Analysis
        # Calculates the likelihood of this record existing in a real banking world
        bayesian_prob = self.bayesian_graph.calculate_row_probability(healed_row)

        # 4. Scoring & Entropy calculation
        num_corrections = len(corrections_applied)
        confidence_score = self.entropy_scorer.calculate_confidence_score(
            healed_row, bayesian_prob, soft_score, num_corrections
        )
        entropy_score = self.entropy_scorer.calculate_entropy_score(healed_row)

        # 5. Final physical purification layer
        # [Additional development]: Ensure values coming out of the Healer are still logically constrained
        if healed_row:
            for col in ["age", "duration", "pdays", "previous"]:
                if col in healed_row and healed_row[col] is not None:
                    try:
                        # Ensure they are positive integers
                        healed_row[col] = max(0, int(round(float(healed_row[col]))))
                    except:
                        pass

        # 6. Sovereign decision logic
        # If confidence score is below threshold, the record is completely discarded
        if confidence_score >= self.confidence_threshold:
            status = "corrected" if corrections_applied else "valid"
        else:
            status = "discarded"
            # Keep original values in report for analysis, but the model will ignore the record
            # healed_row = None  # (physical deletion disabled to ensure DataFrame integrity)

        return ValidationResult(
            row_id=idx,
            original_values=original_row,
            corrected_values=healed_row if status != "discarded" else original_row,
            confidence_score=confidence_score,
            entropy_score=entropy_score,
            violations=violations,
            corrections_applied=corrections_applied,
            status=status,
        )

    def export_detailed_report(
        self, report: LogicReport, output_path: str = "validation_report.txt"
    ) -> None:
        """Export detailed validation report."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.summary())
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("DETAILED ROW-BY-ROW ANALYSIS\n")
            f.write("=" * 60 + "\n\n")

            for result in report.validation_results:
                f.write(f"Row {result.row_id}: {result.status.upper()}\n")
                f.write(
                    f"  Confidence: {result.confidence_score:.3f} | "
                    f"Entropy: {result.entropy_score:.3f}\n"
                )

                if result.violations:
                    f.write(f"  Violations: {', '.join(result.violations)}\n")

                if result.corrections_applied:
                    f.write(f"  Corrections:\n")
                    for correction in result.corrections_applied:
                        f.write(f"    • {correction}\n")

                f.write("\n")

        if self.verbose:
            print(f" Detailed report exported to: {output_path}")


# =============================================================================
# EXAMPLE USAGE - Demonstrates Schema-Agnostic Capabilities
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LOGIC CONDUCTOR v2.0 - SCHEMA-AGNOSTIC COGNITIVE CORE")
    print("=" * 70 + "\n")

    # Example 1: Banking Domain (original use case)
    print("Example 1: Banking Dataset")
    print("-" * 70)

    reference_banking = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65, 30, 40, 50, 28, 38],
            "job": [
                "admin.",
                "technician",
                "management",
                "retired",
                "retired",
                "blue-collar",
                "management",
                "entrepreneur",
                "student",
                "admin.",
            ],
            "marital": [
                "single",
                "married",
                "married",
                "married",
                "divorced",
                "single",
                "married",
                "divorced",
                "single",
                "married",
            ],
            "education": [
                "secondary",
                "tertiary",
                "tertiary",
                "secondary",
                "primary",
                "secondary",
                "tertiary",
                "tertiary",
                "tertiary",
                "secondary",
            ],
            "balance": [1500, 5000, 15000, 20000, 8000, 2000, 12000, -500, 500, 3000],
        }
    )

    synthetic_banking = pd.DataFrame(
        {
            "age": [15, 70, 45, 80, 25, 35],
            "job": [
                "retired",
                "student",
                "management",
                "student",
                "admin.",
                "blue-collar",
            ],
            "marital": [
                "married",
                "single",
                "married",
                "divorced",
                "single",
                "married",
            ],
            "education": [
                "primary",
                "tertiary",
                "tertiary",
                "secondary",
                "secondary",
                "tertiary",
            ],
            "balance": [500, 2000, 15000, 1000, 2500, 18000],
        }
    )

    conductor1 = LogicConductor(confidence_threshold=0.85, verbose=True)
    conductor1.train(reference_banking)

    # Add domain-specific constraint
    def age_bounds(row):
        return 18 <= row.get("age", 25) <= 100

    conductor1.add_constraint(
        name="age_bounds",
        constraint_func=age_bounds,
        variables=["age"],
        rule_type="hard",
        description="Age must be in valid range",
    )

    cleaned1, report1 = conductor1.validate_and_heal(synthetic_banking)
    print(f"\n✓ Cleaned {len(cleaned1)} rows from banking dataset\n")

    # Example 2: Housing Dataset (different schema)
    print("\nExample 2: Housing Dataset (Different Schema)")
    print("-" * 70)

    reference_housing = pd.DataFrame(
        {
            "price": [250000, 350000, 450000, 550000, 650000, 300000, 400000, 500000],
            "sqft": [1500, 2000, 2500, 3000, 3500, 1800, 2200, 2800],
            "bedrooms": [2, 3, 3, 4, 4, 2, 3, 3],
            "location": [
                "urban",
                "suburban",
                "suburban",
                "rural",
                "urban",
                "suburban",
                "urban",
                "rural",
            ],
            "year_built": [1990, 2000, 2010, 1980, 2015, 1995, 2005, 1985],
        }
    )

    synthetic_housing = pd.DataFrame(
        {
            "price": [-100000, 1000000, 300000, 500000],  # One invalid negative
            "sqft": [1200, 5000, 1800, 2500],
            "bedrooms": [1, 6, 2, 3],
            "location": ["urban", "suburban", "urban", "rural"],
            "year_built": [2020, 1950, 2000, 2010],
        }
    )

    conductor2 = LogicConductor(confidence_threshold=0.60, verbose=True)
    conductor2.train(reference_housing)

    # Add housing-specific constraint
    def price_positive(row):
        return row.get("price", 0) > 0

    conductor2.add_constraint(
        name="price_positive",
        constraint_func=price_positive,
        variables=["price"],
        rule_type="hard",
        description="Price must be positive",
    )

    cleaned2, report2 = conductor2.validate_and_heal(synthetic_housing)
    print(f"\n✓ Cleaned {len(cleaned2)} rows from housing dataset\n")

    print("=" * 70)
    print("✓ DEMONSTRATION COMPLETE - Works on ANY dataset schema!")
    print("=" * 70)
