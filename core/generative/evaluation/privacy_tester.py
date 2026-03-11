from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union

"""
core/generative/evaluation/privacy_tester.py
Production-grade module for assessing privacy risks in a hybrid data generator:
- Membership risk index via non-invasive shadow models
- Information leakage analysis (unique values/rare patterns)
- Simplified differential privacy analyzer
- General privacy tester measuring distribution proximity between real and generated data

Design:
- Clear and replaceable interfaces
- Separation of concerns: data transformer, metrics, reports
- Vectorized O(n) operations per batch
- No internal layer creation within forward; no dependency on generator details outside generation contract

Copyright: Microsoft AI (Production-grade re-engineering)
"""
from core.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/system.log")


from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union

import numpy as np
import pandas as pd
import logging
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ========= Logging =========


import sys

logger = logging.getLogger(__name__)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)

    # Force UTF-8 encoding (UnicodeEncodeError fix)
    try:
        handler.stream.reconfigure(encoding="utf-8")
    except Exception:
        pass  # For compatibility with older Python versions

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False


# ========= Type aliases and contracts =========

DataFrameLike = pd.DataFrame
ArrayLike = np.ndarray


class GeneratorInterface(Protocol):
    """Unified protocol for generators to ensure compatibility with the hybrid system."""

    def generate(self, num_samples: int, **kwargs) -> Union[ArrayLike, DataFrameLike]:
        """Generates new samples of size [num_samples, D]."""
        ...

    def to(self, device: torch.device) -> "GeneratorInterface":
        """Moves the generator to a computation device if needed."""
        ...


# ========= Errors =========


class PrivacyError(ValueError):
    """Custom exception for privacy evaluation errors with precise diagnostic messages."""

    pass


# ========= Utilities =========


def ensure_dataframe(data: Union[ArrayLike, DataFrameLike], name: str) -> DataFrameLike:
    """
    Converts input to DataFrame and checks that it is not empty.

    Raises:
        PrivacyError: If input is empty or unsupported.
    """
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        raise PrivacyError(
            f"{name} must be DataFrame or ndarray, but got type {type(data)}."
        )

    if df.empty:
        raise PrivacyError(f"{name} is empty; cannot evaluate privacy on empty data.")
    return df


def safe_numeric(df: DataFrameLike) -> DataFrameLike:
    """
    Returns only numeric columns from the dataframe.
    """
    return df.select_dtypes(include=[np.number])


# ========= Configs =========


@dataclass(frozen=True)
class PrivacyEvalConfig:
    """
    Privacy evaluation settings.
    """

    n_shadow_models: int = 3
    test_size: float = 0.3
    random_state: int = 42
    rf_estimators: int = 100
    rf_max_depth: int = 10

    def __post_init__(self) -> None:
        if self.n_shadow_models <= 0:
            raise PrivacyError("Number of shadow models must be > 0.")
        if not (0.0 < self.test_size < 1.0):
            raise PrivacyError("test_size must be within (0, 1).")
        if self.rf_estimators <= 0 or self.rf_max_depth <= 0:
            raise PrivacyError("RandomForest parameters are invalid.")


# ========= Membership risk assessor (non-invasive) =========


class MembershipRiskAssessor:
    """
    Assesses membership risk in a non-invasive manner.

    Idea:
    - Create shadow models trained on subsets of real + generated data.
    - Measure the confidence/probability gap between training and test data for those models.
    - If models can significantly distinguish between "in" and "out" (high AUC), this indicates higher risk.

    This assessor is not used to launch practical attacks; it provides quantitative indicators to improve privacy.

    IO:
    - assess(real_train, real_test, synthetic_data) -> Dict[str, Any]
    """

    def __init__(self, config: Optional[PrivacyEvalConfig] = None) -> None:
        self.config = config or PrivacyEvalConfig()
        self.history: List[Dict[str, Any]] = []

    def assess(
        self,
        real_train: Union[ArrayLike, DataFrameLike],
        real_test: Union[ArrayLike, DataFrameLike],
        synthetic_data: Union[ArrayLike, DataFrameLike],
    ) -> Dict[str, Any]:
        """
        Assesses membership risk index using shadow models.

        Returns:
            Dict[str, Any]: Contains risk indicators (auc_mean, auc_std) and detailed information.
        """
        rt = ensure_dataframe(real_train, "real_train")
        rv = ensure_dataframe(real_test, "real_test")
        syn = ensure_dataframe(synthetic_data, "synthetic_data")

        # Work only on numeric columns to avoid unsafe conversions
        rt_num = safe_numeric(rt)
        rv_num = safe_numeric(rv)
        syn_num = safe_numeric(syn)

        aucs: List[float] = []
        details: List[Dict[str, Any]] = []

        for i in range(self.config.n_shadow_models):
            try:
                # Shadow split: merge (train + synthetic) then random split
                merged = pd.concat([rt_num, syn_num], ignore_index=True)
                shadow_in, shadow_out = train_test_split(
                    merged,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state + i,
                )

                # Train shadow model to distinguish between "in" and "out" via model confidence
                clf = RandomForestClassifier(
                    n_estimators=self.config.rf_estimators,
                    max_depth=self.config.rf_max_depth,
                    random_state=self.config.random_state + i,
                    class_weight="balanced",
                    n_jobs=-1,
                )

                # Build binary classification data: in=1, out=0
                X_in = shadow_in
                X_out = (
                    rv_num  # Real test is considered "out" compared to training shadow
                )
                y_in = np.ones(len(X_in), dtype=np.int64)
                y_out = np.zeros(len(X_out), dtype=np.int64)

                X = pd.concat([X_in, X_out], ignore_index=True)
                y = np.concatenate([y_in, y_out], axis=0)

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.config.test_size,
                    stratify=y,
                    random_state=self.config.random_state + i,
                )

                clf.fit(X_train, y_train)
                proba = clf.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, proba))

                aucs.append(auc)
                details.append(
                    {
                        "shadow_index": i + 1,
                        "auc": auc,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to evaluate shadow model {i+1}: {e}")
                continue

        if not aucs:
            raise PrivacyError(
                "Could not compute membership risk indicators; check dimension consistency and numeric data."
            )

        result = {
            "metric_name": "membership_risk_index",
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "shadow_details": details,
            "config": asdict(self.config),
        }
        self.history.append(result)
        return result


# ========= Information leakage analyzer =========


class InformationLeakageAnalyzer:
    """
    Analyzes information leakage between real and generated data.

    Measures:
    - Unique value overlap for categorical/textual columns
    - Rare pattern overlap (unique/rare rows)

    IO:
    - analyze(real_data, synthetic_data) -> Dict[str, float]
    """

    def analyze(
        self,
        real_data: Union[ArrayLike, DataFrameLike],
        synthetic_data: Union[ArrayLike, DataFrameLike],
    ) -> Dict[str, float]:
        rd = ensure_dataframe(real_data, "real_data")
        sd = ensure_dataframe(synthetic_data, "synthetic_data")

        # Measure unique value overlap for non-numeric columns
        unique_real: set = set()
        unique_synth: set = set()

        for col in rd.columns:
            if rd[col].dtype == "object" or str(rd[col].dtype) == "category":
                unique_real.update(pd.Series(rd[col]).dropna().unique().tolist())
                unique_synth.update(pd.Series(sd[col]).dropna().unique().tolist())

        total_unique = len(unique_real)
        unique_overlap_ratio = (
            (len(unique_real & unique_synth) / total_unique)
            if total_unique > 0
            else 0.0
        )

        # Rare patterns (unique rows)
        rare_real = self._extract_rare_patterns(rd, n=10)
        rare_synth = self._extract_rare_patterns(sd, n=10)
        total_rare = len(rare_real)
        rare_overlap_ratio = (
            (len(set(rare_real) & set(rare_synth)) / total_rare)
            if total_rare > 0
            else 0.0
        )

        return {
            "unique_value_overlap": float(unique_overlap_ratio),
            "rare_pattern_overlap": float(rare_overlap_ratio),
        }

    def _extract_rare_patterns(self, data: DataFrameLike, n: int = 10) -> List[str]:
        """
        Extracts rare patterns via unique rows (simplified representation).
        """
        unique_rows = data.drop_duplicates()
        if len(unique_rows) > n:
            sample = unique_rows.sample(n=n, random_state=42)
        else:
            sample = unique_rows

        return ["|".join(map(str, row.values)) for _, row in sample.iterrows()]


# ========= Differential privacy analyzer (simplified) =========


class DifferentialPrivacyAnalyzer:
    """
    Simplified differential privacy analyzer (preliminary).

    Approximates (ε, δ) bounds and generates an interpretive report.
    """

    def __init__(self) -> None:
        self.epsilon_history: List[float] = []
        self.delta_history: List[float] = []

    def compute_dp_bounds(
        self, noise_scale: float, batch_size: int, dataset_size: int, iterations: int
    ) -> Dict[str, float]:
        """
        Computes approximate differential privacy bounds (ε, δ).

        Returns:
            Dict[str, float]: epsilon, delta, privacy_budget_used
        """
        if (
            dataset_size <= 0
            or batch_size <= 0
            or iterations <= 0
            or noise_scale <= 0.0
        ):
            raise PrivacyError(
                "Differential privacy parameters must be positive and valid."
            )

        delta = 1.0 / float(dataset_size)
        epsilon = (
            noise_scale
            * np.sqrt(2.0 * iterations * np.log(1.25 / delta))
            / float(batch_size)
        )

        self.epsilon_history.append(float(epsilon))
        self.delta_history.append(float(delta))

        return {
            "epsilon": float(epsilon),
            "delta": float(delta),
            "privacy_budget_used": float(min(epsilon, 10.0)),
        }

    def generate_dp_report(self) -> str:
        """
        Generates a brief textual report on differential privacy.
        """
        if not self.epsilon_history:
            return "Differential privacy bounds have not been computed yet."

        avg_epsilon = float(np.mean(self.epsilon_history))
        max_epsilon = float(np.max(self.epsilon_history))
        avg_delta = float(np.mean(self.delta_history))

        lines = [
            "Differential Privacy Report:",
            f"Average ε: {avg_epsilon:.4f}",
            f"Maximum ε: {max_epsilon:.4f}",
            f"Average δ: {avg_delta:.6f}",
            "",
            "Interpretation:",
        ]
        if avg_epsilon < 1.0:
            lines.append(" Strong differential privacy (ε < 1)")
        elif avg_epsilon < 3.0:
            lines.append(" Acceptable differential privacy (ε < 3)")
        elif avg_epsilon < 10.0:
            lines.append(" Weak differential privacy (ε < 10)")
        else:
            lines.append(" Insufficient differential privacy")

        return "\n".join(lines)


# ========= General privacy tester =========


class PrivacyTester:
    """
    General privacy tester measuring distribution proximity between real and generated data.

    Computes a simplified privacy score via differences in means for numeric columns:
    The smaller the difference, the higher the privacy (no over-matching of specific real data points).
    """

    def __init__(self, real_data: Union[ArrayLike, DataFrameLike]) -> None:
        self.real_df = ensure_dataframe(real_data, "real_data")
        self.real_num = safe_numeric(self.real_df)

    def compute_privacy_score(
        self, synthetic_data: Union[ArrayLike, DataFrameLike]
    ) -> float:
        """
        Computes privacy score via average differences of numeric columns.
        """
        synth_df = ensure_dataframe(synthetic_data, "synthetic_data")
        synth_num = safe_numeric(synth_df)

        common_cols = [c for c in self.real_num.columns if c in synth_num.columns]
        if not common_cols:
            raise PrivacyError(
                "No common numeric columns between real and generated data."
            )

        diffs: List[float] = []
        for col in common_cols:
            r = pd.to_numeric(self.real_num[col], errors="coerce").dropna().values
            s = pd.to_numeric(synth_num[col], errors="coerce").dropna().values
            if r.size == 0 or s.size == 0:
                continue
            diff = float(abs(np.mean(r) - np.mean(s)))
            diffs.append(diff)

        if not diffs:
            return 1.0  # Insufficient information, assume high privacy conservatively

        # Convert mean differences to score [0,1]: smaller difference -> higher score
        mean_diff = float(np.mean(diffs))
        score = 1.0 / (1.0 + mean_diff)
        return float(min(1.0, max(0.0, score)))


# ========= Orchestrator for privacy evaluation =========


class PrivacyEvaluationEngine:
    """
    Privacy evaluation orchestrator: coordinates membership assessor, leakage analyzer, general tester, and differential analyzer.

    IO:
    - evaluate(real_train, real_test, synthetic) -> Dict[str, Any]
    """

    def __init__(self, config: Optional[PrivacyEvalConfig] = None) -> None:
        self.config = config or PrivacyEvalConfig()
        self.membership_assessor = MembershipRiskAssessor(self.config)
        self.leakage_analyzer = InformationLeakageAnalyzer()
        self.dp_analyzer = DifferentialPrivacyAnalyzer()

    def evaluate(
        self,
        real_train: Union[ArrayLike, DataFrameLike],
        real_test: Union[ArrayLike, DataFrameLike],
        synthetic_data: Union[ArrayLike, DataFrameLike],
        dp_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a comprehensive and safe privacy evaluation.
        """
        # Membership risk index
        membership_idx = self.membership_assessor.assess(
            real_train, real_test, synthetic_data
        )

        # Information leakage
        leakage = self.leakage_analyzer.analyze(real_train, synthetic_data)

        # General privacy score
        tester = PrivacyTester(real_train)
        privacy_score = tester.compute_privacy_score(synthetic_data)

        # Differential privacy (optional)
        dp_report: Optional[str] = None
        dp_bounds: Optional[Dict[str, float]] = None
        if dp_params:
            dp_bounds = self.dp_analyzer.compute_dp_bounds(
                noise_scale=float(dp_params.get("noise_scale", 1.0)),
                batch_size=int(dp_params.get("batch_size", 256)),
                dataset_size=int(dp_params.get("dataset_size", 10000)),
                iterations=int(dp_params.get("iterations", 1000)),
            )
            dp_report = self.dp_analyzer.generate_dp_report()

        result = {
            "membership_risk_index": membership_idx,
            "information_leakage": leakage,
            "privacy_score": float(privacy_score),
            "dp_bounds": dp_bounds,
            "dp_report": dp_report,
            "metadata": {"config": asdict(self.config)},
        }
        return result

    def generate_report(self, evaluation: Dict[str, Any]) -> str:
        """
        Generates a simplified textual report on privacy.
        """
        lines: List[str] = []
        lines.append("=" * 80)
        lines.append("Privacy Evaluation Report (Production-Ready and Safe)")
        lines.append("=" * 80)

        # General score
        lines.append(
            f"\nGeneral Privacy Score: {evaluation.get('privacy_score', 0.0):.2f}"
        )

        # Membership risk
        mri = evaluation.get("membership_risk_index", {})
        if mri:
            lines.append("\nMembership Risk Index:")
            lines.append("-" * 40)
            lines.append(
                f"Average shadow AUC: {mri.get('auc_mean', 0.0):.4f} ± {mri.get('auc_std', 0.0):.4f}"
            )
            lines.append(f"Shadow models used: {len(mri.get('shadow_details', []))}")

        # Information leakage
        leak = evaluation.get("information_leakage", {})
        if leak:
            lines.append("\nInformation Leakage Indicators:")
            lines.append("-" * 40)
            lines.append(
                f"Unique value overlap: {leak.get('unique_value_overlap', 0.0):.4f}"
            )
            lines.append(
                f"Rare pattern overlap: {leak.get('rare_pattern_overlap', 0.0):.4f}"
            )

        # Differential privacy
        bounds = evaluation.get("dp_bounds")
        report = evaluation.get("dp_report")
        if bounds:
            lines.append("\nDifferential Privacy Bounds (ε, δ):")
            lines.append("-" * 40)
            lines.append(f"epsilon: {bounds.get('epsilon', 0.0):.4f}")
            lines.append(f"delta: {bounds.get('delta', 0.0):.6f}")
            if report:
                lines.append("\n" + report)

        # Brief recommendations
        lines.append("\nRecommendations:")
        lines.append("-" * 40)
        auc_mean = float(mri.get("auc_mean", 0.5)) if mri else 0.5
        if auc_mean <= 0.55 and leak.get("rare_pattern_overlap", 0.0) < 0.2:
            lines.append(
                " Low risk; continue monitoring batches and gradually improve differential noise."
            )
        elif auc_mean <= 0.65 or leak.get("unique_value_overlap", 0.0) < 0.3:
            lines.append(
                " Medium risk; enhance model generalization and reduce matching of rare samples."
            )
        else:
            lines.append(
                " High risk; review privacy policies and add stronger protection mechanisms (obfuscation/precision reduction/regularization)."
            )

        return "\n".join(lines)
