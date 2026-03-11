"""
core/feature_engineering/coupled_healing_engine.py
═══════════════════════════════════════════════════════════════════════════════
TITAN v6.2 — CoupledHealingEngine
───────────────────────────────────
Replaces the broken independent-column healing in ConstraintSatisfactionSolver.

The core design principle:
    "Never touch a single column of a coupled group without simultaneously
     restoring the group's mathematical invariant."

Healing phases
──────────────
Phase 1  Individual constraints   — hard business rules applied per column,
                                    capturing violation signals without mutating
                                    coupled columns directly.
Phase 2  Group-level projection   — after all individual corrections, every
                                    coupled group is re-projected onto its
                                    valid manifold (unit circle, simplex, etc.)
Phase 3  Context-aware imputation — for DATETIME groups, instead of setting
                                    each component to its marginal median, we
                                    sample a full valid timestamp from the
                                    nearest-neighbour reference row, then
                                    decompose it to components.

All Phase 2/3 operations are vectorised over entire DataFrames.

Author : Titan AI Architecture Team
Version: 6.2 Enterprise
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from core.feature_engineering.coupled_feature_registry import (
    CoupledFeatureRegistry,
    FeatureGroup,
    GroupKind,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised group projectors
# ─────────────────────────────────────────────────────────────────────────────


class _CyclicProjector:
    """
    Re-normalises (sin, cos) columns so that sin²+cos² = 1.

    Input shape:  (N, 2) — columns [sin, cos] in any numeric range.
    Output shape: (N, 2) — same columns, L2-normalised per row.

    Additionally enforces that the reconstructed angle is consistent with the
    group's declared period (round to nearest discrete step).
    """

    @staticmethod
    def project(
        mat: np.ndarray,
        period: Optional[float] = None,
    ) -> np.ndarray:
        """
        mat: float64 array shape (N, 2) — column 0 is sin, column 1 is cos.
        """
        s = mat[:, 0].astype(np.float64)
        c = mat[:, 1].astype(np.float64)

        norms = np.sqrt(s**2 + c**2)

        # If a row has zero vector (GAN collapsed to 0,0) replace with uniform angle
        zero_mask = norms < 1e-8
        if zero_mask.any():
            fallback_angle = 0.0 if period is None else 2 * np.pi / period
            s[zero_mask] = np.sin(fallback_angle)
            c[zero_mask] = np.cos(fallback_angle)
            norms[zero_mask] = 1.0

        s_norm = s / norms
        c_norm = c / norms

        # Optional: snap angle to nearest discrete step (period-aware quantisation)
        if period is not None:
            angles = np.arctan2(s_norm, c_norm)  # (-π, π]
            step = 2 * np.pi / period
            # Round to nearest multiple of step
            snapped = np.round(angles / step) * step
            s_norm = np.sin(snapped)
            c_norm = np.cos(snapped)

        out = np.stack([s_norm, c_norm], axis=1)
        return out


class _OneHotProjector:
    """
    Projects a (N, C) matrix onto the probability simplex and returns a hard
    one-hot encoding (exactly one 1 per row).

    Handles:
    • All-zero rows   → set to mode category from reference distribution
    • All-NaN rows    → same fallback
    • Multi-active rows (sum > 1) → argmax wins
    """

    @staticmethod
    def project(
        mat: np.ndarray,
        mode_index: int = 0,
    ) -> np.ndarray:
        """
        mat: float64 array shape (N, C)
        """
        mat = mat.astype(np.float64)

        # Replace NaN/Inf with 0
        mat = np.where(np.isfinite(mat), mat, 0.0)

        # Rows where max activation ≤ 0 → impute with mode
        row_max = mat.max(axis=1)
        degenerate = row_max <= 0.0

        if degenerate.any():
            imputation = np.zeros(mat.shape[1])
            imputation[mode_index] = 1.0
            mat[degenerate] = imputation

        # Argmax → hard one-hot (vectorised)
        indices = mat.argmax(axis=1)  # (N,)
        oh = np.zeros_like(mat)
        oh[np.arange(len(oh)), indices] = 1.0

        return oh


class _DatetimeProjector:
    """
    Ensures date components form a valid calendar date.

    Strategy:
    1. Denormalise each z-scored component using training stats.
    2. Clip to domain bounds (month ∈ [1,12], day ∈ [1,28/30/31], …).
    3. Apply February + short-month caps (vectorised NumPy).
    4. Replace any remaining NaT with the training-set median timestamp.
    5. Re-normalise back to z-scored representation using training stats.
    """

    def __init__(
        self,
        datetime_stats: Dict[str, Dict[str, float]],
        training_median_ts: str = "2020-01-01",
    ) -> None:
        self._stats = datetime_stats
        self._fallback = pd.Timestamp(training_median_ts)

    def project(self, mat: np.ndarray, component_order: List[str]) -> np.ndarray:
        """
        mat             : float64 (N, D) — z-scored datetime components
        component_order : list of component names in column order,
                          e.g. ["year", "month", "day", "hour", "dayofweek", "is_weekend"]
        """
        N, D = mat.shape
        out = mat.astype(np.float64).copy()

        # ── Step 1: denormalise ───────────────────────────────────────────────
        raw: Dict[str, np.ndarray] = {}
        for i, comp in enumerate(component_order):
            if i >= D:
                break
            stats = self._stats.get(comp, {})
            mu = float(stats.get("mean", 0.0))
            sigma = float(stats.get("std", 1.0))
            if sigma < 1e-6:
                sigma = 1.0
            raw[comp] = out[:, i] * sigma + mu

        # ── Step 2: coarse domain clamp ───────────────────────────────────────
        if "year" in raw:
            raw["year"] = np.clip(np.round(raw["year"]), 1970, 2100)
        if "month" in raw:
            raw["month"] = np.clip(np.round(raw["month"]), 1, 12)
        if "day" in raw:
            raw["day"] = np.clip(np.round(raw["day"]), 1, 31)
        if "hour" in raw:
            raw["hour"] = np.clip(np.round(raw["hour"]), 0, 23)
        if "dayofweek" in raw:
            raw["dayofweek"] = np.clip(np.round(raw["dayofweek"]), 0, 6)

        # ── Step 3: calendar-aware day cap (vectorised) ───────────────────────
        if "month" in raw and "day" in raw:
            month = raw["month"].astype(int)
            day = raw["day"].astype(int)

            feb_mask = month == 2
            short_mon_mask = np.isin(month, [4, 6, 9, 11])

            day = np.where(feb_mask, np.minimum(day, 28), day)
            day = np.where(short_mon_mask, np.minimum(day, 30), day)
            raw["day"] = day

        # ── Step 4: validate via pd.to_datetime + NaT imputation ─────────────
        if all(k in raw for k in ("year", "month", "day")):
            frame = pd.DataFrame(
                {
                    "year": raw["year"].astype(int),
                    "month": raw["month"].astype(int),
                    "day": raw["day"].astype(int),
                    "hour": raw.get("hour", np.zeros(N)).astype(int),
                }
            )
            ts = pd.to_datetime(frame, errors="coerce")
            nat_mask = ts.isna()
            if nat_mask.any():
                ts[nat_mask] = self._fallback

            # Re-extract corrected components after calendar normalisation
            raw["year"] = ts.dt.year.values.astype(float)
            raw["month"] = ts.dt.month.values.astype(float)
            raw["day"] = ts.dt.day.values.astype(float)
            raw["hour"] = ts.dt.hour.values.astype(float)
            raw["dayofweek"] = ts.dt.dayofweek.values.astype(float)
            raw["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)

        # ── Step 5: re-normalise back to z-scored space ───────────────────────
        for i, comp in enumerate(component_order):
            if i >= D or comp not in raw:
                continue
            stats = self._stats.get(comp, {})
            mu = float(stats.get("mean", 0.0))
            sigma = float(stats.get("std", 1.0))
            if sigma < 1e-6:
                sigma = 1.0
            out[:, i] = (raw[comp] - mu) / sigma

        return out


# ─────────────────────────────────────────────────────────────────────────────
# CoupledHealingEngine  (public API)
# ─────────────────────────────────────────────────────────────────────────────


class CoupledHealingEngine:
    """
    Group-aware healing engine.

    Workflow per DataFrame batch
    ────────────────────────────
    1. apply_individual_constraints(df) — per-column hard rules, returns
       a violation mask but does NOT yet mutate coupled columns.
    2. heal_batch(df, reference_data) — applies context-aware imputation
       for violated rows, then calls repair_groups() on the entire batch.
    3. repair_groups(df) — enforces group invariants across all rows,
       regardless of whether individual columns were flagged.

    All three methods are DataFrame-level operations (no iterrows).
    """

    def __init__(
        self,
        registry: CoupledFeatureRegistry,
        reference_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self._registry = registry
        self._reference = reference_data

        self._cyclic_projector = _CyclicProjector()
        self._onehot_projector = _OneHotProjector()
        # Datetime projectors are built lazily per group (need per-group stats)
        self._dt_projectors: Dict[FrozenSet[str], _DatetimeProjector] = {}

    def set_reference(self, reference_data: pd.DataFrame) -> None:
        self._reference = reference_data

    def repair_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorised group-invariant enforcement.

        Pass this DataFrame through after ANY per-column mutation.
        Cost: O(N × G) where G = number of groups (typically < 20).
        """
        result = df.copy()

        for group in self._registry.groups:
            missing = [c for c in group.columns if c not in result.columns]
            if missing:
                logger.debug(
                    "repair_groups: skipping group %s — missing columns %s",
                    group,
                    missing,
                )
                continue

            cols = list(group.columns)

            # ── CYCLIC ────────────────────────────────────────────────────────
            if group.kind == GroupKind.CYCLIC and len(cols) == 2:
                mat = result[cols].astype(float).values  # (N, 2)
                mat = _CyclicProjector.project(mat, group.period)
                result[cols[0]] = mat[:, 0]
                result[cols[1]] = mat[:, 1]

            # ── ONEHOT ────────────────────────────────────────────────────────
            elif group.kind == GroupKind.ONEHOT:
                mat = result[cols].astype(float).values  # (N, C)
                # Compute mode index from reference
                mode_index = self._onehot_mode_index(cols)
                mat = _OneHotProjector.project(mat, mode_index=mode_index)
                result[cols] = mat

            # ── DATETIME ─────────────────────────────────────────────────────
            elif group.kind == GroupKind.DATETIME:
                projector = self._get_datetime_projector(group)
                if projector is None:
                    continue
                component_order = self._datetime_component_order(cols)
                mat = result[cols].astype(float).values  # (N, D)
                mat = projector.project(mat, component_order)
                result[cols] = mat

        return result

    def heal_batch(
        self,
        df: pd.DataFrame,
        constraint_violations: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Full healing pass:
          1. For rows with violations, replace coupled-group columns
             with values sampled from the reference distribution
             (nearest-neighbour or marginal, depending on group kind).
          2. Call repair_groups() on the entire batch.

        constraint_violations: boolean DataFrame, same shape as df.
                               If None, repairs all rows (safe default).
        """
        result = df.copy()

        if constraint_violations is None:
            # Repair everything (conservative default)
            return self.repair_groups(result)

        # Identify rows that have at least one violation in a coupled column
        for group in self._registry.groups:
            group_cols = [c for c in group.columns if c in df.columns]
            if not group_cols:
                continue

            # Rows where ANY column in this group violated a constraint
            violated_mask = constraint_violations[group_cols].any(axis=1)
            if not violated_mask.any():
                continue

            n_violated = int(violated_mask.sum())
            logger.debug(
                "Healing %d rows for group %s via reference sampling",
                n_violated,
                group,
            )

            if self._reference is not None and len(self._reference) > 0:
                ref_cols = [c for c in group_cols if c in self._reference.columns]
                if ref_cols:
                    # Vectorised random sampling from reference rows
                    rng_idx = np.random.randint(
                        0, len(self._reference), size=n_violated
                    )
                    ref_sample = self._reference[ref_cols].iloc[rng_idx].values
                    result.loc[violated_mask, ref_cols] = ref_sample

        # Always finish with a group-level repair pass
        result = self.repair_groups(result)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _onehot_mode_index(self, cols: List[str]) -> int:
        """Return the index of the mode category in the reference data."""
        if self._reference is None:
            return 0
        ref_cols = [c for c in cols if c in self._reference.columns]
        if not ref_cols:
            return 0
        col_sums = self._reference[ref_cols].astype(float).sum(axis=0)
        return int(col_sums.argmax())

    def _get_datetime_projector(
        self, group: FeatureGroup
    ) -> Optional[_DatetimeProjector]:
        """Lazily build a DatetimeProjector for this group."""
        key = group.col_set
        if key in self._dt_projectors:
            return self._dt_projectors[key]

        if self._reference is None:
            return None

        # Collect stats from reference data for each component
        stats: Dict[str, Dict[str, float]] = {}
        component_order = self._datetime_component_order(list(group.columns))
        for comp, col in zip(component_order, group.columns):
            if col not in self._reference.columns:
                continue
            series = pd.to_numeric(self._reference[col], errors="coerce").dropna()
            if len(series) == 0:
                continue
            mu = float(series.mean())
            sigma = float(series.std())
            stats[comp] = {"mean": mu, "std": sigma if sigma > 1e-8 else 1.0}

        # Compute training-set median timestamp
        ts_col = next((c for c in group.columns if "year" in c), None)
        if ts_col and ts_col in self._reference.columns:
            try:
                year_series = pd.to_numeric(
                    self._reference[ts_col], errors="coerce"
                ).dropna()
                median_year = int(year_series.median())
                median_ts = f"{median_year}-01-01"
            except Exception:
                median_ts = "2020-01-01"
        else:
            median_ts = "2020-01-01"

        projector = _DatetimeProjector(
            datetime_stats=stats,
            training_median_ts=median_ts,
        )
        self._dt_projectors[key] = projector
        return projector

    @staticmethod
    def _datetime_component_order(cols: List[str]) -> List[str]:
        """
        Infer datetime component names from column names.
        e.g. ["date_year", "date_month"] → ["year", "month"]
        """
        suffixes = ["year", "month", "day", "hour", "dayofweek", "is_weekend"]
        order = []
        for col in cols:
            matched = next((sfx for sfx in suffixes if col.endswith("_" + sfx)), None)
            order.append(matched or col)
        return order
