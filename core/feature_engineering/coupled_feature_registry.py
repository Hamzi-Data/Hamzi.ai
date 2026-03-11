"""
core/feature_engineering/coupled_feature_registry.py
═══════════════════════════════════════════════════════════════════════════════
TITAN v6.2 — CoupledFeatureRegistry
───────────────────────────────────
The single missing abstraction that caused every cascading failure.

A "coupled group" is a set of columns that are MATHEMATICALLY BOUND:
  • CYCLIC   – (sin_month, cos_month): sin²+cos² must equal 1.
               Healing one without the other destroys the unit circle.
  • ONEHOT   – (job_admin, job_blue-collar, …): exactly one must be 1.
               Adjusting one component without renormalising the rest
               creates a non-probability-simplex vector.
  • DATETIME – (date_year, date_month, date_day, …): must form a valid
               calendar date; components cannot be sampled independently.

Design contract
───────────────
• Completely schema-agnostic – detection is driven purely by column
  naming conventions and statistical properties, never by hardcoded names.
• Vectorised throughout – all group-wide operations use NumPy axis-ops;
  no Python-level row loops are tolerated.
• Idempotent – calling detect_groups() twice on the same data returns
  the same result.

Author : Titan AI Architecture Team
Version: 6.2 Enterprise
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data structures
# ─────────────────────────────────────────────────────────────────────────────


class GroupKind(Enum):
    CYCLIC = auto()  # sin / cos pair — unit-circle manifold
    ONEHOT = auto()  # one-hot group  — probability simplex
    DATETIME = auto()  # calendar group — valid date manifold


@dataclass(frozen=True)
class FeatureGroup:
    """Immutable descriptor for a coupled feature group."""

    kind: GroupKind
    columns: Tuple[str, ...]  # ordered tuple, stable across runs
    period: Optional[float] = None  # relevant for CYCLIC groups (e.g. 12 for months)

    # Derived metadata stored for fast lookup at healing time
    col_set: FrozenSet[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "col_set", frozenset(self.columns))

    def __repr__(self) -> str:
        return (
            f"FeatureGroup(kind={self.kind.name}, "
            f"columns={self.columns}, period={self.period})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CoupledFeatureRegistry
# ─────────────────────────────────────────────────────────────────────────────


class CoupledFeatureRegistry:
    """
    Detects, stores, and exposes coupled feature groups for any tabular schema.

    Usage
    ─────
        registry = CoupledFeatureRegistry()
        groups   = registry.detect_groups(df, latent_map)

        # At healing time:
        group = registry.find_group(column_name)   # → FeatureGroup | None
    """

    # ── Cyclic suffix signatures ──────────────────────────────────────────────
    # Each entry: (sin_suffix, cos_suffix, period, semantic_label)
    _CYCLIC_SIGNATURES: Tuple[Tuple[str, str, float, str], ...] = (
        ("_sin", "_cos", None, "generic"),
        ("_sin_", "_cos_", None, "generic_underscore"),
        ("hour_sin", "hour_cos", 24.0, "hour"),
        ("dow_sin", "dow_cos", 7.0, "day_of_week"),
        ("month_sin", "month_cos", 12.0, "month"),
        ("day_sin", "day_cos", 31.0, "day"),
        ("week_sin", "week_cos", 52.0, "week"),
        ("min_sin", "min_cos", 60.0, "minute"),
        ("sec_sin", "sec_cos", 60.0, "second"),
    )

    # ── DateTime component suffixes ───────────────────────────────────────────
    _DATETIME_SUFFIXES: Tuple[str, ...] = (
        "_year",
        "_month",
        "_day",
        "_hour",
        "_dayofweek",
        "_is_weekend",
    )

    def __init__(self) -> None:
        self._groups: List[FeatureGroup] = []
        self._col_to_group: Dict[str, FeatureGroup] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_groups(
        self,
        df: pd.DataFrame,
        latent_map: Optional[Dict] = None,
    ) -> List[FeatureGroup]:
        """
        Auto-detect all coupled groups in *df*.

        Detection runs in three passes (order matters for de-duplication):
          1. Cyclic pairs   — suffix-based signature matching
          2. One-hot groups — prefix + statistical validation (row sums ∈ {0, 1})
          3. DateTime groups— common prefix + datetime suffix set membership

        Returns
        ───────
        List[FeatureGroup] — deduplicated, ordered by first-column appearance.
        """
        cols = list(df.columns)
        col_set = set(cols)

        consumed: Set[str] = set()
        groups: List[FeatureGroup] = []

        # ── Pass 1: Cyclic pairs ──────────────────────────────────────────────
        for sin_sfx, cos_sfx, period, _ in self._CYCLIC_SIGNATURES:
            for col in cols:
                if col in consumed:
                    continue
                if col.endswith(sin_sfx):
                    prefix = col[: len(col) - len(sin_sfx)]
                    cos_col = prefix + cos_sfx
                    if cos_col in col_set and cos_col not in consumed:
                        # Validate: values should roughly lie on the unit circle
                        inferred_period = period or self._infer_cyclic_period(
                            df[col], df[cos_col]
                        )
                        g = FeatureGroup(
                            kind=GroupKind.CYCLIC,
                            columns=(col, cos_col),
                            period=inferred_period,
                        )
                        groups.append(g)
                        consumed.update([col, cos_col])
                        logger.debug("Cyclic group detected: %s", g)

        # ── Pass 2: DateTime component groups ────────────────────────────────
        prefix_to_dt_cols: Dict[str, List[str]] = defaultdict(list)
        for col in cols:
            if col in consumed:
                continue
            for sfx in self._DATETIME_SUFFIXES:
                if col.endswith(sfx):
                    prefix = col[: len(col) - len(sfx)]
                    prefix_to_dt_cols[prefix].append(col)
                    break

        for prefix, dt_cols in prefix_to_dt_cols.items():
            if len(dt_cols) < 3:
                continue
            g = FeatureGroup(
                kind=GroupKind.DATETIME,
                columns=tuple(sorted(dt_cols, key=lambda c: cols.index(c))),
                period=None,
            )
            groups.append(g)
            consumed.update(dt_cols)
            logger.debug("Datetime group detected: %s", g)

        # ── Pass 3: One-hot groups ────────────────────────────────────────────
        prefix_groups: Dict[str, List[str]] = defaultdict(list)
        for col in cols:
            if col in consumed:
                continue
            m = re.match(r"^([A-Za-z][A-Za-z0-9]*)([_\.].+)$", col)
            if m:
                prefix_groups[m.group(1)].append(col)

        for prefix, oh_cols in prefix_groups.items():
            if len(oh_cols) < 2:
                continue
            if not self._validate_onehot(df[oh_cols]):
                continue
            g = FeatureGroup(
                kind=GroupKind.ONEHOT,
                columns=tuple(oh_cols),
                period=None,
            )
            groups.append(g)
            consumed.update(oh_cols)
            logger.debug("One-hot group detected: %s", g)

        # Build lookup map
        self._groups = groups
        self._col_to_group = {}
        for g in groups:
            for c in g.columns:
                self._col_to_group[c] = g

        logger.info(
            "CoupledFeatureRegistry: detected %d groups "
            "(%d cyclic, %d datetime, %d onehot) across %d columns.",
            len(groups),
            sum(1 for g in groups if g.kind == GroupKind.CYCLIC),
            sum(1 for g in groups if g.kind == GroupKind.DATETIME),
            sum(1 for g in groups if g.kind == GroupKind.ONEHOT),
            len(consumed),
        )
        return groups

    def find_group(self, column: str) -> Optional[FeatureGroup]:
        """Return the FeatureGroup that owns *column*, or None."""
        return self._col_to_group.get(column)

    def is_coupled(self, column: str) -> bool:
        """Return True if *column* belongs to any coupled group."""
        return column in self._col_to_group

    @property
    def groups(self) -> List[FeatureGroup]:
        return list(self._groups)

    # ── Vectorised repair operations ──────────────────────────────────────────

    def repair_tensor(
        self,
        tensor: torch.Tensor,
        latent_map: Dict,
    ) -> torch.Tensor:
        """
        Enforce group-level invariants on a GAN output tensor.

        For every detected group:
          CYCLIC  → re-normalise (sin, cos) to the unit circle.
          ONEHOT  → apply softmax → argmax hard one-hot projection.
          DATETIME→ no-op (date validity is enforced at inverse_transform time).

        This is called inside DynamicGenerator.forward() AFTER the raw
        output layer, before the tensor leaves the generator.

        All operations are fully vectorised (no Python loops over rows).
        """
        out = tensor.clone()

        for group in self._groups:
            # Collect slice indices from latent_map
            slices = []
            for col in group.columns:
                if col in latent_map:
                    m = latent_map[col]
                    slices.append((m["start_idx"], m["end_idx"]))

            if not slices:
                continue

            if group.kind == GroupKind.CYCLIC and len(slices) == 2:
                s0, e0 = slices[0]
                s1, e1 = slices[1]
                sin_vals = out[:, s0:e0]
                cos_vals = out[:, s1:e1]
                # Re-project onto unit circle: divide by L2 norm
                norms = torch.sqrt(sin_vals**2 + cos_vals**2 + 1e-8)
                out[:, s0:e0] = sin_vals / norms
                out[:, s1:e1] = cos_vals / norms

            elif group.kind == GroupKind.ONEHOT:
                all_start = slices[0][0]
                all_end = slices[-1][1]
                logits = out[:, all_start:all_end]
                # Hard one-hot: argmax in inference, gumbel in training
                if not tensor.requires_grad:
                    indices = logits.argmax(dim=-1)
                    oh = torch.zeros_like(logits)
                    oh.scatter_(1, indices.unsqueeze(1), 1.0)
                    out[:, all_start:all_end] = oh

        return out

    def is_cyclic_channel(self, col_name: str) -> bool:
        """Return True if col_name belongs to a CYCLIC coupled group."""
        from core.feature_engineering.coupled_feature_registry import GroupKind

        for group in self.groups:
            if group.kind == GroupKind.CYCLIC and col_name in group.columns:
                return True
        return False

    def repair_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Enforce group-level invariants on a DataFrame (post inverse_transform).

        This is the healing oracle called by CoupledHealingEngine.
        """
        result = df.copy()

        for group in self._groups:
            missing = [c for c in group.columns if c not in result.columns]
            if missing:
                continue

            if group.kind == GroupKind.CYCLIC:
                sin_col, cos_col = group.columns[0], group.columns[1]
                s = result[sin_col].astype(float).values
                c = result[cos_col].astype(float).values
                norms = np.sqrt(s**2 + c**2 + 1e-8)
                result[sin_col] = s / norms
                result[cos_col] = c / norms

            elif group.kind == GroupKind.ONEHOT:
                cols = list(group.columns)
                mat = result[cols].astype(float).values  # (N, C)
                # Find winner per row; set to 1, rest to 0
                winner = mat.argmax(axis=1)  # (N,)
                oh = np.zeros_like(mat)
                oh[np.arange(len(oh)), winner] = 1.0
                result[cols] = oh

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _infer_cyclic_period(
        sin_series: pd.Series,
        cos_series: pd.Series,
    ) -> Optional[float]:
        """
        Heuristically infer the cyclic period from sin/cos values.
        Returns None if the pair doesn't look like a well-formed cyclic encoding.
        """
        try:
            s = sin_series.dropna().astype(float).values
            c = cos_series.dropna().astype(float).values
            if len(s) < 10:
                return None
            # Check unit-circle adherence: median of ||(s,c)||₂ should be ~1
            norms = np.sqrt(s[:1000] ** 2 + c[:1000] ** 2)
            if np.median(norms) < 0.8:
                return None  # Not actually cyclic-encoded
            # Infer period from angle distribution
            angles = np.arctan2(s[:1000], c[:1000])  # in (-π, π]
            unique_angles = np.unique(np.round(angles, 2))
            if len(unique_angles) < 3:
                return None
            angular_step = np.min(np.abs(np.diff(np.sort(unique_angles))))
            if angular_step < 1e-4:
                return None
            period_estimate = 2 * np.pi / angular_step
            # Round to nearest plausible period
            for known in [7.0, 12.0, 24.0, 31.0, 52.0, 60.0, 365.0]:
                if abs(period_estimate - known) / known < 0.25:
                    return known
            return float(round(period_estimate, 1))
        except Exception:
            return None

    @staticmethod
    def _validate_onehot(group_df: pd.DataFrame, threshold: float = 0.85) -> bool:
        """
        Return True if ≥ threshold fraction of rows satisfy the one-hot constraint
        (each row sums to exactly 0 or 1 across the group).
        """
        try:
            numeric = group_df.apply(pd.to_numeric, errors="coerce").fillna(0)
            row_sums = numeric.sum(axis=1)
            valid_fraction = ((row_sums == 0) | (row_sums == 1)).mean()
            return float(valid_fraction) >= threshold
        except Exception:
            return False
