"""
core/generative/training/scheduler.py
Productivity learning rate schedulers for GAN systems with adaptive and cyclical mechanisms, scalable.

Features:
- AdaptiveGANScheduler: adapts learning rate based on loss direction, G/D balance, and gradient oscillation.
- CyclicalLRScheduler: stable cyclical scheduling (triangular/triangular2/exp_range) integrated with PyTorch.
- Clear contracts, diagnostic messages, and safe parameters without fragile assumptions on param_group names.

Rights: Microsoft AI (Production re-engineering)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Protocol

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# ========= Errors =========


class SchedulerError(ValueError):
    """Custom exception for scheduler errors with clear diagnostic messages."""

    pass


# ========= Contracts =========


class LossSnapshot(Protocol):
    """Protocol for recent batch loss values (supports flexibility in measurement sources)."""

    @property
    def g_loss(self) -> Optional[float]: ...
    @property
    def d_loss(self) -> Optional[float]: ...
    @property
    def gradient_norm(self) -> Optional[float]: ...


@dataclass(frozen=True)
class AdaptiveConfig:
    """
    Configuration for the adaptive GAN scheduler.

    Attributes:
        total_training_steps: Total steps for safe cosine decay application.
        window_divergence: Divergence evaluation window (number of recent values).
        window_stagnation: Stagnation evaluation window (number of recent values).
        divergence_threshold_ratio: Increase ratio considered as divergence (e.g. 1.4 means 40% increase).
        stagnation_var_threshold: Variance threshold considered as stagnation.
        lr_decay_on_divergence: LR reduction factor on divergence.
        lr_boost_on_stagnation: LR increase factor on stagnation.
        balance_ratio_upper: Loss ratio threshold to consider generator weak (g/d > upper).
        balance_ratio_lower: Loss ratio threshold to consider discriminator weak (g/d < lower).
        balance_adjust_factor: Adjustment factor for balance correction.
        cosine_use_base_lrs: Whether to use optimizer's base_lrs or pre-set values.
    """

    total_training_steps: int = 100_000
    window_divergence: int = 20
    window_stagnation: int = 20
    divergence_threshold_ratio: float = 1.4
    stagnation_var_threshold: float = 1e-3
    lr_decay_on_divergence: float = 0.5
    lr_boost_on_stagnation: float = 1.1
    balance_ratio_upper: float = 2.0
    balance_ratio_lower: float = 0.5
    balance_adjust_factor: float = 0.05
    cosine_use_base_lrs: bool = True


# ========= Adaptive scheduler =========


class AdaptiveGANScheduler:
    """
    Adaptive learning rate scheduler for GANs that depends on:
    - Loss direction (divergence/stagnation)
    - Generator/Discriminator balance via g_loss/d_loss ratio
    - Cosine decay over total training steps
    - Gradient history (optional) to detect gradient explosions

    IO:
    - step(loss_snapshot: LossSnapshot) -> None
    - get_current_lrs() -> Dict[int, float]

    Works with a single Optimizer (can be used separately for G and D),
    and makes no assumptions about param_group names; use separate instances
    if you want independent G/D scheduling.
    """

    def __init__(
        self, optimizer: Optimizer, config: Optional[AdaptiveConfig] = None
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise SchedulerError("optimizer must be a torch.optim.Optimizer.")

        self.optimizer: Optimizer = optimizer
        self.config: AdaptiveConfig = config or AdaptiveConfig()

        # History
        self.g_history: List[float] = []
        self.d_history: List[float] = []
        self.grad_history: List[float] = []

        # State
        self.step_count: int = 0
        # Snapshot of base LR for each group
        self.base_lrs: List[float] = [
            group.get("lr", 1e-3) for group in optimizer.param_groups
        ]

    def step(self, loss_snapshot: LossSnapshot) -> None:
        """
        Updates learning rates based on current losses and applies adaptive logic.
        """
        self.step_count += 1
        # Update history
        if loss_snapshot.g_loss is not None:
            self._append_with_cap(
                self.g_history,
                float(loss_snapshot.g_loss),
                cap=max(self.config.window_divergence, self.config.window_stagnation),
            )
        if loss_snapshot.d_loss is not None:
            self._append_with_cap(
                self.d_history,
                float(loss_snapshot.d_loss),
                cap=max(self.config.window_divergence, self.config.window_stagnation),
            )
        if loss_snapshot.gradient_norm is not None:
            self._append_with_cap(
                self.grad_history, float(loss_snapshot.gradient_norm), cap=100
            )

        # Divergence
        if self._check_divergence():
            self._scale_all_lrs(self.config.lr_decay_on_divergence)

        # Stagnation
        if self._check_stagnation():
            self._scale_all_lrs(self.config.lr_boost_on_stagnation)

        # Cosine decay
        self._apply_cosine_decay()

        # G/D balance via loss ratio (requires both)
        if self._can_balance():
            self._apply_balance_adjustment()

        # Gradient explosion protection: if recent grad mean is too high, reduce LR further
        if self.grad_history:
            grad_mean = float(
                np.mean(self.grad_history[-min(10, len(self.grad_history)) :])
            )
            if grad_mean > 10.0:  # conservative threshold; can be tuned
                self._scale_all_lrs(0.9)

    def _append_with_cap(self, arr: List[float], val: float, cap: int) -> None:
        arr.append(val)
        if len(arr) > cap:
            arr.pop(0)

    def _check_divergence(self) -> bool:
        """
        Check for divergence: recent window average vs older window, against a ratio.
        """
        w = self.config.window_divergence
        if len(self.g_history) < w or len(self.d_history) < w:
            return False
        half = w // 2
        g_recent = float(np.mean(self.g_history[-half:]))
        d_recent = float(np.mean(self.d_history[-half:]))
        g_old = float(np.mean(self.g_history[-w:-half]))
        d_old = float(np.mean(self.d_history[-w:-half]))
        return (g_recent > g_old * self.config.divergence_threshold_ratio) or (
            d_recent > d_old * self.config.divergence_threshold_ratio
        )

    def _check_stagnation(self) -> bool:
        """
        Check for stagnation: very low variance in the recent loss window.
        """
        w = self.config.window_stagnation
        if len(self.g_history) < w or len(self.d_history) < w:
            return False
        g_var = float(np.var(self.g_history[-w:]))
        d_var = float(np.var(self.d_history[-w:]))
        return (g_var < self.config.stagnation_var_threshold) and (
            d_var < self.config.stagnation_var_threshold
        )

    def _scale_all_lrs(self, factor: float) -> None:
        """Safely scales LR for all param_groups by a given factor."""
        if factor <= 0:
            raise SchedulerError("LR adjustment factor must be > 0.")
        for group in self.optimizer.param_groups:
            current_lr = float(group.get("lr", 1e-3))
            group["lr"] = max(1e-8, current_lr * factor)

    def _apply_cosine_decay(self) -> None:
        """
        Apply cosine decay based on total training steps.
        """
        total_steps = max(1, int(self.config.total_training_steps))
        current_step = min(self.step_count, total_steps)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * current_step / total_steps))
        base_lrs = (
            self.base_lrs
            if self.config.cosine_use_base_lrs
            else [float(pg.get("lr", 1e-3)) for pg in self.optimizer.param_groups]
        )
        for group, base_lr in zip(self.optimizer.param_groups, base_lrs):
            group["lr"] = max(1e-8, float(base_lr) * float(cosine_decay))

    def _can_balance(self) -> bool:
        """Checks if loss ratio for G/D balance can be calculated."""
        return bool(self.g_history) and bool(self.d_history)

    def _apply_balance_adjustment(self) -> None:
        """
        Adjusts LR to reduce imbalance between G/D using g/d ratio.
        Applies a small uniform adjustment to all groups;
        if separate G/D scheduling is desired, use separate scheduler instances.
        """
        g_avg = float(np.mean(self.g_history[-min(10, len(self.g_history)) :]))
        d_avg = float(np.mean(self.d_history[-min(10, len(self.d_history)) :]))
        ratio = g_avg / (d_avg + 1e-8)
        adjust = self.config.balance_adjust_factor
        if ratio > self.config.balance_ratio_upper:
            # Generator is weaker (higher loss): slight LR increase
            self._scale_all_lrs(1.0 + adjust)
        elif ratio < self.config.balance_ratio_lower:
            # Discriminator is weaker: slight LR increase
            self._scale_all_lrs(1.0 + adjust)

    def get_current_lrs(self) -> Dict[int, float]:
        """
        Returns current learning rates for each parameter group.
        """
        return {
            i: float(pg.get("lr", 0.0))
            for i, pg in enumerate(self.optimizer.param_groups)
        }


# ========= Cyclical LR scheduler =========


class CyclicalLRScheduler(_LRScheduler):
    """
    Cyclical Learning Rate (CLR) scheduler supporting patterns:
    - triangular: linear rise/fall
    - triangular2: triangular with per-cycle halving decay
    - exp_range: triangular with exponential decay per step

    Args:
        optimizer: PyTorch optimizer.
        step_size: Half cycle length (number of steps).
        min_lr: Minimum LR.
        max_lr: Maximum LR.
        mode: 'triangular' | 'triangular2' | 'exp_range'.
        gamma: Exponential decay factor for exp_range.
        last_epoch: Last epoch (default -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int = 2000,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        mode: str = "triangular",
        gamma: float = 0.99,
        last_epoch: int = -1,
    ) -> None:
        self.step_size = int(step_size)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.mode = mode
        self.gamma = float(gamma)

        if self.step_size <= 0:
            raise SchedulerError("step_size must be > 0.")
        if not (self.min_lr > 0 and self.max_lr > self.min_lr):
            raise SchedulerError("Invalid LR values: max_lr must be > min_lr > 0.")
        if mode not in ("triangular", "triangular2", "exp_range"):
            raise SchedulerError(f"Unsupported CLR mode: {mode}")

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns current LR list for each param_group according to the cyclical formula.
        """
        cycle = np.floor(1 + self.last_epoch / (2.0 * self.step_size))
        x = np.abs(self.last_epoch / float(self.step_size) - 2.0 * cycle + 1.0)
        base = max(0.0, 1.0 - x)

        if self.mode == "triangular":
            scale = 1.0
        elif self.mode == "triangular2":
            scale = 1.0 / (2.0 ** (cycle - 1.0))
        else:  # exp_range
            scale = self.gamma**self.last_epoch

        lr_span = self.max_lr - self.min_lr
        current = self.min_lr + lr_span * base * scale

        # Distribute same LR across all groups (can be customized if needed)
        return [float(current) for _ in self.base_lrs]
