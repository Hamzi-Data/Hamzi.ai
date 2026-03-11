from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import time
import gc
import json
import yaml
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from scipy import stats
from scipy.spatial.distance import jensenshannon


@dataclass
class HybridEngineConfig:
    """Complete configuration for Hybrid Synthetic Engine."""

    # Project
    project_name: str = "titan_synthetic"
    version: str = "5.0"

    # Paths
    data_path: str = ""
    output_dir: str = "output/training"
    checkpoint_dir: str = "checkpoints"
    save_dir: Path = field(default_factory=lambda: Path("models/hybrid_engine"))

    # Data
    sample_size: int = 10000
    validation_split: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 0.0002
    generator_lr: float = 0.0002
    discriminator_lr: float = 0.0002
    encoder_lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    data_dim: int = 0
    continuous_dim: int = 0
    categorical_cardinalities: List[int] = field(default_factory=list)

    noise_dim: int = 256
    ms_loss_weight: float = 2.5
    # ---------------------------

    # Model Architecture
    hidden_dim: int = 256
    embedding_dim: int = 64
    num_attention_blocks: int = 1
    attention_heads: int = 4
    generator_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    discriminator_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.2

    # Training Stabilization
    n_critic: int = 2
    apply_gradient_penalty: bool = True
    lambda_gp: float = 10.0
    gradient_clip_norm: float = 1.0
    use_spectral_norm: bool = True

    # Learning Rate Scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"
    min_lr: float = 1e-6

    # Mixed Precision
    use_mixed_precision: bool = False
    use_mixed_precision_generation: bool = False

    # Generation
    generation_batch_size: int = 5000
    save_samples: bool = True
    sample_size_eval: int = 1000

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_frequency: int = 10
    keep_last_n_checkpoints: int = 5
    resume_from: Optional[str] = None

    # Monitoring
    log_interval: int = 10
    log_frequency: int = 10
    validate_interval: int = 50
    diversity_check_frequency: int = 10
    mode_collapse_threshold: float = 0.85

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Distributed
    distributed: bool = False
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    dist_backend: str = "nccl"

    # Memory Management
    max_memory_gb: float = 16.0

    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60

    # Early Stopping
    early_stopping_patience: int = 20

    def __post_init__(self):
        """Post-initialization validation and type correction."""

        for attr in ["noise_dim", "epochs", "sample_size", "batch_size"]:
            val = getattr(self, attr)
            if isinstance(val, (list, tuple)):
                setattr(self, attr, int(val[0]))

        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.generation_batch_size > 10000:
            self.generation_batch_size = 10000
            print(f"generation_batch_size capped at 10000 for stability.")

        if self.noise_dim <= 0:
            raise ValueError(f"noise_dim must be positive. Got {self.noise_dim}")

    @classmethod
    def from_yaml(
        cls, config_path: Union[str, Path], **overrides
    ) -> HybridEngineConfig:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        config_dict.update(overrides)
        return cls(**config_dict)

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> HybridEngineConfig:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, save_path: Union[str, Path]):
        save_path = Path(save_path)
        if save_path.suffix in [".yaml", ".yml"]:
            with open(save_path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif save_path.suffix == ".json":
            with open(save_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {save_path.suffix}")
