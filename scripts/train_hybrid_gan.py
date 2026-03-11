# scripts/train_hybrid_gan.py
"""
Streamlined Training Orchestrator for Titan Synthetic Data Platform
High-Performance 10M Record Pipeline - Zero Redundancy Architecture

Core Responsibilities:
- CLI argument parsing and validation
- Graceful shutdown handling (SIGINT/SIGTERM)
- Training loop orchestration
- Logging and monitoring

Delegates all business logic to core.integration.hybrid_engine2.HybridSyntheticEngine
Eliminates redundancy by importing shared components from core modules.
"""

from __future__ import annotations
import sys
import os
import signal
import argparse
import logging

# Configure logging
logger = logging.getLogger(__name__)

import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import traceback

import pandas as pd
import torch
import torch.distributed as dist

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ================== Import from Core Modules ==================
from core.integration.hybrid_engine2 import (
    HybridSyntheticEngine,
    HybridEngineConfig,
    CheckpointManager,
)
from core.feature_engineering.deep_type_categorizer import DeepFeatureAnalyzer


# ================== Logging Setup ==================
def setup_logging(output_dir: str, rank: int = 0) -> logging.Logger:
    """
    Setup comprehensive logging with file and console handlers.

    Args:
        output_dir: Directory for log files
        rank: Process rank for distributed training

    Returns:
        Configured logger instance
    """
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("TitanTrainer")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    logger.handlers.clear()

    if rank == 0:
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(output_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# ================== Graceful Shutdown Handler ==================
class GracefulShutdownHandler:
    """
    Handles graceful shutdown on SIGINT/SIGTERM for long training sessions.
    Ensures checkpoints are saved before exit.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.shutdown_requested = False
        self.checkpoint_saved = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        self.logger.warning(f"\n{'='*60}")
        self.logger.warning(f"Received {signal_name} - Initiating graceful shutdown...")
        self.logger.warning(f"{'='*60}")

        if self.shutdown_requested:
            self.logger.error("Force shutdown requested - exiting immediately")
            sys.exit(1)

        self.shutdown_requested = True

    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_requested

    def mark_checkpoint_saved(self):
        """Mark that checkpoint was successfully saved."""
        self.checkpoint_saved = True


# ================== Feature Analysis ==================
def analyze_features(
    data: pd.DataFrame,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze features and generate metadata schema.

    Args:
        data: Input DataFrame
        logger: Logger instance

    Returns:
        Feature metadata dictionary compatible with HybridSyntheticEngine
    """
    logger.info("Analyzing feature schema...")

    analyzer = DeepFeatureAnalyzer()
    feature_metadata = {}

    for column in data.columns:
        try:
            metadata = analyzer.analyze_column(data[column], column)
            deep_type = metadata["deep_type"]

            # Convert to engine-compatible format
            feature_metadata[column] = {
                "type": deep_type,
                "output_dim": (
                    metadata.get("unique_count", metadata.get("cardinality", 1))
                    if deep_type in ["categorical", "ordinal", "binary"]
                    else 1
                ),
                "num_classes": (
                    metadata.get("unique_count", metadata.get("cardinality", 1))
                    if deep_type in ["categorical", "ordinal", "binary"]
                    else None
                ),
                "min": metadata.get("min"),
                "max": metadata.get("max"),
                "cardinality": metadata.get(
                    "unique_count", metadata.get("cardinality")
                ),
            }

            logger.debug(
                f"Feature '{column}': type={deep_type}, "
                f"cardinality={feature_metadata[column].get('cardinality', 'N/A')}"
            )

        except Exception as e:
            logger.error(f"Error analyzing column '{column}': {e}")
            # Fallback to continuous
            feature_metadata[column] = {
                "type": "continuous",
                "output_dim": 1,
            }

    logger.info(f"Feature analysis complete: {len(feature_metadata)} features")
    return feature_metadata


# ================== Data Loading ==================
def load_and_prepare_data(
    data_path: str,
    sample_size: Optional[int],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Load and prepare training data with validation.

    Args:
        data_path: Path to data file
        sample_size: Optional sample size for testing
        logger: Logger instance

    Returns:
        Tuple of (data, feature_metadata)
    """
    logger.info(f"Loading data from: {data_path}")

    # Detect delimiter
    with open(data_path, "r") as f:
        first_line = f.readline()
        delimiter = ";" if ";" in first_line else ","

    # Load data
    data = pd.read_csv(data_path, sep=delimiter)
    logger.info(f"Loaded {len(data):,} records with {len(data.columns)} features")

    # Sample if requested
    if sample_size and sample_size < len(data):
        logger.info(f"Sampling {sample_size:,} records for training")
        data = data.sample(n=sample_size, random_state=42)

    # Analyze features
    feature_metadata = analyze_features(data, logger)

    return data, feature_metadata


# ================== Streamlined Training Orchestrator ==================
class TrainingOrchestrator:
    """
    Streamlined training orchestrator - delegates all logic to HybridSyntheticEngine.

    Responsibilities:
    - Argument parsing
    - Signal handling
    - Epoch loop coordination
    - Logging/monitoring
    """

    def __init__(self, engine_config: HybridEngineConfig):
        self.config = engine_config
        self.logger = setup_logging(engine_config.output_dir, engine_config.rank)
        self.shutdown_handler = GracefulShutdownHandler(self.logger)

        # Engine and checkpoint manager (from core)
        self.engine: Optional[HybridSyntheticEngine] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None

        # Training state
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.patience_counter = 0

    def setup_distributed(self):
        """Setup distributed training if enabled."""
        if self.config.distributed:
            if not dist.is_available():
                raise RuntimeError("Distributed training not available")

            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

            self.logger.info(
                f"Distributed training initialized: "
                f"rank {self.config.rank}/{self.config.world_size}"
            )

    def initialize_engine(
        self,
        data: pd.DataFrame,
        feature_metadata: Dict[str, Dict[str, Any]],
    ):
        """Initialize the hybrid engine and checkpoint manager."""
        self.logger.info("Initializing Hybrid Synthetic Engine...")

        # Initialize engine (from core.integration.hybrid_engine2)
        self.engine = HybridSyntheticEngine(self.config)

        # Load data into engine
        self.logger.info("Loading data into engine...")
        self.engine.load_data(data, feature_metadata)

        # Build models
        self.logger.info("Building models...")
        self.engine.build_models()

        # Initialize checkpoint manager (from core)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
            keep_last_n=self.config.keep_last_n_checkpoints,
            logger=self.logger,
        )

        # Resume from checkpoint if specified
        if self.config.resume_from:
            self.logger.info(f"Resuming from checkpoint: {self.config.resume_from}")
            self.engine.load_checkpoint(self.config.resume_from)

            # Extract epoch from checkpoint name
            try:
                epoch_str = Path(self.config.resume_from).stem.split("_")[-1]
                self.start_epoch = int(epoch_str) + 1
            except:
                self.start_epoch = 0

        self.logger.info("Engine initialization complete")

    def train(self):
        """Execute main training loop - delegates to engine."""
        self.logger.info("=" * 60)
        self.logger.info("Starting Training Pipeline")
        self.logger.info("=" * 60)

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                # Check for shutdown request
                if self.shutdown_handler.should_stop():
                    self.logger.warning("Shutdown requested - saving checkpoint...")
                    self._save_emergency_checkpoint(epoch)
                    break

                # Train epoch (delegated to engine)
                self.logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
                epoch_metrics = self.engine.train_epoch(epoch)

                # Log metrics
                self._log_metrics(epoch, epoch_metrics)

                # Validation (delegated to engine)
                if (epoch + 1) % self.config.validate_interval == 0:
                    val_metrics = self._validate()
                    epoch_metrics.update(val_metrics)

                # ============================================================
                # Checkpointing (Fixed Mapping)
                # ============================================================
                # ============================================================
                # Checkpointing (Safe Positional Execution)
                # ============================================================
                # ============================================================
                # Checkpointing - Final Production-Ready Fix
                # ============================================================
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    is_best = self._is_best_model(epoch_metrics)

                    try:
                        # We pass each value to its dedicated parameter by explicit name
                        # according to the function signature: (epoch, models, optimizers, metrics, is_best)
                        self.checkpoint_manager.save_checkpoint(
                            epoch=epoch + 1,
                            models={
                                "generator": self.engine.generator,
                                "discriminator": self.engine.discriminator,
                                "encoder": self.engine.encoder,
                            },
                            optimizers={
                                "g_opt": self.engine.optimizer_g,
                                "d_opt": self.engine.optimizer_d,
                            },
                            metrics=epoch_metrics,
                            is_best=is_best,
                        )
                    except Exception as e:
                        logger.error(f"Failed to save regular checkpoint: {e}")

                # Early stopping
                if self._should_stop_early():
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Training complete
            self.logger.info("=" * 60)
            self.logger.info("Training Complete")
            self.logger.info("=" * 60)

            # Save final outputs (delegated to engine)
            self._save_final_outputs()

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            self.logger.error(traceback.format_exc())
            self._save_emergency_checkpoint(-1)
            raise

    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch + 1} Metrics: {metric_str}")

    def _validate(self) -> Dict[str, float]:
        """Run validation - delegated to engine."""
        self.logger.info("Running validation...")

        # Generate samples (delegated to engine)
        samples = self.engine.generate_samples(self.config.sample_size_eval)

        # Evaluate quality (delegated to engine)
        metrics = self.engine.evaluate_quality(samples)

        return metrics

    def _is_best_model(self, metrics: Dict[str, float]) -> bool:
        """Check if current model is the best so far."""
        current_metric = metrics.get("loss", float("inf"))

        if current_metric < self.best_metric:
            self.best_metric = current_metric
            self.patience_counter = 0
            return True

        self.patience_counter += 1
        return False

    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria is met."""
        if self.config.early_stopping_patience <= 0:
            return False
        return self.patience_counter >= self.config.early_stopping_patience

    def _save_emergency_checkpoint(self, epoch: int):
        """Save emergency checkpoint on interruption."""
        try:
            emergency_path = (
                Path(self.config.checkpoint_dir) / "emergency_checkpoint.pt"
            )
            self.engine.save_checkpoint(emergency_path)
            self.logger.info(f"Emergency checkpoint saved: {emergency_path}")
            self.shutdown_handler.mark_checkpoint_saved()
        except Exception as e:
            self.logger.error(f"Failed to save emergency checkpoint: {e}")

    def _save_final_outputs(self):
        """Save final model and generate reports - delegated to engine."""
        self.logger.info("Saving final outputs...")

        # Save final model
        final_model_path = Path(self.config.output_dir) / "final_model.pt"
        self.engine.save_checkpoint(final_model_path)

        # Generate samples
        if self.config.save_samples:
            self.logger.info(
                f"Generating {self.config.sample_size_eval} test samples..."
            )
            samples = self.engine.generate_samples(self.config.sample_size_eval)
            samples_path = Path(self.config.output_dir) / "test_samples.csv"
            samples.to_csv(samples_path, index=False)
            self.logger.info(f"Samples saved: {samples_path}")

        # Generate training report (delegated to engine)
        report = self.engine.generate_training_report()
        report_path = Path(self.config.output_dir) / "training_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        self.logger.info(f"Report saved: {report_path}")


# ================== CLI Argument Parser ==================
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with comprehensive validation."""
    parser = argparse.ArgumentParser(
        description="Titan Synthetic Data Platform - Training Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input data file (CSV format)",
    )
    data_group.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample size for quick testing (None = use all data)",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Training batch size",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
        help="Learning rate",
    )
    train_group.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for training",
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension size",
    )
    model_group.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension size",
    )
    model_group.add_argument(
        "--num_attention_blocks",
        type=int,
        default=1,
        help="Number of attention blocks",
    )

    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="output/training",
        help="Directory for outputs and logs",
    )
    output_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints",
    )
    output_group.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    output_group.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Advanced arguments
    advanced_group = parser.add_argument_group("Advanced Options")
    advanced_group.add_argument(
        "--early_stopping_patience",
        type=int,
        default=20,
        help="Early stopping patience (0 = disabled)",
    )
    advanced_group.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training",
    )

    return parser.parse_args()


# ================== Main Entry Point ==================
def main():
    """Main execution entry point - orchestrates the pipeline."""
    print("=" * 60)
    print("Titan Synthetic Data Platform - Training Orchestrator")
    print("High-Performance 10M Record Pipeline")
    print("=" * 60)

    # Parse CLI arguments
    args = parse_arguments()

    try:
        # Build HybridEngineConfig from arguments
        if args.config:
            # Load from YAML with CLI overrides
            engine_config = HybridEngineConfig.from_yaml(
                args.config, **{k: v for k, v in vars(args).items() if v is not None}
            )
        else:
            # Build from CLI arguments directly
            engine_config = HybridEngineConfig(
                data_path=args.data_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_attention_blocks=args.num_attention_blocks,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_interval=args.checkpoint_interval,
                resume_from=args.resume_from,
                early_stopping_patience=args.early_stopping_patience,
                distributed=args.distributed,
            )

        # Initialize orchestrator
        orchestrator = TrainingOrchestrator(engine_config)

        # Setup distributed if needed
        if engine_config.distributed:
            orchestrator.setup_distributed()

        # Load and prepare data
        data, feature_metadata = load_and_prepare_data(
            args.data_path,
            args.sample_size,
            orchestrator.logger,
        )

        # Initialize engine
        orchestrator.initialize_engine(data, feature_metadata)

        # Execute training
        orchestrator.train()

        # Success
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f" Outputs saved to: {engine_config.output_dir}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
        return 130

    except Exception as e:
        print(f"\n\n Training failed with error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
