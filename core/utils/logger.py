import logging
import sys
from pathlib import Path


class SafeFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        try:
            msg.encode(sys.stdout.encoding or "utf-8")
            return msg
        except UnicodeEncodeError:
            # fallback: strip non-ascii for console
            return msg.encode("ascii", "ignore").decode("ascii")


def setup_logger(name: str, log_file: str = "logs/system.log") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    base_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console (SAFE)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(SafeFormatter(base_format))

    # File (UTF-8 FULL)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(base_format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def sanitize_module_name(name: str) -> str:
    """
    PyTorch Modules
    """
    return name.replace(".", "_").replace(" ", "_").replace("-", "_")
