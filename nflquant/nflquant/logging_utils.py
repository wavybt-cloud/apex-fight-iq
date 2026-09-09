"""Shared logging setup."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "nflquant", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger
