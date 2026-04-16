from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config import get_settings


@lru_cache(maxsize=1)
def _configure_root() -> None:
    s = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(s.log_level.upper())


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
