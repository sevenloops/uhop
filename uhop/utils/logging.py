from __future__ import annotations

import logging
import os
from typing import Optional

try:
    import coloredlogs  # type: ignore
except Exception:  # pragma: no cover
    coloredlogs = None  # type: ignore

from .. import config as _cfg

_LOGGER_CREATED: dict[str, logging.Logger] = {}


def _resolve_level(level_name: Optional[str]) -> int:
    if not level_name:
        return logging.INFO
    name = str(level_name).upper()
    return getattr(logging, name, logging.INFO)


def get_logger(name: str = "uhop") -> logging.Logger:
    if name in _LOGGER_CREATED:
        return _LOGGER_CREATED[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = _resolve_level(os.environ.get("UHOP_LOG_LEVEL") or _cfg.get("UHOP_LOG_LEVEL"))
        logger.setLevel(level)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if coloredlogs is not None:
            coloredlogs.install(level=level, logger=logger, fmt=fmt)  # type: ignore
        else:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
        logger.propagate = False
    _LOGGER_CREATED[name] = logger
    return logger
