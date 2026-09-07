from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


log = logging.getLogger("openaiq")
log.addHandler(logging.NullHandler())


def configure_logging() -> None:
    if any(isinstance(handler, RotatingFileHandler) for handler in log.handlers):
        return
    try:
        folder = Path(__file__).resolve().parent / "logs"
        folder.mkdir(exist_ok=True)
        handler = RotatingFileHandler(
            folder / "runtime.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(threadName)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    except OSError:
        pass
