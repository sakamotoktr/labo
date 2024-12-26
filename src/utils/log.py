from logging import Logger, DEBUG, INFO, CRITICAL
from logging.config import dictConfig
from sys import stdout
from pathlib import Path
from typing import Union


configured_verbosity = DEBUG if settings.debug else INFO


def _initialize_log_storage() -> "Path":
    """establish required directory structure for logging output

    Returns: the storage location Path object
    """
    storage_location = Path(settings.labo_dir / "storage" / "system_events.log")
    storage_location.parent.mkdir(parents=True, exist_ok=True)
    storage_location.touch(exist_ok=True)
    return storage_location


RUNTIME_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s"},
        "minimal": {"format": "%(name)s | %(levelname)s | %(message)s"},
    },
    "handlers": {
        "terminal": {
            "level": configured_verbosity,
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "minimal",
        },
        "persistent": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": _initialize_log_storage(),
            "maxBytes": 10485760,
            "backupCount": 3,
            "formatter": "verbose",
        },
    },
    "root": {
        "level": DEBUG if settings.debug else INFO,
        "handlers": ["terminal", "persistent"],
    },
    "loggers": {
        "LABO": {
            "level": DEBUG if settings.debug else INFO,
            "propagate": True,
        },
        "uvicorn": {
            "level": "CRITICAL",
            "handlers": ["terminal"],
            "propagate": False,
        },
    },
}


def initialize_event_tracker(namespace: Union[str, None] = None) -> "Logger":
    """create or retrieve a configured logging instance
    Args:
        namespace: optional scope identifier for the logger
    """
    dictConfig(RUNTIME_CONFIG)
    primary_tracker = Logger.manager.getLogger("LABO")
    if namespace:
        return primary_tracker.getChild(namespace)
    return primary_tracker
