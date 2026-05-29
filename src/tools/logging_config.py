# logger.py
import logging
import logging.config
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOG_DIR / "pipeline.log"),
            "formatter": "standard",
            "level": "INFO",
            "encoding": "utf-8",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}


def setup_logging():
    logging.config.dictConfig(LOGGING)
