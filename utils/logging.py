# utils/logging.py
import logging
import sys


def setup_logging(level: str | None = "INFO") -> None:
    """Configure global logging format and level."""
    resolved_level = str(level or "INFO").strip().upper() or "INFO"
    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
