from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.logging import setup_logging


def test_setup_logging_accepts_none() -> None:
    setup_logging(None)


def main() -> None:
    test_setup_logging_accepts_none()
    print("logging_setup_ok")


if __name__ == "__main__":
    main()
