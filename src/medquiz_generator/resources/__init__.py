"""Data files shipped inside the wheel.

Kept under the package rather than at the repository root so that an installed
distribution finds them regardless of the working directory: ``examples/`` at the
root only ever worked when the process was launched from a source checkout.

``MQ_EXAMPLES_PATH`` still overrides the bundled file, which is what lets a customer
supply their own few-shot set.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

DEFAULT_EXAMPLES_FILE = "ssm_examples.txt"


def bundled_examples_path(name: str = DEFAULT_EXAMPLES_FILE) -> Path:
    """Absolute path to a bundled examples file (``.txt`` free text or ``.json``)."""
    return Path(str(files(__package__) / "ssm" / name))
