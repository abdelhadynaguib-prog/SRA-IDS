"""Run the repository's frozen official-test inference utility.

Execute from the repository root after `pip install -e .`.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_frozen_test import main

if __name__ == "__main__":
    main()
