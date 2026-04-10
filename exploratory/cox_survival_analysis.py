"""
Deprecated exploratory entry point.

Use `exploratory/restricted_window_survival.py` for the current claiming
survival analysis. This wrapper is kept only so older shell history and notes
still resolve to the cleaned specification.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exploratory.restricted_window_survival import run_restricted_window_claiming


if __name__ == "__main__":
    print("Deprecated: forwarding to exploratory/restricted_window_survival.py")
    run_restricted_window_claiming()
