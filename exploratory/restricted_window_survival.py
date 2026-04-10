"""
Deprecated. Use src/analysis/restricted_window_survival.py instead.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.restricted_window_survival import run_restricted_window_claiming

if __name__ == "__main__":
    print("Redirecting to src/analysis/restricted_window_survival.py")
    run_restricted_window_claiming()
