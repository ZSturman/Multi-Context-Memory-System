#!/usr/bin/env python3
"""Convenience launcher for the MCMS Streamlit dashboard.

Usage:
    python run_dashboard.py
    # or:
    uv run python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

dashboard_path = Path(__file__).parent / "src" / "multi_context_memory" / "dashboard.py"

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=False,
    )
