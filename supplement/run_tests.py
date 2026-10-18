"""Run the standalone synthetic dashboard tests without unrelated application imports."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DESELECT = (
    "tests/test_bi_data.py::test_aspect_names_match_business_dashboard",
    "tests/test_bi_dashboard.py::test_lead_page_dispatch_through_ui_main",
)

if __name__ == "__main__":
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "tests/test_bi_data.py", "tests/test_bi_dashboard.py"]
    command += [f"--deselect={nodeid}" for nodeid in DESELECT]
    command += sys.argv[1:]
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))
