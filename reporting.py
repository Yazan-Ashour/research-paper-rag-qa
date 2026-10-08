import json
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


def save_report(report, filename):
    """Write the report to the results directory as JSON and return its path."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / filename
    stamped = {"created_at": datetime.now().isoformat(timespec="seconds"), **report}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(stamped, file, ensure_ascii=False, indent=2)
    return path