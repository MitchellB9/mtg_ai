from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], out_path: Path) -> None:
    """
    Input:
        Report dictionary and destination path.

    Logic:
        Writes a JSON report with indentation for human-readable inspection.

    Output:
        JSON file saved to the requested path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Saved report: {out_path}")
