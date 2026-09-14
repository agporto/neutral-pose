#!/usr/bin/env python3
"""Compare two neutral-pose result folders for exact equality.

Used to check that a refactor, a packaging change or a different ``--workers``
setting produced the same result. Version strings in the reports are ignored;
everything else — transforms, posed meshes, patches, landmarks, metrics and
report contents — must match bitwise.

    python scripts/compare_results.py specimen_neutral_serial specimen_neutral_parallel
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _tree(folder: Path, name: str):
    return (
        sorted((p.name, p.read_bytes()) for p in (folder / name).iterdir())
        if (folder / name).is_dir()
        else None
    )


def load(folder: Path) -> dict:
    report = json.loads((folder / "neutral_report.json").read_text(encoding="utf-8"))
    report.get("software", {}).pop("version", None)
    report.get("automatic_inference", {}).pop("software_version", None)
    inference = json.loads((folder / "automatic_inference.json").read_text(encoding="utf-8"))
    inference.pop("software_version", None)
    return {
        "neutral_transforms.npy": np.load(folder / "neutral_transforms.npy"),
        "joint_metrics.csv": list(csv.reader((folder / "joint_metrics.csv").open(encoding="utf-8"))),
        "neutral_report.json": report,
        "automatic_setup.json": json.loads((folder / "automatic_setup.json").read_text(encoding="utf-8")),
        "automatic_inference.json": inference,
        "patches/": _tree(folder, "patches"),
        "LMKs_json/": _tree(folder, "LMKs_json"),
        "meshes_vtp/": _tree(folder, "meshes_vtp"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    a, b = load(args.first), load(args.second)
    identical = True
    for key in a:
        same = np.array_equal(a[key], b[key]) if isinstance(a[key], np.ndarray) else a[key] == b[key]
        identical &= bool(same)
        print(f"{'identical' if same else 'DIFFERENT':9s}  {key}")
    print("RESULT:", "identical" if identical else "different")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
