#!/usr/bin/env python3
"""Function-level AST comparison between a previous flat-module release and the package.

Formatting, comments and docstrings are ignored (they are not part of the AST).
Each top-level function or class in the package is looked up by name in the
release it descended from, and reported as identical, changed (with a unified
diff of ``ast.unparse`` output) or new.

Example::

    python scripts/refactor_audit.py path/to/neutral_pose_1.6.2 --out validation/refactor_audit_1.7.0.json
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
from pathlib import Path

# Package module -> flat modules it descended from.
ANCESTRY = {
    "core.py": ["neutral_pose.py"],
    "landmarks.py": ["neutral_pose_landmarks.py"],
    "anatomy.py": ["neutral_pose_anatomy.py"],
    "support.py": ["neutral_pose_support.py"],
    "surfaces.py": ["neutral_pose_auto.py"],
    "discovery.py": ["neutral_pose_auto.py"],
    "auto.py": ["neutral_pose_auto.py"],
    "contacts.py": ["neutral_pose_contacts.py"],
    "recovery.py": ["neutral_pose_contacts.py"],
}


def definitions(source: str) -> dict[str, ast.AST]:
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):  # docstrings are not compared
                if isinstance(
                    child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
                ) and ast.get_docstring(child):
                    child.body = child.body[1:] or [ast.Pass()]
            out[node.name] = node
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("previous", type=Path, help="Folder containing the flat neutral_pose*.py modules.")
    parser.add_argument(
        "--package", type=Path, default=Path(__file__).resolve().parents[1] / "src" / "neutral_pose"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = {"previous": str(args.previous), "modules": {}}
    for module, ancestors in ANCESTRY.items():
        before = {}
        for name in ancestors:
            before.update(definitions((args.previous / name).read_text(encoding="utf-8")))
        after = definitions((args.package / module).read_text(encoding="utf-8"))
        entry = {"identical": [], "changed": {}, "new": []}
        for name, node in after.items():
            if name not in before:
                entry["new"].append(name)
            elif ast.dump(before[name]) == ast.dump(node):
                entry["identical"].append(name)
            else:
                diff = difflib.unified_diff(
                    ast.unparse(before[name]).splitlines(), ast.unparse(node).splitlines(), lineterm="", n=0
                )
                entry["changed"][name] = [line for line in diff if not line.startswith(("---", "+++", "@@"))]
        report["modules"][module] = entry
        print(
            f"{module:14s} identical {len(entry['identical']):3d}  changed {len(entry['changed']):2d}  new {len(entry['new'])}"
        )
        for name, lines in entry["changed"].items():
            print(f"  {name}:")
            for line in lines:
                print(f"    {line}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
