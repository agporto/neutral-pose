#!/usr/bin/env python3
"""Build synthetic multi-bone specimen ZIPs for equivalence and timing runs.

Marching-cubes vertebrae from the test fixtures are chained with a small
random sagittal bend and given mirrored bilateral landmark pairs so that the
landmark symmetry planes are accepted (use ``pairs=0`` for the surface
fallback path). Example::

    python scripts/make_synthetic_specimen.py out_dir --bones 6 --noise 0.1 --pairs 12
    neutral-pose-auto out_dir/synthetic.zip --workers 1 --out serial
    neutral-pose-auto out_dir/synthetic.zip --workers auto --out parallel
    python scripts/compare_results.py serial parallel
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from neutral_pose import core as n  # noqa: E402
from test_realistic import CENTRUM_LENGTH, GAP, marching_cubes_vertebra  # noqa: E402


def build(
    out_dir: Path,
    name: str,
    count: int,
    noise: float,
    voxel: float,
    bend_deg: float,
    seed: int,
    pairs: int,
    coord: str,
):
    rng = np.random.default_rng(seed)
    source = out_dir / name / "specimen"
    source.mkdir(parents=True, exist_ok=True)
    world = n.rigid(translation=[10.0, -7.0, 3.0])
    # Fixed bone-local seed directions shared by every bone so labels agree specimen-wide.
    seeds = np.random.default_rng(99).normal(size=(pairs, 3))
    if pairs:
        seeds[:, 0] = np.abs(seeds[:, 0]) + 0.3
    labels = []
    for i in range(count):
        if i:
            step = Rotation.from_euler("x", np.deg2rad(rng.uniform(-bend_deg, bend_deg))).as_matrix()
            world = world @ n.rigid(step, [0.0, 0.0, CENTRUM_LENGTH + GAP])
        mesh = marching_cubes_vertebra(f"v{i + 1:02d}", world, voxel=voxel, noise=noise, seed=seed * 100 + i)
        n.write_mesh(source / f"{mesh.name}.ply", mesh, n.rigid(translation=mesh.origin))
        local = (mesh.vertices + mesh.origin - world[:3, 3]) @ world[:3, :3]
        points, labels = list(mesh.landmarks.points), list(mesh.landmarks.labels)
        for k, direction in enumerate(seeds):
            j = int(np.argmin(np.linalg.norm(local - direction * 0.6, axis=1)))
            jm = int(np.argmin(np.linalg.norm(local - local[j] * [-1, 1, 1], axis=1)))
            for idx, side in ((j, "R"), (jm, "L")):
                points.append(mesh.vertices[idx] + mesh.origin)
                labels.append(f"P{k + 1}{side}")
        n.write_landmarks(
            source / "LMKs" / f"{mesh.name}.mrk.json",
            np.array(points),
            [f"F-{k + 1}" for k in range(len(labels))],
            coord,
        )
    archive = out_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in source.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(source.parent))
    print(f"built {archive} ({count} bones, {len(labels)} landmarks per bone)")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--name", default="synthetic")
    parser.add_argument("--bones", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--bend-deg", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pairs", type=int, default=12, help="Mirrored landmark pairs per bone (0 = surface fallback)."
    )
    parser.add_argument("--coordinates", default="LPS", choices=["LPS", "RAS"])
    args = parser.parse_args()
    build(
        args.out_dir,
        args.name,
        args.bones,
        args.noise,
        args.voxel,
        args.bend_deg,
        args.seed,
        args.pairs,
        args.coordinates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
