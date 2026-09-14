#!/usr/bin/env python3
"""Reproduce numerical recovery checks and write a synthetic demonstration."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from neutral_pose import core as n

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from scipy.spatial.transform import Rotation

from test_neutral_pose import fixture_chain, fixture_pair
from test_realistic import articulated_truth, marching_cubes_vertebra, realistic_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("validation"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    records = []
    rng = np.random.default_rng(845)
    for i, (angle, scale) in enumerate(
        [(0.0, 0.1), (0.0, 1.0), (0.0, 10.0), (14.0, 1.0), (8.0, 1.0), (0.0, 1.0)]
    ):
        rotation = rng.uniform(-22, 22, 3) if i != 5 else np.zeros(3)
        translation = rng.uniform(-0.15, 0.15, 3) if i != 5 else np.zeros(3)
        a, b, cfg, truth = fixture_pair(np.deg2rad(angle), scale, rotation, translation)
        options = n.Options()  # Validate the shipped defaults, including sensitivity.
        matrices, poses, report, joints = n.fit_column([a, b], cfg, options)
        delta = n.pose_difference(matrices[1], truth, scale)
        records.append(
            {
                "case": f"pair_{i + 1}",
                "reference_bend_deg": angle,
                "scale_mm": scale,
                "applied_rotation_xyz_deg": rotation.tolist(),
                "applied_translation_fraction": translation.tolist(),
                "recovery_rotation_error_deg": float(np.rad2deg(np.linalg.norm(delta[:3]))),
                "recovery_translation_error_fraction": float(np.linalg.norm(delta[3:])),
                "max_sampled_penetration_mm": report["joints"][0]["metrics"]["max_sampled_penetration"],
                "triangle_intersection_detected": report["joints"][0]["metrics"]["intersection_detected"],
                "output_status": report["status"],
            }
        )
    # Noisy marching-cubes ball-and-socket fixture: unrelated tessellations,
    # field noise in fractions of the voxel pitch, landmark-grown patches.
    noisy = []
    # Declared physical allowance for this synthetic protocol, fixed before
    # recovery and independent of mesh edge lengths or observed pose errors.
    noisy_options = n.Options(noise_floor_mm=0.02)
    for noise in (0.0, 0.15, 0.25, 0.4):
        for trial in range(3):
            rng_t = np.random.default_rng(100 + trial)
            truth_pair = articulated_truth(2)
            pert = n.rigid(
                Rotation.from_euler("xyz", rng_t.uniform(-12, 12, 3), degrees=True).as_matrix(),
                rng_t.uniform(-0.12, 0.12, 3),
            )
            a = marching_cubes_vertebra(
                "v01", truth_pair[0], voxel=0.04, noise=noise, seed=10 * trial + 1, offset=0.371
            )
            b = marching_cubes_vertebra(
                "v02", pert @ truth_pair[1], voxel=0.046, noise=noise, seed=10 * trial + 2, offset=0.618
            )
            joint = n.Joint(a, b, realistic_config(), noisy_options, 0)
            H_true = n.rigid(translation=-a.origin) @ n.inverse(pert) @ n.rigid(translation=b.origin)
            seed_error = n.pose_difference(joint.landmark_seed(), H_true, 1.0)
            matrices, poses, report, joints = n.fit_column([a, b], realistic_config(), noisy_options)
            delta = n.pose_difference(matrices[1], n.inverse(pert), 1.0)
            j = report["joints"][0]
            noisy.append(
                {
                    "case": f"noisy_{noise:.2f}_{trial + 1}",
                    "field_noise_voxels": noise,
                    "triangles": [len(a.faces), len(b.faces)],
                    "noise_floor_mm": j["effective_tolerances"]["noise_floor_mm"],
                    "noise_floor_source": j["effective_tolerances"]["noise_floor_source"],
                    "seed_rotation_error_deg": float(np.rad2deg(np.linalg.norm(seed_error[:3]))),
                    "seed_translation_error_fraction": float(np.linalg.norm(seed_error[3:])),
                    "recovery_rotation_error_deg": float(np.rad2deg(np.linalg.norm(delta[:3]))),
                    "recovery_translation_error_fraction": float(np.linalg.norm(delta[3:])),
                    "spherical_centrum_used": next(
                        p["spherical_centrum"] for p in j["metrics"]["patches"] if p["kind"] == "centrum"
                    ),
                    "output_status": report["status"],
                    "review_reasons": j["review_reasons"],
                }
            )
    meshes, config, truth, truth_poses = fixture_chain(count=4, angle_deg=8.0)
    matrices, poses, report, joints = n.fit_column(meshes, config, n.Options())
    chain = []
    for m, M, expected in zip(meshes, matrices, truth):
        delta = n.pose_difference(M, expected, 1.0)
        chain.append(
            {
                "mesh": m.name,
                "recovery_rotation_error_deg": float(np.rad2deg(np.linalg.norm(delta[:3]))),
                "recovery_translation_error_fraction": float(np.linalg.norm(delta[3:])),
            }
        )
    demo = args.out / "synthetic_curved_column"
    demo.mkdir(exist_ok=True)
    inputs = demo / "input"
    inputs.mkdir(exist_ok=True)
    for m in meshes:
        n.write_mesh(inputs / f"{m.name}.ply", m, n.rigid(translation=m.origin))
        n.write_landmarks(
            inputs / "LMKs" / f"{m.name}.mrk.json", m.landmarks.points, m.landmarks.labels, "RAS"
        )
    (inputs / "neutral_config.json").write_text(json.dumps(config, indent=2))
    output = demo / "fitted"
    if output.exists():
        import shutil

        shutil.rmtree(output)
    n.save_result(output, meshes, matrices, poses, report, joints)
    data = {
        "fixture_description": "Synthetic closed wedges with three specified articular patches, plus noisy marching-cubes ball-and-socket bones; none are biological specimens.",
        "pair_cases": records,
        "noisy_marching_cubes_cases": noisy,
        "curved_chain": chain,
        "chain_global_refinement": report["global_refinement"],
        "chain_nonadjacent_contacts": report["nonadjacent_contacts"],
        "runtime_seconds": time.perf_counter() - start,
        "software": report["software"],
        "scope": "Tests numerical recovery under known surface and spacing assumptions. Real vertebrae and cartilage assumptions remain unvalidated.",
    }
    (args.out / "recovery_metrics.json").write_text(json.dumps(data, indent=2, allow_nan=False))
    lines = [
        "# Numerical validation",
        "",
        data["fixture_description"],
        "",
        "| Case | Reference bend | Scale (mm) | Rotation error (deg) | Translation error / local scale | Intersections |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['case']} | {r['reference_bend_deg']:.1f} | {r['scale_mm']:.1f} | {r['recovery_rotation_error_deg']:.6g} | {r['recovery_translation_error_fraction']:.6g} | {r['triangle_intersection_detected']} |"
        )
    lines += [
        "",
        "In every wedge case above the anatomical (Kabsch) seed already equals the reference pose, so those rows",
        "verify that the objective's fixed point is the truth, not that the optimizer recovers it.",
        "",
        "Noisy marching-cubes ball-and-socket pairs (unrelated tessellations, landmark-grown patches, three random",
        "perturbations per noise level; field noise in fractions of the voxel pitch, voxel = 4% of centrum length).",
        "These cases use an explicit 0.02 mm surface-error allowance, fixed across all noise levels;",
        "it is a declared test assumption, not an estimate from triangulation or a biological recommendation.",
        "",
        "| Case | Noise (voxels) | Seed rot. error (deg) | Recovered rot. error (deg) | Recovered trans. error / scale | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in noisy:
        lines.append(
            f"| {r['case']} | {r['field_noise_voxels']:.2f} | {r['seed_rotation_error_deg']:.3g} | {r['recovery_rotation_error_deg']:.3g} | {r['recovery_translation_error_fraction']:.3g} | {r['output_status']} |"
        )
    lines += [
        "",
        "Default component preservation retains segmentation islands in the noisy fixtures. These cases carry",
        "`disconnected_mesh_components` even when pose error is small; see recovery_metrics.json for every",
        "case's review reasons, including any penetration, coverage or ambiguity flags. They are not certified fits.",
        "",
        "Rotation error at nonzero noise is dominated by yaw about the column axis, which a ball-and-socket centrum",
        "leaves free and near-planar facets constrain only weakly; the `multiple_similarly_scoring_poses` and",
        "`pose_depends_on_patch_centering` review reasons report when this is the case.",
        "",
        "Four-bone curved-column recovery (8 degrees per joint):",
        "",
        "| Bone | Rotation error (deg) | Translation error / local scale |",
        "|---|---:|---:|",
    ]
    for r in chain:
        lines.append(
            f"| {r['mesh']} | {r['recovery_rotation_error_deg']:.6g} | {r['recovery_translation_error_fraction']:.6g} |"
        )
    lines += [
        "",
        f"Runtime for this validation: {data['runtime_seconds']:.2f} seconds on the available environment.",
        "",
        "The curved planar fixtures have finite patch-boundary differences after bending; translation errors therefore include that deliberate geometric mismatch.",
        "",
        data["scope"],
        "",
        "See test_results.txt for the additional regression checks.",
    ]
    (args.out / "VALIDATION.md").write_text("\n".join(lines) + "\n")
    # Exact mesh projection, not generated anatomy.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for k, mesh in enumerate(meshes):
        before = mesh.vertices + mesh.origin
        after = n.transform(poses[k], mesh.vertices)
        for panel, V in zip(ax, [before, after]):
            panel.add_collection(
                PolyCollection(
                    V[mesh.faces][:, :, [1, 2]], facecolor=palette[k], edgecolor="none", alpha=0.35
                )
            )
            panel.autoscale_view()
    for panel, title in zip(ax, ["Perturbed input", "Fitted osteological reference"]):
        panel.set_title(title)
        panel.set_aspect("equal")
        panel.set_xlabel("Y (mm)")
        panel.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Z (mm)")
    fig.suptitle("Synthetic curved-column recovery — 8° per joint", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out / "curved_column_recovery.png", dpi=180)
    plt.close(fig)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
