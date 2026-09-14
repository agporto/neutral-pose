#!/usr/bin/env python3
"""Estimate an osteological articulation directly from a specimen folder or ZIP.

Example: neutral-pose-auto specimen.zip
Meshes must retain an approximate articulation. Physical cartilage thickness
cannot be recovered from bone surfaces alone; observed clearances are used as
spacing evidence and their dependence on the supplied pose is recorded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import anatomy, contacts, recovery, support
from . import core as n
from . import landmarks as landmark_anatomy
from .discovery import infer_joint
from .parallel import parallel_map, resolve_workers
from .version import __version__

LOG = logging.getLogger(__name__)
MESH_SUFFIXES = {".ply", ".vtp", ".stl", ".obj"}


def specimen_folders(root):
    """Find actual mesh folders; analysis tables and landmark subfolders are ignored."""
    root = Path(root)
    folders = [root] + sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: str(p))
    return [
        p
        for p in folders
        if not any(part.startswith(".") or part == "__MACOSX" for part in p.relative_to(root).parts)
        and not (p / "neutral_report.json").exists()
        and sum(f.is_file() and f.suffix.lower() in MESH_SUFFIXES for f in p.iterdir()) >= 2
    ]


def safe_extract(archive, destination):
    """Extract a ZIP, refusing path traversal and symlinks."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination):
                raise ValueError("ZIP contains a path outside its specimen folder.")
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("ZIP symlinks are not supported.")
        z.extractall(destination)


def input_files(folder):
    """Ordered mesh paths in a specimen folder and the matching landmark file (or ``None``) for each."""
    paths = sorted(
        (p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in MESH_SUFFIXES),
        key=n.numeric_key,
    )
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError("Multiple mesh formats have the same basename; retain one mesh per vertebra.")
    if len(paths) < 2:
        raise ValueError("At least two meshes are needed in a specimen folder.")
    landmarks = []
    for p in paths:
        candidates = [
            f
            for parent in (p.parent / "LMKs", p.parent / "LMKs_json", p.parent)
            if parent.is_dir()
            for f in parent.iterdir()
            if f.name.casefold() in {(p.stem + ".fcsv").casefold(), (p.stem + ".mrk.json").casefold()}
        ]
        if len(candidates) > 1:
            raise ValueError(f"Multiple landmark files match {p.name}.")
        landmarks.append(candidates[0] if candidates else None)
    return paths, landmarks


def mesh_units(paths):
    """Read declared units from PLY headers; only millimetres are accepted."""
    evidence = []
    for p in paths:
        if p.suffix.lower() != ".ply":
            continue
        with p.open("rb") as f:
            header = f.read(8192).split(b"end_header", 1)[0].decode("ascii", errors="ignore")
        hit = re.search(r"\bUnit\s*:\s*([A-Za-z]+)", header, re.I)
        if hit:
            unit = hit.group(1).lower()
            if unit not in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
                raise ValueError(f"{p.name}: declares {unit}; export meshes and landmarks in millimeters.")
            evidence.append({"file": p.name, "unit": "mm", "source": "mesh_header"})
    return {
        "unit": "mm",
        "source": "mesh_headers" if evidence else "Slicer_millimeter_convention_assumed",
        "files_with_unit_metadata": len(evidence),
        "evidence": evidence,
    }


def infer_coordinates(meshes, landmark_paths):
    """Test RAS and LPS against surfaces; never infer anatomy from a numbered label."""
    scores, converted = {}, {}
    per_mesh = []
    for coordinate in ("RAS", "LPS"):
        normalized, loaded = [], []
        for mesh, path in zip(meshes, landmark_paths):
            lm = n.read_landmarks(path, coordinate) if path else None
            loaded.append(lm)
            if lm is not None:
                distances = np.abs(mesh.signed_distance(lm.points - mesh.origin))
                normalized.append(float(np.median(distances) / mesh.scale))
                per_mesh.append(
                    {
                        "mesh": mesh.name,
                        "candidate": coordinate,
                        "median_surface_distance_mm": float(np.median(distances)),
                        "p90_surface_distance_mm": float(np.quantile(distances, 0.9)),
                    }
                )
        scores[coordinate] = float(np.median(normalized)) if normalized else None
        converted[coordinate] = loaded
    available = any(p is not None for p in landmark_paths)
    if available:
        best = min(scores, key=scores.get)
        worst = max(scores, key=scores.get)
        decisive = scores[best] < 0.05 and scores[worst] > max(5 * scores[best], 0.03)
        if not decisive:
            # Prefer the explicit coordinate header when geometry cannot resolve
            # a symmetric object. This is recorded as an assumption.
            best = next(lm.coordinate_system for lm in converted["RAS"] if lm is not None)
        if scores[best] > 0.15:
            raise ValueError("Landmarks do not match meshes under either RAS or LPS; check paired exports.")
    else:
        best, decisive = "RAS", False
    for mesh, lm in zip(meshes, converted[best]):
        mesh.landmarks = lm
    return {
        "coordinate_system": best,
        "source": "landmark_surface_match"
        if decisive
        else "landmark_header_assumed"
        if available
        else "RAS_assumed_no_landmarks",
        "geometrically_resolved": decisive,
        "median_normalized_distances": scores,
        "per_mesh": per_mesh,
    }


def _discover_joint_task(meshes, index):
    """Contact discovery for one adjacent pair (independent of all other pairs)."""
    a, b = meshes[index], meshes[index + 1]
    try:
        return infer_joint(a, b, index)
    except ValueError as error:
        # Delay a failed proximity inference so a specimen consensus can
        # recover it. Failure remains explicit if that consensus is absent.
        definition = {"a": a.name, "b": b.name, "patches": []}
        evidence = {
            "a": a.name,
            "b": b.name,
            "patches": [],
            "local_scale_mm": math.sqrt(a.scale * b.scale),
            "discovery_error": str(error),
            "bilateral_contact_selection": None,
        }
        return definition, evidence


def prepare_specimen(folder, workers=1):
    """Load a specimen and infer coordinates, symmetry planes, contacts and spacing.

    Returns ``(meshes, config, options, inference)`` ready for :func:`neutral_pose.core.fit_column`.
    ``workers`` > 1 discovers contacts for adjacent pairs in parallel processes.
    """
    paths, landmark_paths = input_files(folder)
    units = mesh_units(paths)
    meshes = []
    for i, p in enumerate(paths):
        LOG.info("Reading mesh %d/%d: %s", i + 1, len(paths), p.name)
        meshes.append(n.Mesh.read(p, reject_self_intersections=False))
    coordinates = infer_coordinates(meshes, landmark_paths)
    landmark_symmetry = landmark_anatomy.prepare_landmark_planes(meshes)
    origins = np.array([m.origin for m in meshes])
    nearest = cKDTree(origins).query(origins, k=min(4, len(meshes)))[1]
    neighbor_fraction = float(np.mean([i + 1 in nearest[i] for i in range(len(meshes) - 1)]))
    options = n.Options(
        samples=96,
        target_samples=1200,
        collision_samples=192,
        starts=3,
        max_nfev=80,
        global_max_nfev=40,
        center_weight=0.3,
        normal_weight=0.08,
    )
    config = {
        "mesh_coordinate_system": coordinates["coordinate_system"],
        "length_unit": "mm",
        "order": [p.name for p in paths],
        "automatic_inference": True,
        "require_anatomy": False,
        "joints": [],
        "options": n.asdict(options),
    }
    inference = {
        "software_version": n.__version__,
        "specimen": Path(folder).name,
        "landmark_symmetry": landmark_symmetry,
        "coordinates": coordinates,
        "units": units,
        "ordering": {
            "source": "natural_filename_order",
            "mesh_order": config["order"],
            "consecutive_pairs_among_three_nearest_centers_fraction": neighbor_fraction,
        },
        "landmark_files": [p.name if p else None for p in landmark_paths],
        "joints": [],
        "assumptions": [
            "Meshes retain an approximate articulation and filename order follows the column.",
            "Spacing is estimated from the supplied pose, not an independent cartilage measurement.",
            "Contact identities are geometric inferences, not user-annotated anatomy.",
            "Surface-model residuals are an automatic error allowance, not measured segmentation noise.",
        ],
    }
    # Discovery reuses a mesh's surface-symmetry plane once a neighbouring joint
    # has computed and cached it, so adjacent pairs are sequentially dependent
    # unless every plane was already fixed by an accepted landmark plane.
    planes_fixed = all(getattr(m, "_automatic_bilateral_plane", None) is not None for m in meshes)
    discovery_workers = workers if planes_fixed else 1
    if not planes_fixed and resolve_workers(workers) > 1:
        LOG.info("Discovering contacts serially: surface symmetry planes are seeded by neighbouring joints")
    for i, (definition, evidence) in enumerate(
        parallel_map(_discover_joint_task, len(meshes) - 1, meshes, discovery_workers)
    ):
        LOG.info(
            "Discovered contacts and spacing %d/%d: %s / %s",
            i + 1,
            len(meshes) - 1,
            meshes[i].name,
            meshes[i + 1].name,
        )
        config["joints"].append(definition)
        inference["joints"].append(evidence)
    anatomy.pool_bilateral_spacing(config, inference)
    recovery.complete_contacts(meshes, config, inference, options)
    config["neutral_frames"] = anatomy.infer_frames(meshes, config)
    inference["anatomical_frames"] = config["neutral_frames"]
    inference["assumptions"].extend(
        [
            "Landmark reflection pairs are inferred by specimen-wide agreement; "
            "label numbers do not supply anatomical roles.",
            "Accepted landmark planes are checked against surfaces without surface-driven refinement; "
            "fallbacks are recorded per bone.",
            "Neutral lateral bending and axial twist mean coincident inferred midsagittal planes.",
            "Sagittal rotation and in-plane translation are fitted to the joint surfaces.",
            "Left and right facet spacing evidence is pooled equally to remove input bending bias.",
        ]
    )
    inference["assumptions"].extend(
        [
            "Missing contacts require specimen-local landmark agreement and opposing-surface re-discovery.",
            "Isolated high facet gaps use a robust complete-joint median; "
            "recovered contacts use specimen spacing statistics.",
            "Uncertainty is retained as a review flag; "
            "selection compares required contacts and measurable penetration.",
        ]
    )
    inference["input_mesh_quality"] = [
        {
            "mesh": m.name,
            "components": m.connected_components,
            "boundary_edges": m.boundary_edges,
            "nonmanifold_edges": m.nonmanifold_edges,
            "self_intersection": m.self_intersection,
            "signed_distance_reliable": m.closed,
        }
        for m in meshes
    ]
    inference["inputs"] = [
        {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in paths + list(filter(None, landmark_paths))
    ]
    return meshes, config, options, inference


def preview(directory, meshes, poses, frames=None):
    """Write a geometrically faithful overview in two orthogonal projection planes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    originals = np.vstack([m.vertices + m.origin for m in meshes])
    center = originals.mean(axis=0)
    if frames is not None:
        first = frames[meshes[0].name]
        axes = np.asarray(first["axes"])[:, [2, 0, 1]].T
        center = np.asarray(first["center"])
    else:
        _, _, axes = np.linalg.svd(
            originals[:: max(1, len(originals) // 10000)] - center, full_matrices=False
        )
    fig, panels = plt.subplots(2, 2, figsize=(14, 5.5))
    palette = ["#4477aa", "#ee9944", "#228877", "#aa4499"]
    for i, (mesh, pose) in enumerate(zip(meshes, poses)):
        for col, V in enumerate([mesh.vertices + mesh.origin, n.transform(pose, mesh.vertices)]):
            q = (V - center) @ axes.T
            for row, dims in enumerate(([0, 1], [0, 2])):
                panels[row, col].add_collection(
                    PolyCollection(
                        q[mesh.faces][:, :, dims],
                        facecolor=palette[i % len(palette)],
                        edgecolor="none",
                        alpha=0.32,
                        rasterized=True,
                    )
                )
    if frames is not None:
        before = np.array([frames[m.name]["center"] for m in meshes])
        after = np.array([n.transform(T, p - m.origin) for p, m, T in zip(before, meshes, poses)])
        for col, points in enumerate((before, after)):
            q = (points - center) @ axes.T
            for row in range(2):
                panels[row, col].plot(
                    q[:, 0], q[:, row + 1], color="#28384a", lw=0.8, marker=".", markersize=2.5
                )
    for row in range(2):
        for col in range(2):
            ax = panels[row, col]
            ax.autoscale_view()
            ax.set_aspect("equal")
            ax.set_xlabel(
                "Posterior direction (mm)" if frames is not None else "Longitudinal projection (mm)"
            )
            ax.set_ylabel(
                (["Lateral position (mm)", "Dorsal position (mm)"][row])
                if frames is not None
                else f"Transverse projection {row + 1} (mm)"
            )
            ax.spines[["top", "right"]].set_visible(False)
        xlims = [ax.get_xlim() for ax in panels[row]]
        ylims = [ax.get_ylim() for ax in panels[row]]
        for ax in panels[row]:
            ax.set_xlim(min(x[0] for x in xlims), max(x[1] for x in xlims))
            ax.set_ylim(min(y[0] for y in ylims), max(y[1] for y in ylims))
    for row in range(2):
        view = (["Dorsal view", "Lateral view"][row] + " — ") if frames is not None else ""
        panels[row, 0].set_title(view + "input arrangement")
        panels[row, 1].set_title(view + "neutral estimate")
    fig.suptitle(
        "Anatomical neutral pose" if frames is not None else "Automatic articulation from supplied surfaces",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(Path(directory) / "articulation_preview.png", dpi=180)
    plt.close(fig)


def _select_joint_task(shared, index):
    """Pose selection for one joint (independent of all other joints).

    Returns the chosen relative pose ``H``, the updated report entry and the
    selection decision. The caller chains the poses and stores the entries.
    """
    joints, poses, entries = shared
    i, joint = index, joints[index]
    contact_policy = joint._fitting_config().get("automatic_contact_policy") == contacts.POLICY
    proposed = n.inverse(poses[i]) @ poses[i + 1]
    entry = entries[i]
    original_metrics = joint.metrics(joint.initial, final=True)
    reference, reference_metrics = joint.initial, original_metrics
    reference_fit = None
    if joint.neutral_plane is not None:
        reference_fit = n.optimize_joint(joint, joint.initial, fixed_sagittal=True)
        reference = reference_fit.transform
        reference_metrics = joint.metrics(reference, final=True)
    proposed_metrics = entry["metrics"]
    if contact_policy:
        if not contacts.complete({"patches": joint.specs}):
            raise ValueError(f"{joint.name}: incomplete anatomical neutral reference")
        repaired, repair = support.refine_dense_feasibility(joint, proposed)
        entry["dense_refinement"] = repair
        if repair["accepted"]:
            entry["pre_dense_fit"] = {
                "transform_b_local_to_a_local": proposed.tolist(),
                "metrics": proposed_metrics,
            }
            proposed = repaired
            proposed_metrics = joint.metrics(proposed, final=True)
            if not repair["converged"]:
                entry["review_reasons"].append("dense_optimizer_not_converged")
        entry["metrics"] = proposed_metrics
        entry["transform_b_local_to_a_local"] = proposed.tolist()
        entry["final_score"] = float(np.sum(joint.residual(proposed) ** 2))
    unc = entry["uncertainty"]
    reasons = []
    if any(unc.get(key) for key in ("ambiguous", "sensitive", "centering_dependent", "poorly_identified")):
        reasons.append("automatic_fit_not_identified")
    if not entry["pairwise_converged"] or "pairwise_search_boundary_reached" in entry["review_reasons"]:
        reasons.append("automatic_fit_not_converged_within_bounds")
    before, after = (
        reference_metrics["max_sampled_penetration"],
        proposed_metrics["max_sampled_penetration"],
    )
    allowance = max(joint.noise_fraction * joint.scale, 1e-6 * joint.scale)
    if before is not None and after is not None and after > before + allowance:
        reasons.append("automatic_fit_increased_penetration")
    before_coverage = np.mean(
        [
            side["coverage_fraction"]
            for p in reference_metrics["patches"]
            for side in (p["a_to_b"], p["b_to_a"])
        ]
    )
    after_coverage = np.mean(
        [
            side["coverage_fraction"]
            for p in proposed_metrics["patches"]
            for side in (p["a_to_b"], p["b_to_a"])
        ]
    )
    if after_coverage < before_coverage - 0.05:
        reasons.append("automatic_fit_reduced_contact_coverage")
    if contact_policy:
        before_coverage = support.contact_support(reference_metrics)["mean"]
        after_coverage = support.contact_support(proposed_metrics)["mean"]
        converged = repair.get("converged", entry["pairwise_converged"])
        boundary = repair.get("boundary", "pairwise_search_boundary_reached" in entry["review_reasons"])
        reasons = support.selection_reasons(
            reference_metrics,
            proposed_metrics,
            joint.penetration_tolerance * joint.scale,
            allowance,
            converged,
            boundary,
        )
        if reasons and before is not None and before > joint.penetration_tolerance * joint.scale:
            revised_reference, reference_repair = support.refine_dense_feasibility(
                joint, reference, fixed_sagittal=True
            )
            entry["reference_dense_refinement"] = reference_repair
            if reference_repair["accepted"]:
                reference = revised_reference
                reference_metrics = joint.metrics(reference, final=True)
                before = reference_metrics["max_sampled_penetration"]
                before_coverage = support.contact_support(reference_metrics)["mean"]
                reasons = support.selection_reasons(
                    reference_metrics,
                    proposed_metrics,
                    joint.penetration_tolerance * joint.scale,
                    allowance,
                    converged,
                    boundary,
                )
    H = reference if reasons else proposed
    if joint.neutral_plane is not None:
        diagnostics = joint.neutral_plane.diagnostics(H)
        if (
            diagnostics["lateral_axis_misalignment_deg"] > 1e-6
            or abs(diagnostics["lateral_center_offset_mm"]) > 1e-8 * joint.scale
        ):
            raise ValueError("A proposed automatic pose violates its anatomical neutral constraint.")
    decision = {
        "a": joint.a.name,
        "b": joint.b.name,
        "choice": ("neutral_sagittal_reference" if joint.neutral_plane is not None else "input_retained")
        if reasons
        else "fitted_pose",
        "reasons": reasons,
        "input_max_sampled_penetration_mm": original_metrics["max_sampled_penetration"],
        "reference_max_sampled_penetration_mm": before,
        "proposed_max_sampled_penetration_mm": after,
        "input_mean_coverage": float(
            np.mean(
                [
                    side["coverage_fraction"]
                    for p in original_metrics["patches"]
                    for side in (p["a_to_b"], p["b_to_a"])
                ]
            )
        ),
        "reference_mean_coverage": float(before_coverage),
        "proposed_mean_coverage": float(after_coverage),
    }
    if reference_fit is not None:
        decision["reference"] = "input_sagittal_angle_with_surface_fitted_in_plane_translation"
        decision["reference_fit_converged"] = reference_fit.converged
        decision["reference_fit_boundary"] = reference_fit.boundary
        entry["neutral_reference_metrics"] = reference_metrics
    entry["automatic_selection"] = decision
    entry["input_metrics"] = original_metrics
    entry["proposed_fit"] = {
        "transform_b_local_to_a_local": proposed.tolist(),
        "metrics": proposed_metrics,
        "review_reasons": entry["review_reasons"].copy(),
        "final_score": entry["final_score"],
    }
    if reasons:
        entry["transform_b_local_to_a_local"] = H.tolist()
        entry["metrics"] = reference_metrics
        entry["final_score"] = float(np.sum(joint.residual(H) ** 2))
        entry["review_reasons"] = [
            r
            for r in entry["review_reasons"]
            if r
            not in {
                "penetration_exceeds_tolerance",
                "triangle_intersections_require_review",
                "poor_surface_fit",
                "insufficient_surface_coverage",
                "anchor_offset_exceeds_tolerance",
                "global_refinement_changed_pairwise_reference",
            }
        ]
        metrics = reference_metrics
        depth = metrics["max_sampled_penetration"]
        if depth is not None and depth > joint.penetration_tolerance * joint.scale:
            entry["review_reasons"].append("penetration_exceeds_tolerance")
        if metrics["intersection_detected"]:
            if (
                not metrics["signed_distance_reliable"]
                or depth is None
                or depth > joint.penetration_tolerance * joint.scale
            ):
                entry["review_reasons"].append("triangle_intersections_require_review")
            else:
                metrics["contact_within_noise_tolerance"] = True
        for p in metrics["patches"]:
            for side in (p["a_to_b"], p["b_to_a"]):
                if side["rms_gap_error_fraction"] > joint.acceptable_surface_error:
                    entry["review_reasons"].append("poor_surface_fit")
                if side["coverage_fraction"] < joint.options.minimum_coverage:
                    entry["review_reasons"].append("insufficient_surface_coverage")
            if (
                not p["spherical_centrum"]
                and p["anchor_tangential_offset_mm"]
                > joint.options.centering_tolerance_fraction * joint.scale
            ):
                entry["review_reasons"].append("anchor_offset_exceeds_tolerance")
        entry["review_reasons"].append(
            "neutral_sagittal_reference_due_to_uncertain_fit"
            if joint.neutral_plane is not None
            else "input_pose_retained_due_to_uncertain_automatic_fit"
        )
        if reference_fit is not None and (not reference_fit.converged or reference_fit.boundary):
            entry["review_reasons"].append("neutral_reference_optimizer_requires_review")
        entry["review_reasons"] = sorted(set(entry["review_reasons"]))
        entry["status"] = "needs_review"
    entry["uncertainty"]["reference"] = "proposed_fit_before_automatic_selection"
    if contact_policy:
        decision["policy"] = contacts.POLICY
        decision["reference_contact_support"] = support.contact_support(reference_metrics)
        decision["proposed_contact_support"] = support.contact_support(proposed_metrics)
        entry["uncertainty"]["reference"] = "pairwise_fit_before_global_refinement_and_selection"
        support.refresh_geometric_review(entry, joint)
    if joint.neutral_plane is not None:
        entry["neutral_plane"] = joint.neutral_plane.diagnostics(H)
        entry["anatomical_orientation_b_in_a"] = (
            joint.neutral_plane.fa.T @ H[:3, :3] @ joint.neutral_plane.fb
        ).tolist()
    return H, entry, decision


def select_supported_poses(meshes, poses, report, joints, workers=1):
    """Select stable poses without releasing an anatomical neutral constraint.

    A constrained fallback retains only the input sagittal angle and refits
    in-plane translation. Legacy unconstrained calls retain their input pose.
    ``workers`` > 1 evaluates joints in parallel processes with identical results.
    """
    result = copy.deepcopy(report)
    selected = [n.rigid(translation=meshes[0].origin)]
    decisions = []
    for i, (H, entry, decision) in enumerate(
        parallel_map(_select_joint_task, len(joints), (joints, poses, result["joints"]), workers)
    ):
        LOG.info("Selected joint %d/%d: %s", i + 1, len(joints), joints[i].name)
        selected.append(selected[-1] @ H)
        result["joints"][i] = entry
        decisions.append(decision)
    # Recheck nonadjacent contacts on the selected column, not discarded poses.
    nonadjacent_contacts = []
    boxes = [n.pair_aabb(m, T) for m, T in zip(meshes, selected)]
    for i in range(len(meshes)):
        for j in range(i + 2, len(meshes)):
            if not n.aabb_overlap(boxes[i], boxes[j]):
                continue
            H = n.inverse(selected[i]) @ selected[j]
            hit = n.intersects(meshes[i], meshes[j], H)
            depth = 0.0
            verified = meshes[i].closed and meshes[j].closed
            for target, other, T in [(meshes[i], meshes[j], H), (meshes[j], meshes[i], n.inverse(H))]:
                if target.closed:
                    P = np.vstack([other.vertices, other.centers, other.sample(2048, seed=71).points])
                    depth = max(depth, float(target.penetration(n.transform(T, P)).max()))
            if hit or depth > 0 or not verified:
                nonadjacent_contacts.append(
                    {
                        "a": meshes[i].name,
                        "b": meshes[j].name,
                        "intersection_detected": hit,
                        "max_sampled_penetration": depth,
                        "signed_distance_reliable": verified,
                    }
                )
    result["nonadjacent_contacts"] = nonadjacent_contacts
    result["review_reasons"] = [
        r for r in result["review_reasons"] if r != "nonadjacent_meshes_require_review"
    ]
    if nonadjacent_contacts:
        result["review_reasons"].append("nonadjacent_meshes_require_review")
    result["automatic_pose_selection"] = {
        "decisions": decisions,
        "input_joints_retained": sum(d["choice"] == "input_retained" for d in decisions),
        "sagittal_reference_joints": sum(d["choice"] == "neutral_sagittal_reference" for d in decisions),
        "note": "Constrained fallbacks retain the neutral plane and report sagittal uncertainty.",
    }
    result["global_refinement"]["applies_to"] = "proposed_column_before_automatic_selection"
    matrices = np.stack([T @ n.rigid(translation=-m.origin) for m, T in zip(meshes, selected)])
    for M in matrices:
        n.validate_rigid(M)
    return matrices, selected, result


def process_specimen(folder, output, overwrite=False, workers=1):
    """Run the automatic pipeline on one specimen folder and write results atomically to ``output``."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if output == folder or folder.is_relative_to(output):
        raise ValueError("Output cannot replace an input folder or one of its parents.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")
    meshes, config, options, inference = prepare_specimen(folder, workers)
    # Keep references portable when the source was unpacked into a temporary
    # directory. Input hashes and original basenames remain in the receipt.
    for mesh in meshes:
        mesh.source = Path(mesh.source).name
    matrices, poses, report, joints = n.fit_column(meshes, config, options, workers)
    LOG.info("Selecting supported poses and checking the selected column")
    matrices, poses, report = select_supported_poses(meshes, poses, report, joints, workers)
    report["automatic_inference"] = inference
    report["inputs"] = inference["inputs"]
    report["definition"] = (
        "Automatic osteological neutral estimate with coincident inferred midsagittal planes, "
        "surface-fitted sagittal articulation and robust specimen spacing estimates."
    )
    inference["neutrality"] = anatomy.column_diagnostics(meshes, poses, config["neutral_frames"])
    report["anatomical_neutrality"] = inference["neutrality"]
    changes = []
    for i, joint in enumerate(joints):
        delta = n.pose_difference(n.inverse(poses[i]) @ poses[i + 1], joint.initial, joint.scale)
        changes.append(
            {
                "a": joint.a.name,
                "b": joint.b.name,
                "relative_rotation_change_deg": float(np.rad2deg(np.linalg.norm(delta[:3]))),
                "relative_translation_change_mm": float(joint.scale * np.linalg.norm(delta[3:])),
            }
        )
    inference["pose_changes"] = changes
    inference["pose_selection"] = report["automatic_pose_selection"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="neutral-auto-", dir=output.parent))
    backup = None
    try:
        n.save_result(temporary, meshes, matrices, poses, report, joints)
        (temporary / "automatic_setup.json").write_text(json.dumps(config, indent=2, allow_nan=False))
        (temporary / "automatic_inference.json").write_text(json.dumps(inference, indent=2, allow_nan=False))
        preview(temporary, meshes, poses, config["neutral_frames"])
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix="neutral-previous-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        temporary.rename(output)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not output.exists():
            backup.rename(output)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def run(source, output=None, overwrite=False, workers=1):
    """Process a specimen folder, ZIP, or parent folder of specimens.

    ``workers`` is a process count or ``"auto"``; results do not depend on it.

    Returns a list of ``{'specimen', 'status', 'output'}`` records (with
    ``'error'`` for failed specimens in batch mode).
    """
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if output is None:
        output = source.with_name((source.stem if source.is_file() else source.name) + "_neutral")
    output = Path(output).expanduser().resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError("Output cannot replace the original input or one of its parents.")
    with tempfile.TemporaryDirectory(prefix="neutral-input-") as temporary:
        if source.is_file():
            if source.suffix.lower() != ".zip":
                raise ValueError("Pass a specimen folder or ZIP archive.")
            safe_extract(source, temporary)
            root = Path(temporary)
        else:
            root = source
        folders = specimen_folders(root)
        folders = [p for p in folders if p != output and not p.is_relative_to(output)]
        if not folders:
            raise ValueError("No specimen folder containing at least two meshes was found.")
        if len(folders) == 1:
            report = process_specimen(folders[0], output, overwrite, workers)
            return [{"specimen": folders[0].name, "status": report["status"], "output": str(output)}]
        results = []
        for folder in folders:
            target = output / folder.relative_to(root)
            try:
                report = process_specimen(folder, target, overwrite, workers)
                results.append({"specimen": folder.name, "status": report["status"], "output": str(target)})
            except Exception as error:
                LOG.error("%s: %s", folder.name, error)
                results.append({"specimen": folder.name, "status": "failed", "error": str(error)})
        output.mkdir(parents=True, exist_ok=True)
        (output / "batch_summary.json").write_text(json.dumps(results, indent=2))
        return results


def main(argv=None):
    """Command-line entry point for the automatic workflow (``neutral-pose-auto``)."""
    parser = argparse.ArgumentParser(
        prog="neutral-pose-auto", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Specimen folder, parent of specimen folders, or ZIP.")
    parser.add_argument("--out", type=Path, help="Default: input name with _neutral appended.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing result folder.")
    parser.add_argument(
        "--workers",
        default="auto",
        help="Worker processes for independent per-joint work: a number or 'auto' (all CPUs, the default). "
        "Results are identical for any value; use 1 to run serially.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console verbosity (default: INFO).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    try:
        results = run(args.input, args.out, args.overwrite, resolve_workers(args.workers))
        for result in results:
            LOG.info("%s", json.dumps(result))
        return int(any(x["status"] == "failed" for x in results))
    except Exception as error:
        LOG.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
