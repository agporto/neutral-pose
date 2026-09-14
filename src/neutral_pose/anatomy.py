"""Use landmark-first symmetry planes, with a checked surface-only fallback.

Accepted landmark planes are cached before contacts are assigned. If landmark
correspondence or fit quality is unresolved, intrinsic shape directions and
unlabelled contact pairs seed robust whole-surface reflection fitting.
Reflection is used only as a measurement: output bones undergo proper rigid
transforms through :class:`neutral_pose.core.NeutralPlane`.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import vtk
from scipy.optimize import least_squares

from . import core as n

LOG = logging.getLogger(__name__)


def closest_surface(locator, points):
    """Closest surface points for ``points`` using a prebuilt VTK cell locator."""
    output = np.empty_like(points)
    cell, sub, distance = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    for i, point in enumerate(points):
        q = [0.0, 0.0, 0.0]
        locator.FindClosestPoint(point, q, cell, sub, distance)
        output[i] = q
    return output


def symmetry_plane(mesh, normal_seed, center_seed, samples=1024):
    """Fit a local bilateral plane with a rotationally invariant robust loss.

    Area sampling makes dense triangulation and small disconnected islands
    contribute in proportion to physical surface area. Independent samples
    measure held-out reflection mismatch; this does not certify true symmetry.
    """
    normal_seed = n.unit(normal_seed)
    center_seed = np.asarray(center_seed, float)
    transverse = np.eye(3)[np.argmin(np.abs(normal_seed))]
    u = n.unit(np.cross(normal_seed, transverse))
    v = np.cross(normal_seed, u)
    P = mesh.sample(samples, seed=1741).points
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh.poly)
    locator.BuildLocator()

    def decode(x):
        normal = n.unit(normal_seed + x[0] * u + x[1] * v)
        return normal, center_seed + x[2] * mesh.scale * normal

    def distances(x, points):
        normal, center = decode(x)
        mirrored = points - 2 * ((points - center) @ normal)[:, None] * normal
        return np.linalg.norm(mirrored - closest_surface(locator, mirrored), axis=1) / mesh.scale

    bounds = np.array([math.tan(math.radians(25))] * 2 + [0.2])
    candidates = []
    for seed in (np.zeros(3), np.array([0.06, -0.04, 0.0]), np.array([-0.06, 0.04, 0.0])):
        fit = least_squares(
            distances,
            seed,
            args=(P,),
            bounds=(-bounds, bounds),
            loss="soft_l1",
            f_scale=0.015,
            x_scale="jac",
            diff_step=1e-4,
            max_nfev=55,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        candidates.append(fit)
    best = min(candidates, key=lambda fit: fit.cost)
    normal, center = decode(best.x)
    Q = mesh.sample(samples, seed=2917).points
    held_out = distances(best.x, Q) * mesh.scale
    sv = np.linalg.svd(best.jac, compute_uv=False)
    condition = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else None
    near = [fit for fit in candidates if fit.cost <= best.cost * 1.1 + 1e-8]
    alternative_angles = [
        float(np.rad2deg(np.arccos(np.clip(abs(decode(fit.x)[0] @ normal), 0.0, 1.0)))) for fit in near
    ]
    reasons = []
    if not best.success:
        reasons.append("symmetry_plane_optimizer_not_converged")
    if np.any(np.abs(best.x) > 0.98 * bounds):
        reasons.append("symmetry_plane_search_boundary")
    if condition is None or condition > 1e5:
        reasons.append("symmetry_plane_weakly_identified")
    if max(alternative_angles) > 3.0:
        reasons.append("multiple_symmetry_planes")
    if np.median(held_out) > 0.06 * mesh.scale or np.quantile(held_out, 0.9) > 0.15 * mesh.scale:
        reasons.append("poor_bilateral_surface_match")
    evidence = {
        "source": "paired_facets_and_robust_whole_surface_reflection",
        "robust_objective": float(2 * best.cost / len(P)),
        "converged": bool(best.success),
        "evaluations": int(best.nfev),
        "seed_correction_deg": float(np.rad2deg(np.arccos(np.clip(normal @ normal_seed, -1.0, 1.0)))),
        "held_out_median_reflection_error_mm": float(np.median(held_out)),
        "held_out_p90_reflection_error_mm": float(np.quantile(held_out, 0.9)),
        "held_out_median_error_fraction": float(np.median(held_out) / mesh.scale),
        "jacobian_condition_number": condition,
        "near_optimal_plane_disagreement_deg": max(alternative_angles),
        "review_reasons": reasons,
        "note": "Inferred bilateral plane; reflection is diagnostic only, never an output transformation.",
    }
    return normal, center, evidence


def intrinsic_symmetry_plane(mesh, longitudinal_hint, lateral_hints=()):
    """Estimate bilateral symmetry before deciding which contacts are facets.

    Intrinsic shape axes and unlabelled contact-pair directions are checked
    against the full surface before assigning any facet labels.
    This breaks the circular dependency in which incorrectly named contacts
    previously supplied the only plane seed. Search and fit-quality gates are
    retained, and materially different near-optimal planes remain a failure.
    """
    cached = getattr(mesh, "_automatic_bilateral_plane", None)
    if cached is not None:
        return cached
    longitudinal_hint = n.unit(longitudinal_hint)
    P = mesh.sample(4096, seed=11).points
    center = P.mean(0)
    _, singular, axes = np.linalg.svd(P - center, full_matrices=False)
    seeds = [(axis, f"principal_axis_{i}") for i, axis in enumerate(axes)]
    # Near-equal moments make principal-axis direction unstable. Include
    # diagonal directions in those eigenspaces instead of trusting one basis.
    for i in range(3):
        for j in range(i + 1, 3):
            if singular[i] / singular[j] < 1.2:
                seeds.extend(
                    [(n.unit(axes[i] + sign * axes[j]), f"principal_axes_{i}_{j}_{sign}") for sign in (-1, 1)]
                )
    seeds.extend((n.unit(hint), f"contact_pair_{i}") for i, hint in enumerate(lateral_hints))
    unique = []
    for axis, label in seeds:
        if abs(axis @ longitudinal_hint) > 0.8 or any(
            abs(axis @ other) > math.cos(math.radians(0.5)) for other, _ in unique
        ):
            continue
        unique.append((axis, label))
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh.poly)
    locator.BuildLocator()
    probe = mesh.sample(384, seed=317).points
    ranked = []
    for axis, label in unique:
        mirrored = probe - 2 * ((probe - center) @ axis)[:, None] * axis
        distances = np.linalg.norm(mirrored - closest_surface(locator, mirrored), axis=1) / mesh.scale
        objective = float(np.mean(np.sqrt(1 + (distances / 0.015) ** 2) - 1))
        ranked.append((objective, axis, label))
    ranked.sort(key=lambda x: x[0])
    if not ranked:
        raise ValueError(f"{mesh.name}: no transverse anatomical-plane starting directions.")
    # Broad screening removes only seeds whose reflected shape is already far
    # worse. All remaining seeds share the same bounded refinement and gates.
    screened = [x for x in ranked if x[0] <= 4 * ranked[0][0] + 0.02]
    candidates = []
    for _, axis, label in screened:
        # Bilateral normals should be transverse to the ordered column. The
        # broad gate allows curvature and individual-bone shape variation.
        if abs(axis @ longitudinal_hint) > 0.8:
            continue
        normal, point, evidence = symmetry_plane(mesh, axis, center)
        if abs(normal @ longitudinal_hint) > 0.7:
            evidence["review_reasons"].append("plane_normal_is_longitudinal")
        candidates.append((normal, point, evidence, label))
    valid = [c for c in candidates if not c[2]["review_reasons"]]
    if not valid:
        reasons = sorted({reason for c in candidates for reason in c[2]["review_reasons"]})
        raise ValueError(f"{mesh.name}: no reliable intrinsic bilateral plane: " + ", ".join(reasons))
    best = min(valid, key=lambda c: c[2]["robust_objective"])
    # A lower-loss but unresolved alternative is evidence of ambiguity, not
    # permission to accept a worse, apparently converged plane.
    near = [
        c
        for c in candidates
        if "plane_normal_is_longitudinal" not in c[2]["review_reasons"]
        and c[2]["robust_objective"] <= best[2]["robust_objective"] * 1.15 + 1e-8
    ]
    disagreement = max(float(np.rad2deg(np.arccos(np.clip(abs(c[0] @ best[0]), 0.0, 1.0)))) for c in near)
    if disagreement > 5.0 or any("multiple_symmetry_planes" in c[2]["review_reasons"] for c in near):
        raise ValueError(f"{mesh.name}: multiple distinct intrinsic bilateral planes fit similarly well.")
    normal, point, evidence, label = best
    evidence = dict(
        evidence,
        source="whole_surface_symmetry_from_shape_and_unlabelled_contacts",
        selected_seed=label,
        cross_seed_disagreement_deg=disagreement,
        intrinsic_candidates=[dict(c[2], seed=c[3]) for c in candidates],
    )
    landmark_attempt = getattr(mesh, "_landmark_symmetry_status", None)
    if landmark_attempt is not None:
        evidence["landmark_fallback"] = landmark_attempt
    mesh._automatic_bilateral_plane = (normal, point, evidence)
    return mesh._automatic_bilateral_plane


def mirrored_patch_overlap(mesh, first, second, normal, point):
    """Check unequal contact footprints without treating their centers as homologues."""
    ids = [np.asarray(first, int), np.asarray(second, int)]
    areas = [float(mesh.areas[x].sum()) for x in ids]
    coverage, medians = [], []
    for i, j in ((0, 1), (1, 0)):
        probe = mesh.sample(192, ids[i], seed=4601).points
        mirrored = probe - 2 * ((probe - point) @ normal)[:, None] * normal
        poly = n.polydata(mesh.vertices, mesh.faces[ids[j]])
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(poly)
        locator.BuildLocator()
        distances = np.linalg.norm(mirrored - closest_surface(locator, mirrored), axis=1) / mesh.scale
        coverage.append(float(np.mean(distances < 0.08)))
        medians.append(float(np.median(distances)))
    small, large = np.argsort(areas)
    ratio = areas[small] / areas[large]
    return {
        "distance_allowance_fraction": 0.08,
        "directional_coverage": coverage,
        "directional_median_distance_fraction": medians,
        "supported": bool(coverage[small] >= 0.75 and coverage[large] >= 0.5 * ratio),
        "criterion": (
            "At least 75% of the smaller patch and half its equivalent area on the larger patch "
            "overlap after reflection."
        ),
    }


def choose_bilateral_contacts(a, b, candidates, centrum):
    """Choose opposing left/right contacts by reflection, not surface-area rank."""
    from itertools import combinations

    tangent = n.unit(b.origin - a.origin)
    anchors = [
        {
            side: np.average(mesh.centers[c["i" + side]], axis=0, weights=mesh.areas[c["i" + side]])
            for side, mesh in (("a", a), ("b", b))
        }
        for c in candidates
    ]
    pairs = list(combinations([k for k in range(len(candidates)) if k != centrum], 2))
    planes = [
        intrinsic_symmetry_plane(
            mesh,
            tangent,
            [
                anchors[j][side] - anchors[i][side]
                for i, j in pairs
                if np.linalg.norm(anchors[j][side] - anchors[i][side]) > 1e-10 * mesh.scale
            ],
        )
        for side, mesh in (("a", a), ("b", b))
    ]
    ranked = []
    for i, j in pairs:
        diagnostics = []
        for side, mesh, plane in zip(("a", "b"), (a, b), planes):
            normal, point, _ = plane
            p, q = anchors[i][side], anchors[j][side]
            dp, dq = float((p - point) @ normal), float((q - point) @ normal)
            mirrored = p - 2 * dp * normal
            error = float(np.linalg.norm(mirrored - q) / mesh.scale)
            np_, nq = candidates[i]["n" + side], candidates[j]["n" + side]
            normal_agreement = float((np_ - 2 * (np_ @ normal) * normal) @ nq)
            area = [float(mesh.areas[candidates[k]["i" + side]].sum()) for k in (i, j)]
            ratio = min(area) / max(area)
            side_fractions = []
            for k, signed_center in ((i, dp), (j, dq)):
                ids = candidates[k]["i" + side]
                signed = (mesh.centers[ids] - point) @ normal
                side_fractions.append(
                    float(
                        np.average(
                            np.sign(signed_center) * signed > 0.005 * mesh.scale, weights=mesh.areas[ids]
                        )
                    )
                )
            geometric_support = (
                dp * dq < 0
                and min(abs(dp), abs(dq)) > 0.025 * mesh.scale
                and min(side_fractions) >= 0.9
                and normal_agreement > 0.35
                and ratio > 0.15
            )
            overlap = None
            if geometric_support and error >= 0.2:
                overlap = mirrored_patch_overlap(
                    mesh, candidates[i]["i" + side], candidates[j]["i" + side], normal, point
                )
            diagnostics.append(
                {
                    "side": side,
                    "signed_center_distances_mm": [dp, dq],
                    "reflection_error_fraction": error,
                    "normal_agreement": normal_agreement,
                    "area_ratio": ratio,
                    "same_side_area_fractions": side_fractions,
                    "partial_patch_overlap": overlap,
                    "supported": bool(geometric_support and (error < 0.2 or overlap["supported"])),
                }
            )
        if all(x["supported"] for x in diagnostics):
            score = float(
                np.mean(
                    [
                        x["reflection_error_fraction"]
                        + 0.04 * (1 - x["normal_agreement"])
                        + 0.015 * abs(math.log(x["area_ratio"]))
                        for x in diagnostics
                    ]
                )
            )
            ranked.append((score, i, j, diagnostics))
    if not ranked:
        raise ValueError(
            f"{a.name}/{b.name}: no two candidate contacts form a supported bilateral pair on both bones."
        )
    ranked.sort(key=lambda x: x[0])
    score, i, j, diagnostics = ranked[0]
    return {i, j}, {
        "method": "opposite_sides_and_mirrored_surface_geometry",
        "selected_candidate_indices": [i, j],
        "score": score,
        "sides": diagnostics,
        "supported_pair_count": len(ranked),
    }


def infer_frames(meshes, config):
    """Use intrinsic bone geometry to infer right-handed (lateral,dorsal,posterior) axes."""
    observations = {m.name: [] for m in meshes}
    for joint in config["joints"]:
        facets = [p for p in joint["patches"] if p["kind"] == "facet"]
        centra = [p for p in joint["patches"] if p["kind"] == "centrum"]
        if len(facets) != 2 or len(centra) != 1:
            continue
        for side, end in (("a", "posterior"), ("b", "anterior")):
            left, right = [np.asarray(p[side]["inferred_anchor"], float) for p in facets]
            centrum = np.asarray(centra[0][side]["inferred_anchor"], float)
            observations[joint[side]].append(
                {
                    "lateral": n.unit(right - left),
                    "facet_midpoint": (left + right) / 2,
                    "centrum": centrum,
                    "end": end,
                }
            )
    frames = {}
    for i, mesh in enumerate(meshes):
        ends = observations[mesh.name]
        if not ends:
            raise ValueError(
                f"{mesh.name}: cannot infer a neutral plane without a centrum and two bilateral "
                "facet contacts. "
                "Automatic processing cannot establish neutral lateral bending from these surfaces."
            )
        directions = np.array([x["lateral"] for x in ends])
        _, eigenvectors = np.linalg.eigh(directions.T @ directions)
        lateral_seed = eigenvectors[:, -1]
        mids = np.array([x["facet_midpoint"] for x in ends] + [x["centrum"] for x in ends])
        center_seed = mids.mean(0) - mesh.origin
        LOG.info("Building anatomical frame %d/%d: %s", i + 1, len(meshes), mesh.name)
        cached = getattr(mesh, "_automatic_bilateral_plane", None)
        lateral, plane_point, evidence = (
            cached if cached is not None else symmetry_plane(mesh, lateral_seed, center_seed)
        )
        if evidence["review_reasons"]:
            raise ValueError(
                f"{mesh.name}: anatomical symmetry plane is unresolved: "
                + ", ".join(evidence["review_reasons"])
            )
        centra = {x["end"]: x["centrum"] for x in ends}
        if len(centra) == 2:
            posterior = centra["posterior"] - centra["anterior"]
        else:
            posterior = meshes[min(i + 1, len(meshes) - 1)].origin - meshes[max(0, i - 1)].origin
        posterior = n.unit(posterior - lateral * (posterior @ lateral))
        dorsal = n.unit(np.cross(posterior, lateral))
        dorsal_hint = np.mean([x["facet_midpoint"] - x["centrum"] for x in ends], axis=0)
        if dorsal @ dorsal_hint < 0:
            lateral, dorsal = -lateral, -dorsal
        axes = np.column_stack([lateral, dorsal, posterior])
        n.validate_rigid(n.rigid(axes))
        center = np.mean(list(centra.values()), axis=0) - mesh.origin
        center -= lateral * ((center - plane_point) @ lateral)
        frames[mesh.name] = {
            "axes": axes.tolist(),
            "center": (center + mesh.origin).tolist(),
            "axis_order": ["lateral", "dorsal", "posterior"],
            "coordinate_system": config["mesh_coordinate_system"],
            "quality": evidence,
            "facet_axis_deviation_deg": [
                float(np.rad2deg(np.arccos(np.clip(abs(d @ lateral), 0.0, 1.0)))) for d in directions
            ],
        }
    return frames


def pool_bilateral_spacing(config, inference):
    """Use equal left/right spacing evidence so input lateral bend is not a target."""
    for definition, receipt in zip(config["joints"], inference["joints"]):
        facets = [p for p in definition["patches"] if p["kind"] == "facet"]
        if len(facets) != 2:
            continue
        by_name = {p["name"]: p for p in receipt["patches"]}
        signed_medians = [by_name[p["name"]]["input_signed_gap_quantiles_mm"][1] for p in facets]
        pooled = max(0.0, float(np.mean(signed_medians)))
        for patch in facets:
            evidence = by_name[patch["name"]]
            evidence["unpooled_estimated_gap_mm"] = evidence["estimated_gap_mm"]
            evidence["estimated_gap_mm"] = pooled
            evidence["spacing_source"] = "equal_bilateral_mean_of_signed_input_gap_medians"
            evidence["bilateral_signed_gap_medians_mm"] = signed_medians
            patch["gap_fraction"] = pooled / receipt["local_scale_mm"]


def column_diagnostics(meshes, poses, frames):
    """Residual lateral bending and axial twist of the posed column relative to the first bone's frame."""
    root_axes = np.asarray(frames[meshes[0].name]["axes"])
    root_center = np.asarray(frames[meshes[0].name]["center"])
    centers = np.array([frames[m.name]["center"] for m in meshes])
    moved = np.array([n.transform(T, p - m.origin) for T, p, m in zip(poses, centers, meshes)])
    input_points = (centers - root_center) @ root_axes
    output_points = (moved - root_center) @ root_axes
    alignment = []
    for mesh, pose in zip(meshes, poses):
        normal = pose[:3, :3] @ np.asarray(frames[mesh.name]["axes"])[:, 0]
        alignment.append(
            float(
                np.rad2deg(
                    np.arctan2(np.linalg.norm(np.cross(normal, root_axes[:, 0])), normal @ root_axes[:, 0])
                )
            )
        )
    return {
        "definition": (
            "Coincident inferred midsagittal planes; "
            "sagittal rotations and in-plane translations remain free."
        ),
        "reference": "First bone's inferred anatomical frame; first bone remains fixed.",
        "input_lateral_center_range_mm": float(np.ptp(input_points[:, 0])),
        "output_lateral_center_range_mm": float(np.ptp(output_points[:, 0])),
        "output_max_absolute_lateral_center_offset_mm": float(np.max(abs(output_points[:, 0]))),
        "output_max_symmetry_plane_misalignment_deg": max(alignment),
        "input_center_coordinates_lateral_dorsal_posterior_mm": input_points.tolist(),
        "output_center_coordinates_lateral_dorsal_posterior_mm": output_points.tolist(),
        "note": (
            "These are constraint checks conditional on inferred anatomical planes, "
            "not independent validation of anatomy or cartilage spacing."
        ),
    }
