"""Contact-aware selection and dense feasibility in the neutral-plane family."""

import logging
import math

import numpy as np
from scipy.optimize import minimize

from . import core as n

LOG = logging.getLogger(__name__)


def contact_support(metrics, minimum=0.10):
    """Per-contact minimum bidirectional coverage and whether each exceeds the presence threshold."""
    patches = metrics.get("patches", [])
    required = [p for p in patches if p["kind"] in ("centrum", "facet", "accessory")]
    names = [p["name"] for p in required]
    if len(set(names)) != len(names):
        raise ValueError("Required contact names must be unique")
    coverage = {
        p["name"]: min(p[side]["coverage_fraction"] for side in ("a_to_b", "b_to_a")) for p in required
    }
    values = [p[side]["coverage_fraction"] for p in required for side in ("a_to_b", "b_to_a")]
    missing = sorted(name for name, value in coverage.items() if value < minimum)
    return dict(
        minimum=min(values, default=0.0),
        mean=float(np.mean(values)) if values else 0.0,
        unsupported=len(missing),
        missing_contacts=missing,
        by_contact=coverage,
        required_contacts=len(required),
        support_threshold=minimum,
    )


def selection_reasons(reference, proposed, tolerance, allowance, converged=True, boundary=False):
    """Compare named contacts, then feasibility, then mean coverage.

    The 10% test detects absent contacts. It never replaces the 60% QC target.
    No uncertainty diagnostic alone is a reason to reinstate input curvature.
    """
    before, after = contact_support(reference), contact_support(proposed)
    reasons = []
    if not after["required_contacts"] or set(after["by_contact"]) != set(before["by_contact"]):
        return ["automatic_contact_identities_incomplete"]
    newly_lost = set(after["missing_contacts"]) - set(before["missing_contacts"])
    if newly_lost:
        reasons.append("automatic_fit_lost_required_contact")
    pb, pa = reference["max_sampled_penetration"], proposed["max_sampled_penetration"]
    feasible_before = pb is None or pb <= tolerance
    feasible_after = pa is None or pa <= tolerance
    if not converged or boundary:
        reasons.append("automatic_fit_not_converged_within_bounds")
    if feasible_after and not feasible_before and not after["unsupported"]:
        # A colliding reference cannot win solely on average coverage.
        return reasons
    if not feasible_after and (feasible_before or pa > pb + allowance):
        reasons.append("automatic_fit_increased_penetration")
    if before["unsupported"] > after["unsupported"] and not newly_lost and feasible_after:
        return reasons
    if after["mean"] < before["mean"] - 0.05:
        reasons.append("automatic_fit_reduced_contact_coverage")
    return reasons


def refine_dense_feasibility(joint, initial, fixed_sagittal=False):
    """Optimize the existing objective under an absolute penetration ceiling.

    At most 64 worst dense witnesses per reliable surface start the active
    set. Every proposed acceptance checks all vertices, centers, and independent
    area samples; additional violated witnesses expand the constraints.
    """
    tolerance = joint.penetration_tolerance * joint.scale
    receipt = dict(
        performed=False,
        accepted=False,
        tolerance_mm=tolerance,
        fixed_sagittal=fixed_sagittal,
        rounds=[],
        validation="all_vertices_triangle_centers_and_independent_area_samples",
    )
    if joint.neutral_plane is None:
        receipt["reason"] = "neutral_plane_required"
        return initial, receipt
    targets = []
    for mesh, other, reverse in ((joint.a, joint.b, False), (joint.b, joint.a, True)):
        if mesh.closed:
            points = np.vstack(
                [
                    other.vertices,
                    other.centers,
                    other.sample(max(2048, joint.options.collision_samples), seed=911).points,
                ]
            )
            targets.append((mesh, points, reverse))
    if not targets:
        receipt["reason"] = "no_reliable_inside_outside_signs"
        return initial, receipt
    plane, scale = joint.neutral_plane, joint.scale
    base = plane.encode(initial, scale)
    bounds = (
        np.full(2, joint.options.translation_bound_fraction)
        if fixed_sagittal
        else np.r_[
            math.radians(joint.options.rotation_bound_deg),
            np.full(2, joint.options.translation_bound_fraction),
        ]
    )

    def decode(x):
        return plane.decode(base + (np.r_[0.0, x] if fixed_sagittal else x), scale)

    def depths(H):
        return [
            mesh.penetration(n.transform(n.inverse(H) if reverse else H, points))
            for mesh, points, reverse in targets
        ]

    all_depths = depths(initial)
    receipt["initial_max_penetration_mm"] = max(float(x.max()) for x in all_depths)
    if receipt["initial_max_penetration_mm"] <= tolerance:
        receipt["reason"] = "already_within_dense_tolerance"
        return initial, receipt
    receipt["performed"] = True
    LOG.info(
        "Dense penetration refinement: %s (%.5g mm; ceiling %.5g mm)",
        joint.name,
        receipt["initial_max_penetration_mm"],
        tolerance,
    )
    active = [np.argsort(d)[-min(64, len(d)) :] for d in all_depths]
    x = np.zeros(len(bounds))
    # Interior margin avoids accepting a point numerically on the boundary.
    limit = max(tolerance - 1e-6 * scale, 0.999 * tolerance)

    def objective(y):
        residual = joint.residual(decode(y))
        return float(residual @ residual)

    def constraint(y):
        H = decode(y)
        # signed_distance >= -limit is the same feasible set as
        # max(-signed_distance, 0) <= limit, with useful exterior gradients.
        return np.concatenate(
            [
                (limit + mesh.signed_distance(n.transform(n.inverse(H) if reverse else H, points[ids])))
                / scale
                for (mesh, points, reverse), ids in zip(targets, active)
            ]
        )

    for round_index in range(4):
        seed_search = None
        if np.min(constraint(x)) < -1e-7:
            # At an interlocked input the linearized inequalities can be
            # inconsistent even though a nearby separated pose is feasible.
            # Deterministic starts cross that region without relaxing the
            # ceiling or changing the final objective. They are never exports.
            alternatives = [(float(np.min(constraint(x))), objective(x), x.copy())]
            angles = (0.0,) if fixed_sagittal else (0.0, -3.0, 3.0)
            for dz in (0.02, 0.04, 0.08, 0.16):
                for dy in (0.0, -0.02, 0.02):
                    for angle in angles:
                        candidate = (
                            np.array([dy, dz]) if fixed_sagittal else np.array([math.radians(angle), dy, dz])
                        )
                        if np.any(np.abs(candidate) > bounds):
                            continue
                        cmin = float(np.min(constraint(candidate)))
                        if cmin >= 0:
                            alternatives.append((cmin, objective(candidate), candidate))
            feasible = [item for item in alternatives if item[0] >= 0]
            if feasible:
                _, _, x = min(feasible, key=lambda item: item[1])
                seed_search = dict(feasible_starts=len(feasible), selected_offset=x.tolist())
        result = minimize(
            objective,
            x,
            method="SLSQP",
            bounds=list(zip(-bounds, bounds)),
            constraints=[dict(type="ineq", fun=constraint)],
            options=dict(maxiter=60, ftol=1e-10, eps=1e-5),
        )
        H = decode(result.x)
        all_depths = depths(H)
        depth = max(float(d.max()) for d in all_depths)
        receipt["rounds"].append(
            dict(
                round=round_index + 1,
                max_penetration_mm=depth,
                optimizer_converged=bool(result.success),
                optimizer_message=str(result.message),
                iterations=int(result.nit),
                active_witnesses=sum(len(a) for a in active),
                objective=float(result.fun),
                feasible_seed_search=seed_search,
            )
        )
        if np.isfinite(result.fun) and depth <= tolerance:
            receipt.update(
                accepted=True,
                final_max_penetration_mm=depth,
                converged=bool(result.success),
                boundary=bool(np.any(np.abs(result.x) > 0.98 * bounds)),
            )
            return n.validate_rigid(H), receipt
        if not np.isfinite(result.x).all():
            break
        for i, d in enumerate(all_depths):
            active[i] = np.union1d(active[i], np.argsort(d)[-min(64, len(d)) :])
        x = result.x
    receipt.update(reason="no_candidate_passed_full_dense_validation", final_max_penetration_mm=depth)
    return initial, receipt


def refresh_geometric_review(entry, joint):
    """Replace pose-specific flags after a selection or dense repair."""
    derived = {
        "penetration_exceeds_tolerance",
        "triangle_intersections_require_review",
        "poor_surface_fit",
        "insufficient_surface_coverage",
        "anchor_offset_exceeds_tolerance",
    }
    reasons = [r for r in entry["review_reasons"] if r not in derived]
    metrics = entry["metrics"]
    depth = metrics["max_sampled_penetration"]
    tolerance = joint.penetration_tolerance * joint.scale
    if depth is not None and depth > tolerance:
        reasons.append("penetration_exceeds_tolerance")
    if metrics["intersection_detected"]:
        if not metrics["signed_distance_reliable"] or depth is None or depth > tolerance:
            reasons.append("triangle_intersections_require_review")
        else:
            metrics["contact_within_noise_tolerance"] = True
    for patch in metrics["patches"]:
        for side in ("a_to_b", "b_to_a"):
            if patch[side]["rms_gap_error_fraction"] > joint.acceptable_surface_error:
                reasons.append("poor_surface_fit")
            if patch[side]["coverage_fraction"] < joint.options.minimum_coverage:
                reasons.append("insufficient_surface_coverage")
        if (
            not patch["spherical_centrum"]
            and patch["anchor_tangential_offset_mm"]
            > joint.options.centering_tolerance_fraction * joint.scale
        ):
            reasons.append("anchor_offset_exceeds_tolerance")
    entry["review_reasons"] = sorted(set(reasons))
    entry["status"] = "needs_review" if reasons else "passed_geometric_checks"
