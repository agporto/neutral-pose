"""Automatic bilateral landmark correspondence and robust reflection planes.

Landmark indices identify correspondence across bones, never anatomical roles.
A shared reflection permutation is inferred from the landmark geometry itself.
Accepted planes are checked against surfaces, which do not move the plane.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import core as n
from .anatomy import closest_surface

LOG = logging.getLogger(__name__)


def landmark_keys(labels):
    """Recognize uniform exporter numbering while preserving semantic labels."""
    labels = [str(label).strip() for label in labels]
    numbered = [re.fullmatch(r"(.+)-(\d+)", label) for label in labels]
    if labels and all(numbered) and len({m.group(1).casefold() for m in numbered}) == 1:
        keys = [f"LMK-{int(m.group(2))}" for m in numbered]
        source = "uniform_prefix_and_numeric_suffix"
    else:
        keys, source = labels, "exact_labels"
    if len(set(keys)) != len(keys) or any(not key for key in keys):
        raise ValueError("duplicate_or_empty_landmark_labels")
    return keys, source


def reflect(points, normal, point):
    """Reflect points through the plane with unit ``normal`` passing through ``point``."""
    return points - 2 * ((points - point) @ normal)[:, None] * normal


def angle(a, b):
    """Angle in degrees between two directions (sign-agnostic)."""
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), abs(a @ b))))


def point_scale(points):
    """Robust extent of a point set used to normalise residuals."""
    radius = np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1))
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("degenerate_landmark_positions")
    return float(radius)


def valid_permutation(permutation):
    """True if a pairing permutation is an involution with no fixed points."""
    p = np.asarray(permutation, dtype=int)
    return (
        p.ndim == 1
        and np.array_equal(np.sort(p), np.arange(len(p)))
        and np.array_equal(p[p], np.arange(len(p)))
    )


def fit_paired_plane(points, permutation, scale=None, seed=None, weights=None):
    """Fit an involutive reflection with paired Cauchy IRLS weights.

    For fixed symmetric weights, the minimum-eigenvalue eigenvector of the
    centered cross-covariance is the least-squares reflection normal. A
    weighted centroid lies on its plane. Every IRLS step preserves equal
    weights for both members of each pair. Reflections are measurements only.
    """
    P, p = np.asarray(points, float), np.asarray(permutation, int)
    if P.shape != (len(p), 3) or not np.isfinite(P).all() or not valid_permutation(p):
        raise ValueError("invalid_bilateral_correspondence")
    scale = point_scale(P) if scale is None else float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid_landmark_scale")
    base = np.ones(len(P)) if weights is None else np.asarray(weights, float).copy()
    if base.shape != (len(P),) or not np.isfinite(base).all() or np.any(base < 0):
        raise ValueError("invalid_landmark_weights")
    base = np.minimum(base, base[p])
    if np.count_nonzero(base) < 6:
        raise ValueError("too_few_paired_landmarks")
    W = base.copy()
    center = np.average(P, axis=0, weights=W)
    normal = None if seed is None else n.unit(seed)
    if normal is not None:
        mids = (P + P[p]) / 2
        center += normal * np.median((mids - center) @ normal)
    converged = False
    for iteration in range(50):  # noqa: B007  (final value is reported after the loop)
        if normal is not None:
            residual = np.linalg.norm(reflect(P, normal, center) - P[p], axis=1)
            active = residual[base > 0]
            cutoff = float(np.clip(1.5 * np.median(active), 0.005 * scale, 0.08 * scale))
            W = base / (1 + (residual / cutoff) ** 2)
            W = np.minimum(W, W[p])
        previous = None if normal is None else (normal.copy(), center.copy())
        center = np.average(P, axis=0, weights=W)
        X = (P - center) / scale
        cross = X.T @ (W[:, None] * X[p]) / W.sum()
        eigenvalues, vectors = np.linalg.eigh((cross + cross.T) / 2)
        normal = vectors[:, 0]
        if (
            previous is not None
            and angle(normal, previous[0]) < 1e-5
            and abs((center - previous[1]) @ normal) < 1e-7 * scale
        ):
            converged = True
            break
    residual = np.linalg.norm(reflect(P, normal, center) - P[p], axis=1)
    active = base > 0
    pairs = [(i, int(j)) for i, j in enumerate(p) if i < j and base[i] > 0]
    effective = sum(min(W[i], W[j]) > 0.2 for i, j in pairs)
    evidence = {
        "converged": converged,
        "iterations": iteration + 1,
        "median_reflection_error_mm": float(np.median(residual[active])),
        "p90_reflection_error_mm": float(np.quantile(residual[active], 0.9)),
        "reflection_error_mm": residual.tolist(),
        "weights": W.tolist(),
        "effective_pair_count": int(effective),
        "eigenvalue_gap": float(eigenvalues[1] - eigenvalues[0]),
        "downweighted_indices": np.flatnonzero((W < 0.2) & active).tolist(),
    }
    return normal, center, evidence


def discover_candidates(points, longitudinal_hint):
    """Find candidate involutions from landmarks without contact or mesh seeds."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 8 or not np.isfinite(P).all():
        raise ValueError("insufficient_finite_landmarks")
    scale = point_scale(P)
    tangent = n.unit(longitudinal_hint)
    center = np.median(P, axis=0)
    _, _, axes = np.linalg.svd(P - center, full_matrices=False)
    seeds = list(axes)
    for i in range(3):
        for j in range(i + 1, 3):
            seeds.extend([n.unit(axes[i] + axes[j]), n.unit(axes[i] - axes[j])])
    # True pair differences provide normals even when shape moments coincide.
    seeds.extend(
        P[j] - P[i]
        for i in range(len(P))
        for j in range(i + 1, len(P))
        if np.linalg.norm(P[j] - P[i]) > 0.1 * scale
    )
    unique, ranked = [], []
    for seed in seeds:
        normal = n.unit(seed)
        if abs(normal @ tangent) > 0.8 or any(
            abs(normal @ other) > math.cos(math.radians(3)) for other in unique
        ):
            continue
        unique.append(normal)
        # Median center resists an isolated misplaced point. Pair midpoints
        # refine plane offset after assignment.
        mirrored = reflect(P, normal, center)
        cost = np.linalg.norm(mirrored[:, None, :] - P[None, :, :], axis=2) / scale
        _, p = linear_sum_assignment(np.log1p((cost / 0.05) ** 2))
        if not valid_permutation(p) or np.count_nonzero(p != np.arange(len(P))) < 6:
            continue
        loss = float(np.mean(np.log1p((cost[np.arange(len(P)), p] / 0.05) ** 2)))
        ranked.append((loss, normal, p))
    ranked.sort(key=lambda item: item[0])
    results = {}
    for _, seed, permutation in ranked[:24]:
        normal = seed
        for _ in range(8):
            normal, point, evidence = fit_paired_plane(P, permutation, scale, normal)
            distances = np.linalg.norm(reflect(P, normal, point)[:, None, :] - P[None, :, :], axis=2) / scale
            _, proposed = linear_sum_assignment(np.log1p((distances / 0.05) ** 2))
            if not valid_permutation(proposed) or np.array_equal(proposed, permutation):
                break
            if np.count_nonzero(proposed != np.arange(len(P))) < 6:
                break
            permutation = proposed
        # Refit the final assignment even if the alternating loop hit its cap.
        normal, point, evidence = fit_paired_plane(P, permutation, scale, normal)
        if abs(normal @ tangent) > 0.7 or evidence["effective_pair_count"] < 3:
            continue
        if not evidence["converged"] or evidence["eigenvalue_gap"] < 0.01:
            continue
        distances = np.asarray(evidence["reflection_error_mm"]) / scale
        if np.median(distances) > 0.1 or np.quantile(distances, 0.9) > 0.25:
            continue
        score = float(np.mean(np.log1p((distances / 0.05) ** 2)))
        key = tuple(permutation.tolist())
        entry = {
            "permutation": permutation,
            "normal": normal,
            "point": point,
            "score": score,
            "quality": evidence,
        }
        if key not in results or score < results[key]["score"]:
            results[key] = entry
    return sorted(results.values(), key=lambda item: item["score"])


def _layout(mesh):
    if mesh.landmarks is None:
        raise ValueError("no_landmarks")
    keys, method = landmark_keys(mesh.landmarks.labels)
    order = sorted(range(len(keys)), key=lambda i: n.numeric_key(keys[i]))
    points = np.asarray(mesh.landmarks.points, float)[order] - mesh.origin
    if not np.isfinite(points).all():
        raise ValueError("nonfinite_landmarks")
    return tuple(keys[i] for i in order), points, [mesh.landmarks.labels[i] for i in order], method


def _assess_plane(mesh, P, permutation, seed, tangent, labels):
    import vtk

    normal, point, fit = fit_paired_plane(P, permutation, mesh.scale, seed)
    reasons = []
    if not fit["converged"]:
        reasons.append("landmark_plane_not_converged")
    if fit["effective_pair_count"] < 3 or fit["eigenvalue_gap"] < 0.005:
        reasons.append("landmark_plane_weakly_identified")
    if (
        fit["median_reflection_error_mm"] > 0.06 * mesh.scale
        or fit["p90_reflection_error_mm"] > 0.18 * mesh.scale
    ):
        reasons.append("poor_bilateral_landmark_match")
    if len(fit["downweighted_indices"]) / len(P) > 0.25:
        reasons.append("too_many_landmark_outliers")
    if abs(normal @ tangent) > 0.7:
        reasons.append("landmark_plane_normal_is_longitudinal")
    # Delete one complete pair (or one midline point), retaining equal paired
    # weights. A single influential annotation must not define the whole plane.
    angles = []
    if not reasons:
        for i, j in enumerate(permutation):
            if i > j:
                continue
            weights = np.ones(len(P))
            weights[i] = weights[j] = 0.0
            try:
                alternate, _, _ = fit_paired_plane(P, permutation, mesh.scale, normal, weights)
            except ValueError:
                reasons.append("insufficient_landmark_redundancy")
                break
            angles.append(angle(normal, alternate))
        if angles and max(angles) > 5.0:
            reasons.append("landmark_plane_depends_on_one_pair")
    probe = mesh.sample(1024, seed=2917).points
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh.poly)
    locator.BuildLocator()
    mirrored = reflect(probe, normal, point)
    errors = np.linalg.norm(mirrored - closest_surface(locator, mirrored), axis=1)
    if np.median(errors) > 0.06 * mesh.scale or np.quantile(errors, 0.9) > 0.15 * mesh.scale:
        reasons.append("landmark_plane_fails_surface_check")
    quality = {
        "source": "robust_paired_landmarks_with_surface_check",
        "review_reasons": reasons,
        "landmark_fit": fit,
        "landmark_pairs": [[labels[i], labels[j]] for i, j in enumerate(permutation) if i < j],
        "midline_landmarks": [labels[i] for i, j in enumerate(permutation) if i == j],
        "downweighted_landmarks": [labels[i] for i in fit["downweighted_indices"]],
        "leave_one_group_out_max_angle_deg": max(angles, default=None),
        "held_out_median_reflection_error_mm": float(np.median(errors)),
        "held_out_p90_reflection_error_mm": float(np.quantile(errors, 0.9)),
        "held_out_median_error_fraction": float(np.median(errors) / mesh.scale),
        "surface_check_refines_plane": False,
        "note": (
            "Bilateral identities are inferred from landmark geometry; anatomical semantics and biological "
            "neutrality are not certified. Reflection never changes output geometry."
        ),
    }
    return normal, point, quality


def prepare_landmark_planes(meshes):
    """Infer specimen-local schemas, then cache accepted landmark-first planes."""
    layouts, groups, tangents = {}, defaultdict(list), {}
    outcomes, schemas, searches = {}, [], {}
    for i, mesh in enumerate(meshes):
        for attribute in ("_automatic_bilateral_plane", "_landmark_symmetry_status"):
            if hasattr(mesh, attribute):
                delattr(mesh, attribute)
        tangent = meshes[min(i + 1, len(meshes) - 1)].origin - meshes[max(0, i - 1)].origin
        if np.linalg.norm(tangent) <= 1e-12 * mesh.scale:
            outcomes[mesh.name] = {"status": "surface_fallback", "reasons": ["no_longitudinal_order_hint"]}
            continue
        tangents[mesh.name] = n.unit(tangent)
        try:
            layout = _layout(mesh)
            if len(layout[0]) < 8:
                raise ValueError("too_few_landmarks_for_automatic_pairing")
        except ValueError as error:
            outcomes[mesh.name] = {"status": "surface_fallback", "reasons": [str(error)]}
            continue
        layouts[mesh.name] = layout
        groups[layout[0]].append(mesh)
    for keys, members in sorted(groups.items(), key=lambda item: (-len(item[0]), item[0])):
        # Partial files use a previously inferred superset when it is unique.
        compatible = [schema for schema in schemas if set(keys).issubset(schema["keys"])]
        if compatible:
            continue
        votes = Counter()
        for mesh in members:
            LOG.info("Inferring landmark symmetry: %s", mesh.name)
            try:
                candidates = discover_candidates(layouts[mesh.name][1], tangents[mesh.name])
            except ValueError as error:
                searches[mesh.name] = {"error": str(error)}
                continue
            searches[mesh.name] = {
                "distinct_candidate_count": len(candidates),
                "candidate_scores": [c["score"] for c in candidates],
            }
            if candidates:
                best = candidates[0]
                near = [c for c in candidates if c["score"] <= best["score"] * 1.2 + 0.01]
                searches[mesh.name]["near_optimal_candidate_count"] = len(near)
                searches[mesh.name]["best_permutation"] = best["permutation"].tolist()
                # Ambiguous point patterns do not get a vote by arbitrary order.
                if len(near) == 1:
                    votes[tuple(best["permutation"].tolist())] += 1
        if not votes:
            continue
        permutation, count = votes.most_common(1)[0]
        if count < 2 or count / len(members) < 0.8:
            continue
        schema = {
            "id": f"landmark_schema_{len(schemas) + 1}",
            "keys": list(keys),
            "permutation": list(permutation),
            "supporting_bones": count,
            "eligible_bones": len(members),
            "agreement_fraction": count / len(members),
            "source": "consensus_of_independent_landmark_reflection_assignments",
            "bilateral_pairs": [[keys[i], keys[j]] for i, j in enumerate(permutation) if i < j],
            "midline_keys": [keys[i] for i, j in enumerate(permutation) if i == j],
        }
        schemas.append(schema)
    for mesh in meshes:
        if mesh.name in outcomes:
            continue
        keys, P, labels, normalization = layouts[mesh.name]
        compatible = [s for s in schemas if set(keys).issubset(s["keys"])]
        if len(compatible) != 1:
            outcomes[mesh.name] = {
                "status": "surface_fallback",
                "reasons": ["unresolved_landmark_pairing_consensus"],
            }
            continue
        schema = compatible[0]
        partner = dict(zip(schema["keys"], [schema["keys"][j] for j in schema["permutation"]]))
        selected = [i for i, key in enumerate(keys) if partner[key] in keys]
        available = [keys[i] for i in selected]
        permutation = np.array([available.index(partner[key]) for key in available])
        if len(selected) < 8 or np.count_nonzero(permutation != np.arange(len(selected))) < 6:
            outcomes[mesh.name] = {
                "status": "surface_fallback",
                "reasons": ["insufficient_complete_landmark_pairs"],
            }
            continue
        P, labels = P[selected], [labels[i] for i in selected]
        # Consensus supplies correspondence, not a borrowed orientation. Each
        # bone's own paired differences provide its local starting normal.
        differences = P - P[permutation]
        norms = np.linalg.norm(differences, axis=1)
        directions = differences[norms > 1e-10 * mesh.scale] / norms[norms > 1e-10 * mesh.scale, None]
        _, vectors = np.linalg.eigh(directions.T @ directions)
        normal, point, quality = _assess_plane(
            mesh, P, permutation, vectors[:, -1], tangents[mesh.name], labels
        )
        quality.update(
            schema_id=schema["id"],
            label_normalization=normalization,
            omitted_unpaired_landmarks=[
                layouts[mesh.name][2][i] for i in range(len(keys)) if i not in selected
            ],
        )
        reasons = quality["review_reasons"]
        outcomes[mesh.name] = {
            "status": "surface_fallback" if reasons else "landmark_plane",
            "reasons": reasons,
            "schema_id": schema["id"],
            "quality": quality,
        }
        if not reasons:
            mesh._automatic_bilateral_plane = (normal, point, quality)
    for mesh in meshes:
        if mesh.name in searches:
            outcomes[mesh.name]["independent_landmark_search"] = searches[mesh.name]
        mesh._landmark_symmetry_status = outcomes[mesh.name]
        if outcomes[mesh.name]["status"] == "surface_fallback":
            LOG.info(
                "Landmark plane unavailable for %s: %s; checking surfaces.",
                mesh.name,
                ", ".join(outcomes[mesh.name]["reasons"]),
            )
    counts = Counter(item["status"] for item in outcomes.values())
    LOG.info(
        "Landmark symmetry: %d accepted planes; %d require surface fallback",
        counts["landmark_plane"],
        counts["surface_fallback"],
    )
    return {
        "method": "landmark_first_with_explicit_surface_fallback",
        "schemas": schemas,
        "landmark_plane_count": counts["landmark_plane"],
        "surface_fallback_count": counts["surface_fallback"],
        "bones": outcomes,
    }
