"""Specimen-local contact learning and checked recovery of incomplete joints.

Labels are identifiers, never anatomical role numbers. A learned alignment is
only a search seed: final patches must be rediscovered on opposing surfaces.
"""

import logging
import math
from collections import Counter
from itertools import permutations

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra

from . import anatomy, landmarks
from . import core as n
from .surfaces import boundary_spec, surface_queries

LOG = logging.getLogger(__name__)
POLICY = "complete_contacts_with_supported_pose_selection_v1"

#: Maximum mean landmark-to-surface distance (fraction of bone scale) for a
#: surface region to be identified with a learned contact role.
ROLE_IDENTITY_MAX_DISTANCE = 0.10
#: Minimum mean cost increase per reassigned role between the best and any
#: alternative one-to-one assignment; smaller margins are rejected as ambiguous.
ROLE_IDENTITY_MIN_MARGIN = 0.015
#: A partitioned sub-region is retained only above this many faces and this
#: fraction of scale² in area (matches the discovery minimum contact area).
PARTITION_MIN_FACES = 3
PARTITION_MIN_AREA_FRACTION = 0.002


def complete(definition):
    """True if a joint definition has exactly one centrum and two facet patches."""
    kinds = Counter(p.get("kind") for p in definition.get("patches", []))
    return kinds["centrum"] == 1 and kinds["facet"] == 2


def landmark_map(mesh):
    """Normalised landmark key -> local (origin-centred) point for a mesh."""
    if mesh.landmarks is None:
        raise ValueError(f"{mesh.name}: no landmarks for contact recovery")
    keys, _ = landmarks.landmark_keys(mesh.landmarks.labels)
    return dict(zip(keys, mesh.landmarks.points - mesh.origin))


def support_keys(mesh, evidence, side):
    """Normalised keys of the landmarks recorded as supporting a contact on one side."""
    keys, _ = landmarks.landmark_keys(mesh.landmarks.labels)
    mapping = dict(zip(mesh.landmarks.labels, keys))
    return {
        mapping[x["label"]] for x in evidence.get("landmark_support_" + side, []) if x["label"] in mapping
    }


def learn_groups(observations, count, expected):
    """Group consistently co-occurring labels; refuse ambiguous partitions."""
    freq = Counter(k for obs in observations for k in obs)
    keys = sorted(k for k, f in freq.items() if f >= max(3, 0.5 * count))
    if not keys:
        raise ValueError("Too few repeatable contact landmarks")
    co = np.array([[sum(a in obs and b in obs for obs in observations) for b in keys] for a in keys])
    linked = co >= 0.8 * np.minimum.outer([freq[k] for k in keys], [freq[k] for k in keys])
    number, labels = connected_components(linked, directed=False)
    groups = [set(k for k, label in zip(keys, labels) if label == i) for i in range(number)]
    if number != expected:
        raise ValueError("Contact landmark co-occurrence does not identify unique groups")
    # A chain of pairwise associations must not hide conflicting assignments.
    for group in groups:
        ids = [keys.index(k) for k in group]
        if not linked[np.ix_(ids, ids)].all():
            raise ValueError("Conflicting contact landmark associations")
    return groups


def group_index(observed, groups):
    """Index of the single learned group intersecting ``observed``; ``None`` if zero or several."""
    hits = [i for i, g in enumerate(groups) if observed & g]
    return hits[0] if len(hits) == 1 else None


def spacing_statistics(config, inference, indices):
    """Median/MAD clearance statistics per contact kind over complete joints."""
    statistics = {}
    for kind in ("centrum", "facet"):
        values = [
            float(np.mean([p["gap_fraction"] for p in config["joints"][i]["patches"] if p["kind"] == kind]))
            for i in indices
        ]
        if not values:
            continue
        median = float(np.median(values))
        mad = float(1.4826 * np.median(np.abs(np.asarray(values) - median)))
        statistics[kind] = dict(
            median=median, mad=mad, upper=median + 3 * max(mad, 0.001), supporting_joints=len(values)
        )
    return statistics


def adjust_spacing(config, inference, statistics, indices):
    """One vote per joint. Only isolated high facet observations are replaced."""
    changes = []
    stat = statistics.get("facet")
    if stat is None or stat["supporting_joints"] < 5:
        return changes
    for i in indices:
        definition, receipt = config["joints"][i], inference["joints"][i]
        evidence = {p["name"]: p for p in receipt["patches"]}
        facets = [p for p in definition["patches"] if p["kind"] == "facet"]
        if float(np.mean([p["gap_fraction"] for p in facets])) <= stat["upper"]:
            continue
        for patch in facets:
            original = patch["gap_fraction"]
            patch["gap_fraction"] = stat["median"]
            e = evidence[patch["name"]]
            e["original_pooled_gap_mm"] = e["estimated_gap_mm"]
            e["estimated_gap_mm"] = stat["median"] * receipt["local_scale_mm"]
            e["spacing_source"] = "specimen_median_for_high_facet_spacing_outlier"
            e["spacing_statistics"] = stat.copy()
            changes.append(
                dict(
                    joint_index=i,
                    patch=patch["name"],
                    original_fraction=original,
                    replacement_fraction=stat["median"],
                )
            )
    return changes


def learn_model(meshes, config, inference, indices):
    """Learn contact landmark groups, pairing, normals and patch sizes from complete joints."""
    if len(indices) < 3:
        raise ValueError("Contact recovery needs at least three complete, supported joints")
    eligible = []
    for i in indices:
        pair = meshes[i : i + 2]
        if all(
            m.landmarks is not None
            and getattr(m, "_landmark_symmetry_status", {}).get("status") == "landmark_plane"
            for m in pair
        ):
            eligible.append(i)
    if len(eligible) < 3:
        raise ValueError("Contact recovery needs accepted landmark symmetry planes on three complete joints")
    observations = {(kind, side): [] for kind in ("centrum", "facet") for side in ("a", "b")}
    for i in eligible:
        for patch in inference["joints"][i]["patches"]:
            kind = patch["inferred_kind"]
            if kind not in ("centrum", "facet"):
                continue
            for side, mesh in zip(("a", "b"), meshes[i : i + 2]):
                observations[kind, side].append(support_keys(mesh, patch, side))
    groups = {
        (kind, side): learn_groups(obs, len(eligible), 1 if kind == "centrum" else 2)
        for (kind, side), obs in observations.items()
    }
    votes = np.zeros((2, 2), int)
    for i in eligible:
        assigned = []
        for patch in inference["joints"][i]["patches"]:
            if patch["inferred_kind"] == "facet":
                assigned.append(
                    [
                        group_index(support_keys(mesh, patch, side), groups["facet", side])
                        for side, mesh in zip(("a", "b"), meshes[i : i + 2])
                    ]
                )
        if len(assigned) == 2 and all(x is not None for row in assigned for x in row):
            if len({row[0] for row in assigned}) == len({row[1] for row in assigned}) == 2:
                for a, b in assigned:
                    votes[a, b] += 1
    pairing = np.argmax(votes, axis=1)
    if len(set(pairing)) != 2 or any(
        votes[i, j] < max(3, 0.8 * len(eligible)) for i, j in enumerate(pairing)
    ):
        raise ValueError("Conflicting or insufficient facet correspondence votes")
    roles = [
        dict(
            name="inferred_centrum",
            kind="centrum",
            a=sorted(groups["centrum", "a"][0]),
            b=sorted(groups["centrum", "b"][0]),
        )
    ]
    roles += [
        dict(
            name=f"inferred_facet_{i + 1}",
            kind="facet",
            a=sorted(groups["facet", "a"][i]),
            b=sorted(groups["facet", "b"][j]),
        )
        for i, j in enumerate(pairing)
    ]
    for side in ("a", "b"):
        assigned = [key for role in roles for key in role[side]]
        if len(assigned) != len(set(assigned)):
            raise ValueError("Learned anatomical contact roles share ambiguous landmarks")
    covered = {config["joints"][i][side] for i in indices for side in ("a", "b")}
    frames = anatomy.infer_frames([m for m in meshes if m.name in covered], config)
    for role in roles:
        role["radius_fraction"], role["normal_in_frame"] = {}, {}
        for side in ("a", "b"):
            radii, normals = [], []
            for i in eligible:
                mesh = meshes[i + (side == "b")]
                definition, receipt = config["joints"][i], inference["joints"][i]
                by_name = {p["name"]: p for p in receipt["patches"]}
                for patch in definition["patches"]:
                    support = support_keys(mesh, by_name[patch["name"]], side)
                    if patch["kind"] != role["kind"] or not support & set(role[side]):
                        continue
                    boundary = patch[side]["automatic_boundary"]
                    radii.append(boundary["radius_mm"] / mesh.scale)
                    normals.append(np.asarray(frames[mesh.name]["axes"]).T @ np.asarray(boundary["normal"]))
            if len(radii) < 3:
                raise ValueError("Insufficient geometric support for learned contact role")
            role["radius_fraction"][side] = float(np.median(radii))
            role["normal_in_frame"][side] = n.unit(np.median(normals, axis=0)).tolist()
    # A common left/right size avoids teaching incidental footprint asymmetry.
    for side in ("a", "b"):
        radii = [
            p[side]["automatic_boundary"]["radius_mm"] / meshes[i + (side == "b")].scale
            for i in eligible
            for p in config["joints"][i]["patches"]
            if p["kind"] == "facet"
        ]
        for role in roles[1:]:
            role["radius_fraction"][side] = float(np.median(radii))
    return dict(roles=roles, supporting_joints=len(eligible), facet_pairing_votes=votes.tolist())


def role_points(mesh, role, side):
    """Local coordinates of the landmarks defining ``role`` on ``side`` of a mesh."""
    points = landmark_map(mesh)
    if any(key not in points for key in role[side]):
        raise ValueError(f"{mesh.name}: missing learned contact landmarks")
    return np.array([points[key] for key in role[side]])


def role_frame(mesh, model):
    """Anatomical axes for a bone from its accepted landmark plane and learned role landmarks."""
    cached = getattr(mesh, "_automatic_bilateral_plane", None)
    if cached is None or getattr(mesh, "_landmark_symmetry_status", {}).get("status") != "landmark_plane":
        raise ValueError(f"{mesh.name}: accepted landmark plane required for recovery")
    lateral, point, _ = cached
    centrum = model["roles"][0]
    anterior = role_points(mesh, centrum, "b").mean(0)
    posterior = role_points(mesh, centrum, "a").mean(0)
    z = posterior - anterior
    z = n.unit(z - lateral * (z @ lateral))
    dorsal_hint = (
        np.mean(
            [role_points(mesh, role, side).mean(0) for role in model["roles"][1:] for side in ("a", "b")],
            axis=0,
        )
        - (anterior + posterior) / 2
    )
    dorsal = n.unit(np.cross(z, lateral))
    if dorsal @ dorsal_hint < 0:
        lateral, dorsal = -lateral, -dorsal
    center = (anterior + posterior) / 2
    center -= lateral * ((center - point) @ lateral)
    return dict(axes=np.column_stack([lateral, dorsal, z]).tolist(), center=(center + mesh.origin).tolist())


def seed_patch(mesh, role, side, frame):
    """Oriented geodesic growth supplies a provisional seed, never final labels."""
    points = role_points(mesh, role, side)
    Q, _ = surface_queries(mesh, np.arange(len(mesh.faces)), points.mean(0)[None])
    axis = np.asarray(frame["axes"]) @ np.asarray(role["normal_in_frame"][side])
    radius = role["radius_fraction"][side] * mesh.scale
    distance = np.linalg.norm(mesh.centers - Q[0], axis=1)
    normals = mesh.smooth_normals()
    cone = math.cos(math.radians(80 if role["kind"] == "centrum" else 55))
    seeds = np.flatnonzero((distance < 0.25 * mesh.scale) & (normals @ axis > cone))
    if not len(seeds):
        raise ValueError("No oriented surface near learned contact landmarks")
    seed = seeds[np.argmin(distance[seeds])]
    geodesic = dijkstra(mesh.adjacency(), directed=False, indices=int(seed), limit=radius)
    candidates = np.flatnonzero((geodesic <= radius) & (normals @ axis > cone))
    if seed not in candidates:
        raise ValueError("Landmark seed is outside the oriented contact component")
    _, labels = connected_components(mesh.adjacency()[candidates][:, candidates], directed=False)
    ids = candidates[labels == labels[np.flatnonzero(candidates == seed)[0]]]
    if len(ids) < 3:
        raise ValueError("Learned seed patch is too small")
    spec = boundary_spec(mesh, ids)
    anchor, _ = surface_queries(mesh, ids, points.mean(0)[None])
    spec["inferred_anchor"] = (anchor[0] + mesh.origin).tolist()
    return spec


def assign_contact_roles(
    distances, names, maximum_distance=ROLE_IDENTITY_MAX_DISTANCE, minimum_margin=ROLE_IDENTITY_MIN_MARGIN
):
    """Resolve all roles together; quantify ambiguity between feasible assignments.

    The distance bound is unchanged. The ambiguity margin is the mean increase
    per reassigned role, so a forced one-to-one match is not rejected merely
    because two rows have the same independent nearest candidate.
    """
    distances = np.asarray(distances, float)
    if distances.ndim != 2 or distances.shape[0] != len(names) or not np.isfinite(distances).all():
        raise ValueError("Invalid contact identity distance matrix")
    count, candidates = distances.shape
    if candidates < count:
        raise ValueError(
            f"Only {candidates} candidate contact patches for {count} anatomical roles; "
            "contact regions may be missing or merged"
        )
    feasible = []
    rows = np.arange(count)
    for choice in permutations(range(candidates), count):
        costs = distances[rows, choice]
        if np.all(costs <= maximum_distance):
            feasible.append((float(costs.sum()), choice))
    if not feasible:
        closest = ", ".join(f"{name}={value:.4g}" for name, value in zip(names, distances.min(axis=1)))
        raise ValueError(
            f"No distinct contact assignment satisfies landmark distance <= {maximum_distance:g} "
            f"of bone scale ({candidates} candidates; closest distances: {closest})"
        )
    feasible.sort()
    score, choice = feasible[0]
    chosen = np.array(choice, int)
    if len(feasible) > 1:
        margin = min(
            (other_score - score) / np.count_nonzero(chosen != other) for other_score, other in feasible[1:]
        )
        if margin < minimum_margin:
            raise ValueError(
                f"Contact identity has near-equivalent one-to-one assignments "
                f"(margin {margin:.4g}; required {minimum_margin:g})"
            )
    return chosen


def split_contact_components(mesh, mask, model, side, diagnostics=None):
    """Partition merged opposing regions using specimen-learned landmark seeds.

    Only triangles in the original proximity/normal mask are eligible. A
    geodesic Voronoi partition separates distinct supported roles inside each
    connected component; it cannot create contact outside the measured mask.
    Pairing, identity, bilateral geometry and fitting still run afterwards.
    """
    raw = n.candidate_components(mesh, mask)
    result = []
    for ids in raw:
        roles, seeds = [], []
        for role in model["roles"]:
            points = role_points(mesh, role, side)
            closest, _ = surface_queries(mesh, ids, points)
            error = float(np.mean(np.linalg.norm(points - closest, axis=1)) / mesh.scale)
            if error <= ROLE_IDENTITY_MAX_DISTANCE:
                anchor, _ = surface_queries(mesh, ids, points.mean(axis=0)[None])
                seeds.append(int(np.argmin(np.linalg.norm(mesh.centers[ids] - anchor[0], axis=1))))
                roles.append(role["name"])
        if len(roles) < 2:
            result.append(ids)
            continue
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{mesh.name}: merged contacts have coincident landmark seeds")
        graph = mesh.adjacency()[ids][:, ids]
        distance = dijkstra(graph, directed=False, indices=seeds)
        labels = np.argmin(distance, axis=0)
        parts = [ids[labels == i] for i in range(len(seeds))]
        accepted = [
            len(part) >= PARTITION_MIN_FACES
            and mesh.areas[part].sum() > PARTITION_MIN_AREA_FRACTION * mesh.scale**2
            for part in parts
        ]
        result.extend(part for part, valid in zip(parts, accepted) if valid)
        if diagnostics is not None:
            diagnostics.append(
                dict(
                    mesh=mesh.name,
                    side=side,
                    method="landmark_seeded_geodesic_partition_of_opposing_component",
                    roles=roles,
                    seed_faces=ids[seeds].tolist(),
                    input_face_count=len(ids),
                    part_face_counts=[len(part) for part in parts],
                    parts_retained=[bool(x) for x in accepted],
                    part_areas_mm2=[float(mesh.areas[part].sum()) for part in parts],
                )
            )
    return sorted(result, key=lambda ids: -mesh.areas[ids].sum())


def identify_candidates(a, b, candidates, model):
    """Assign learned roles to candidates by landmark distance; return chosen candidates and distances."""
    distances = np.empty((len(model["roles"]), len(candidates)))
    for r, role in enumerate(model["roles"]):
        for k, candidate in enumerate(candidates):
            errors = []
            for side, mesh in (("a", a), ("b", b)):
                points = role_points(mesh, role, side)
                Q, _ = surface_queries(mesh, candidate["i" + side], points)
                errors.append(float(np.mean(np.linalg.norm(Q - points, axis=1)) / mesh.scale))
            distances[r, k] = max(errors)
    chosen = assign_contact_roles(distances, [role["name"] for role in model["roles"]])
    return [candidates[k] for k in chosen], distances[np.arange(len(chosen)), chosen].tolist()
