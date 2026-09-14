"""Checked recovery of joints whose automatic contact discovery was incomplete.

An incomplete joint is re-seeded from the specimen-learned landmark model,
rediscovered on the actual opposing surfaces, and accepted only if the
recovered contacts pass the same identity, bilateral and support checks as
originally complete joints. Failed joints are deferred and retried after
neighbouring contacts have been recovered; the learned model and spacing
statistics stay fixed throughout.
"""

from __future__ import annotations

import copy
import logging
import math

import numpy as np

from . import anatomy, contacts
from . import core as n
from .discovery import infer_joint
from .support import contact_support, refine_dense_feasibility

LOG = logging.getLogger(__name__)


def recover_joint(meshes, config, inference, index, model, statistics, options):
    """Re-seed and rediscover contacts for one incomplete joint; return ``(definition, evidence)``."""
    a, b = meshes[index : index + 2]
    covered = {
        definition[side]
        for definition in config["joints"]
        if contacts.complete(definition)
        for side in ("a", "b")
    }
    known_frames = anatomy.infer_frames([m for m in meshes if m.name in covered], config)
    frames = {
        m.name: known_frames[m.name] if m.name in known_frames else contacts.role_frame(m, model)
        for m in (a, b)
    }
    seed_definition = dict(a=a.name, b=b.name, patches=[], noise_floor_mm=0.0)
    for role in model["roles"]:
        seed_definition["patches"].append(
            dict(
                name=role["name"],
                kind=role["kind"],
                a=contacts.seed_patch(a, role, "a", frames[a.name]),
                b=contacts.seed_patch(b, role, "b", frames[b.name]),
                gap_fraction=statistics[role["kind"]]["median"],
            )
        )
    seed_config = dict(
        mesh_coordinate_system=config["mesh_coordinate_system"],
        neutral_frames=frames,
        joints=[seed_definition],
    )
    seed_joint = n.Joint(a, b, seed_config, options, index)
    H = seed_joint.landmark_seed()
    if H is None:
        raise ValueError("Learned contact geometry cannot identify a rigid alignment")
    seed = n.optimize_joint(seed_joint, H).transform
    viable, failures = [], []
    seed_trials = []
    for offset in (-12.0, 0.0, 12.0):
        q = seed_joint.neutral_plane.encode(seed, seed_joint.scale)
        q[0] += math.radians(offset)
        seed_trials.append(("landmark_seed", offset, seed_joint.neutral_plane.decode(q, seed_joint.scale)))
    # Preserve successful existing recovery. Partition only when all ordinary
    # surface trials fail; also inspect the actual input arrangement in that
    # fallback, where a thin connection may be the sole reason for failure.
    refined_trials = []
    for source, base, offsets in (
        ("projected_input", seed_joint.initial, (-12.0, -6.0, 0.0, 6.0, 12.0)),
        ("landmark_seed", seed, (-6.0, 6.0)),
    ):
        for offset in offsets:
            q = seed_joint.neutral_plane.encode(base, seed_joint.scale)
            q[0] += math.radians(offset)
            refined_trials.append((source, offset, seed_joint.neutral_plane.decode(q, seed_joint.scale)))
    # A coarse angle grid can miss supported opposing regions, particularly
    # when the landmark seed changes in-plane translation. Refine the same
    # +/-12 degree window around both the input and optimized seed only after
    # earlier recovery attempts fail. All geometric acceptance bounds remain.
    stages = [
        (False, seed_trials),
        (True, [("input", None, n.rigid(translation=b.origin - a.origin))] + seed_trials),
        (True, refined_trials),
    ]
    for partition, trials in stages:
        for source, offset, trial in trials:
            try:
                definition, receipt = infer_joint(
                    a, b, index, H=trial, learned_model=model, split_contacts=partition
                )
                for patch, evidence in zip(definition["patches"], receipt["patches"]):
                    gap = statistics[patch["kind"]]["median"]
                    evidence["provisional_alignment_gap_mm"] = evidence["estimated_gap_mm"]
                    evidence["estimated_gap_mm"] = gap * receipt["local_scale_mm"]
                    evidence["spacing_source"] = "complete_joint_specimen_median_for_recovered_contact"
                    patch["gap_fraction"] = gap
                trial_config = copy.deepcopy(config)
                trial_config["joints"][index] = definition
                # Other unresolved joints may still contain unframed bones.
                # Their recovery must not block a supported trial for this pair.
                framed = covered | {a.name, b.name}
                trial_config["neutral_frames"] = anatomy.infer_frames(
                    [mesh for mesh in meshes if mesh.name in framed], trial_config
                )
                joint = n.Joint(a, b, trial_config, options, index)
                candidate, _ = n.fit_joint(joint)
                fitted, dense = refine_dense_feasibility(joint, candidate.transform)
                metrics = joint.metrics(fitted, final=True)
                support = contact_support(metrics)
                if support["unsupported"]:
                    raise ValueError("Re-discovered contacts cannot simultaneously support an articulation")
                depth = metrics["max_sampled_penetration"]
                info = dict(
                    seed_source=source,
                    seed_offset_deg=offset,
                    partitioned_contacts=partition,
                    fitted_contact_support=support,
                    dense_penetration_mm=depth,
                    provisional_transform_b_local_to_a_local=trial.tolist(),
                    dense_refinement=dense,
                )
                # Contact completeness precedes mean coverage. Dense feasibility
                # is independently enforced during the final fit.
                rank = (
                    depth is not None and depth > joint.penetration_tolerance * joint.scale,
                    -support["minimum"],
                    -support["mean"],
                    float(np.sum(joint.residual(fitted) ** 2)),
                )
                viable.append((rank, definition, receipt, info))
            except ValueError as error:
                failures.append(
                    dict(
                        seed_source=source,
                        seed_offset_deg=offset,
                        partitioned_contacts=partition,
                        error=str(error),
                    )
                )
        if viable:
            break
    if not viable:
        raise ValueError(f"{a.name}/{b.name}: contact recovery failed: {failures}")
    _, definition, receipt, selected = min(viable, key=lambda x: x[0])
    definition["contact_recovery"] = dict(
        method="landmark_consensus_then_surface_rediscovery",
        supporting_joints=model["supporting_joints"],
        selected_trial=selected,
        failed_trials=failures,
        viable_trials=len(viable),
    )
    receipt["contact_recovery"] = copy.deepcopy(definition["contact_recovery"])
    receipt["original_discovery"] = copy.deepcopy(inference["joints"][index])
    return definition, receipt


def complete_contacts(meshes, config, inference, options):
    """Adjust spacing and recover every incomplete joint in ``config`` in place.

    Raises ``ValueError`` if any joint remains incomplete after no further progress is possible.
    """
    indices = [
        i
        for i, d in enumerate(config["joints"])
        if contacts.complete(d) and inference["joints"][i].get("bilateral_contact_selection") is not None
    ]
    statistics = contacts.spacing_statistics(config, inference, indices)
    changes = contacts.adjust_spacing(config, inference, statistics, indices)
    missing = [i for i, d in enumerate(config["joints"]) if not contacts.complete(d)]
    model = None
    recovery_passes = []
    if missing:
        model = contacts.learn_model(meshes, config, inference, indices)
        pending = list(missing)
        while pending:
            recovered, deferred = [], []
            for i in pending:
                LOG.info(
                    "Recovering incomplete contact %d/%d using %d complete joints",
                    i + 1,
                    len(config["joints"]),
                    model["supporting_joints"],
                )
                try:
                    definition, receipt = recover_joint(
                        meshes, config, inference, i, model, statistics, options
                    )
                except ValueError as error:
                    deferred.append(dict(joint_index=i, error=str(error)))
                    LOG.info("Deferring contact %d until other incomplete contacts have been resolved", i + 1)
                else:
                    config["joints"][i], inference["joints"][i] = definition, receipt
                    recovered.append(i)
            recovery_passes.append(dict(recovered_joint_indices=recovered, deferred=deferred))
            if not recovered:
                raise ValueError(
                    "Contact recovery made no progress for the remaining joints: "
                    + "; ".join(item["error"] for item in deferred)
                )
            pending = [item["joint_index"] for item in deferred]
    if not all(contacts.complete(d) for d in config["joints"]):
        raise ValueError("Incomplete anatomical neutral reference after contact recovery")
    config["automatic_contact_policy"] = contacts.POLICY
    inference["contact_consensus"] = dict(
        spacing_statistics=statistics,
        spacing_changes=changes,
        landmark_model=model,
        recovered_joints=missing,
        recovery_passes=recovery_passes,
    )
