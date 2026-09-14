"""Automatic discovery of opposing contact surfaces between adjacent bones.

Given two meshes in an approximate articulation, find connected opposing
regions, estimate their spacing from the input arrangement, and classify them
as one centrum contact plus a bilateral facet pair. Optionally use a
specimen-learned landmark model to partition merged regions and to identify
roles; that model only seeds the search and never fixes the final patches.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from . import anatomy, contacts
from . import core as n
from .surfaces import boundary_spec, landmark_support, surface_queries, surface_roughness

LOG = logging.getLogger(__name__)

# Contact-discovery thresholds (dimensionless unless stated). See README
# "Automatic method" for the rationale; these are model choices, not options.

#: Two regions oppose each other only if their area-weighted mean normals have
#: a dot product below this value after alignment.
OPPOSING_NORMAL_COSINE = -0.5
#: Minimum area of each side of a candidate contact, as a fraction of scale².
MINIMUM_CONTACT_AREA_FRACTION = 0.002
#: Candidates smaller than this fraction of the largest paired area are
#: treated as peripheral and dropped.
PERIPHERAL_AREA_RATIO = 0.08
#: A candidate is accepted as the centrum only if its normal aligns with the
#: joint's longitudinal direction at least this well (absolute cosine).
CENTRUM_AXIAL_ALIGNMENT = 0.65
#: The fitting dead zone from observed gap scatter is limited to this fraction
#: of scale, and to this fraction of the robust scatter (MAD) itself.
GAP_TOLERANCE_SCALE_CAP = 0.005
GAP_TOLERANCE_SCATTER_FRACTION = 0.25
#: The surface-model residual noise allowance is capped at this fraction of scale.
NOISE_ALLOWANCE_SCALE_CAP = 0.01


def infer_joint(a, b, index, H=None, learned_model=None, *, split_contacts=False):
    """Discover, classify and describe the contact patches between adjacent bones ``a`` and ``b``.

    ``H`` is an optional provisional alignment of ``b`` into ``a``'s local frame;
    ``learned_model`` enables identity assignment and, with ``split_contacts``,
    landmark-seeded partition of merged regions. Returns ``(definition, evidence)``.
    """
    scale = math.sqrt(a.scale * b.scale)
    provisional = H is not None
    H = n.rigid(translation=b.origin - a.origin) if H is None else n.validate_rigid(H)
    partitions = []
    try:
        if split_contacts:
            if learned_model is None:
                raise ValueError("Landmark-supported contact partitioning requires a learned model.")

            def splitter(mesh, mask):
                return contacts.split_contact_components(
                    mesh, mask, learned_model, "a" if mesh is a else "b", partitions
                )

            specs = n.automatic_patch_specs(a, b, H, scale, component_splitter=splitter)
        else:
            specs = n.automatic_patch_specs(a, b, H, scale)
    except ValueError as error:
        raise ValueError(f"{a.name}/{b.name}: {error}") from error
    candidates = []
    for spec in specs:
        ia, ib = np.asarray(spec["a"]["faces"], int), np.asarray(spec["b"]["faces"], int)
        area_a, area_b = a.areas[ia].sum(), b.areas[ib].sum()
        na = n.unit(np.average(a.normals[ia], axis=0, weights=a.areas[ia]))
        nb = n.unit(np.average(b.normals[ib], axis=0, weights=b.areas[ib]))
        if (
            na @ (H[:3, :3] @ nb) > OPPOSING_NORMAL_COSINE
            or min(area_a, area_b) < MINIMUM_CONTACT_AREA_FRACTION * scale**2
        ):
            continue
        candidates.append(
            {"ia": ia, "ib": ib, "na": na, "nb": nb, "area_mm2": float(math.sqrt(area_a * area_b))}
        )
    if not candidates:
        raise ValueError(f"{a.name}/{b.name}: no sufficiently opposing contact patches.")
    # Reject tiny peripheral contacts relative to the strongest paired surface.
    largest = max(c["area_mm2"] for c in candidates)
    candidates = [c for c in candidates if c["area_mm2"] >= PERIPHERAL_AREA_RATIO * largest]
    identity_distances = None
    if learned_model is not None:
        candidates, identity_distances = contacts.identify_candidates(a, b, candidates, learned_model)
    tangent = n.unit(H[:3, 3])
    axial = [abs(float(c["na"] @ tangent)) for c in candidates]
    centrum = (
        0
        if learned_model is not None
        else int(np.argmax(axial))
        if len(candidates) >= 3 and max(axial) > CENTRUM_AXIAL_ALIGNMENT
        else None
    )
    facets, selection = (
        anatomy.choose_bilateral_contacts(a, b, candidates, centrum) if centrum is not None else (set(), None)
    )
    pairs, evidence, roughness = [], [], []
    facet_number = 0
    for k, c in enumerate(candidates):
        ia, ib = c["ia"], c["ib"]
        a_points = a.sample(384, ia, seed=310 + index).points
        b_points = b.sample(384, ib, seed=613 + index).points
        in_b = n.transform(n.inverse(H), a_points)
        in_a = n.transform(H, b_points)
        qb, nb = surface_queries(b, ib, in_b)
        qa, na = surface_queries(a, ia, in_a)
        da = np.sum((in_b - qb) * nb, axis=1)
        db = np.sum((in_a - qa) * na, axis=1)
        gap_values = np.r_[da, db]
        gap = max(0.0, float(np.median(gap_values)))
        scatter = float(1.4826 * np.median(np.abs(gap_values - np.median(gap_values))))
        ra, rb = surface_roughness(a, ia), surface_roughness(b, ib)
        if k == centrum:
            kind, name = "centrum", "inferred_centrum"
        elif k in facets:
            facet_number += 1
            kind, name = "facet", f"inferred_facet_{facet_number}"
            roughness.extend([ra, rb])
        else:
            kind, name = "unclassified", f"contact_{k + 1}"
        # Observed dispersion is not all cartilage variability; limit the fitting
        # dead zone and use it primarily as uncertainty evidence.
        tolerance = min(
            GAP_TOLERANCE_SCALE_CAP * scale, max(1e-6 * scale, GAP_TOLERANCE_SCATTER_FRACTION * scatter)
        )
        pairs.append(
            {
                "name": name,
                "kind": kind,
                "a": boundary_spec(a, ia),
                "b": boundary_spec(b, ib),
                "gap_fraction": gap / scale,
                "gap_tolerance_fraction": tolerance / scale,
            }
        )
        evidence.append(
            {
                "name": name,
                "inferred_kind": kind,
                "kind_is_inferred": True,
                "area_mm2": [float(a.areas[ia].sum()), float(b.areas[ib].sum())],
                "mean_normal_dot": float(c["na"] @ (H[:3, :3] @ c["nb"])),
                "axial_normal_alignment": axial[k],
                "spacing_source": "median_signed_normal_separation_in_input_arrangement",
                "estimated_gap_mm": gap,
                "gap_mad_mm": scatter,
                "input_signed_gap_quantiles_mm": np.quantile(gap_values, [0.1, 0.5, 0.9]).tolist(),
                "input_negative_gap_fraction": float(np.mean(gap_values < 0)),
                "surface_model_residual_mm": [ra, rb],
                "landmark_support_a": landmark_support(a, ia),
                "landmark_support_b": landmark_support(b, ib),
            }
        )
        if provisional:
            evidence[-1]["provisional_signed_gap_quantiles_mm"] = evidence[-1].pop(
                "input_signed_gap_quantiles_mm"
            )
            evidence[-1]["provisional_negative_gap_fraction"] = evidence[-1].pop(
                "input_negative_gap_fraction"
            )
            evidence[-1]["spacing_source"] = "signed_normal_separation_in_provisional_alignment"
    noise = min(NOISE_ALLOWANCE_SCALE_CAP * scale, float(np.median(roughness))) if roughness else 0.0
    LOG.info(
        "Found %d paired patches; estimated clearances %s mm; surface residual allowance %.4g mm",
        len(pairs),
        ", ".join(f"{p['gap_fraction'] * scale:.4g}" for p in pairs),
        noise,
    )
    definition = {
        "a": a.name,
        "b": b.name,
        "patches": pairs,
        "noise_floor_mm": noise,
        "noise_floor_source": "surface_model_residual",
    }
    return definition, {
        "a": a.name,
        "b": b.name,
        "local_scale_mm": scale,
        "landmark_identity_distances_fraction": identity_distances,
        "contact_component_partitions": partitions,
        "bilateral_contact_selection": selection,
        "estimated_noise_allowance_mm": noise,
        "noise_note": "Quadratic surface-model residual; may include real surface texture and model error.",
        "patches": evidence,
    }
