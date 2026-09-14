"""Regression: extra near-midline contacts must not replace a bilateral facet."""

import numpy as np
import pytest

from neutral_pose import anatomy
from neutral_pose import core as n
from test_neutral_pose import wedge
from test_realistic import marching_cubes_vertebra, realistic_config, realistic_options


def contact_fixture():
    meshes = [
        marching_cubes_vertebra(
            f"v{i + 1:02d}", n.rigid(translation=[0.0, 0.0, 1.02 * i]), voxel=0.055, noise=0.0, seed=i
        )
        for i in range(2)
    ]
    joint = n.Joint(*meshes, realistic_config(), realistic_options())
    candidates = []
    for pair in joint.pairs:
        candidates.append({"ia": pair.a.ids, "ib": pair.b.ids, "na": pair.a.normal, "nb": pair.b.normal})
    extra = {}
    for side, index, mesh in zip(("a", "b"), (0, 1), meshes):
        points = mesh.centers + mesh.origin - [0.0, 0.0, 1.02 * index]
        sign = 1 if index == 0 else -1
        ids = np.where(
            (abs(points[:, 0]) < 0.18)
            & (points[:, 1] > 0.5)
            & (sign * points[:, 2] > 0.3)
            & (sign * mesh.normals[:, 2] > 0.5)
        )[0]
        assert len(ids) > 3
        extra["i" + side] = ids
        extra["n" + side] = n.unit(np.average(mesh.normals[ids], axis=0, weights=mesh.areas[ids]))
    return meshes, [candidates[0], extra, candidates[1], candidates[2]]


@pytest.mark.parametrize("permutation", [[0, 1, 2, 3], [3, 1, 0, 2]])
def test_extra_central_contact_does_not_replace_a_bilateral_facet(permutation):
    meshes, source = contact_fixture()
    candidates = [source[i] for i in permutation]
    selected, evidence = anatomy.choose_bilateral_contacts(*meshes, candidates, permutation.index(0))
    assert {permutation[i] for i in selected} == {2, 3}
    for mesh in meshes:
        normal, point, _ = mesh._automatic_bilateral_plane
        # The synthetic surface has the independently defined plane x=0.
        assert abs(normal[0]) > 0.999
        assert abs((point + mesh.origin)[0]) < 0.003
    assert all(x["supported"] for x in evidence["sides"])


def test_missing_opposite_contact_is_a_clear_failure():
    meshes, candidates = contact_fixture()
    with pytest.raises(ValueError, match="no two candidate contacts form a supported bilateral pair"):
        anatomy.choose_bilateral_contacts(*meshes, candidates[:3], 0)


def test_multiple_equally_good_intrinsic_planes_are_not_silently_selected():
    mesh, _ = wedge("symmetric_box")
    with pytest.raises(ValueError, match="multiple distinct intrinsic bilateral planes"):
        anatomy.intrinsic_symmetry_plane(mesh, np.array([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("overlap", [True, False])
def test_unequal_patch_footprints_require_actual_mirrored_overlap(overlap):
    meshes = [wedge(f"v{i}", world=n.rigid(translation=[0.0, 0.0, 1.02 * i]))[0] for i in range(2)]
    candidates = [{}, {}, {}]
    for side, index, mesh in zip(("a", "b"), (0, 1), meshes):
        sign = 1 if index == 0 else -1
        X = mesh.centers
        end = sign * mesh.normals[:, 2] > 0.99
        masks = [end, end & (X[:, 0] < 0) & (X[:, 1] > 0), end & (X[:, 0] > 0)]
        if not overlap:
            masks[2] &= X[:, 1] < -0.3
        for candidate, mask in zip(candidates, masks):
            ids = np.flatnonzero(mask)
            candidate["i" + side] = ids
            candidate["n" + side] = n.unit(np.average(mesh.normals[ids], axis=0, weights=mesh.areas[ids]))
        mesh._automatic_bilateral_plane = (np.array([1.0, 0.0, 0.0]), np.zeros(3), {"review_reasons": []})
    if not overlap:
        with pytest.raises(ValueError, match="no two candidate contacts form a supported bilateral pair"):
            anatomy.choose_bilateral_contacts(*meshes, candidates, 0)
        return
    selected, evidence = anatomy.choose_bilateral_contacts(*meshes, candidates, 0)
    assert selected == {1, 2}
    for side in evidence["sides"]:
        assert side["reflection_error_fraction"] > 0.2
        assert side["partial_patch_overlap"]["supported"]
