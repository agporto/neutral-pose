"""Landmark reflection inference against independent, known correspondences."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from neutral_pose import anatomy
from neutral_pose import core as n
from neutral_pose import landmarks as lm
from test_neutral_pose import wedge


def known_landmarks():
    rng = np.random.default_rng(891)
    points, permutation = [], []
    for i in range(11):
        y, z = rng.uniform(-0.58, 0.58), rng.uniform(-0.44, 0.44)
        points.extend([[-0.65, y, z], [0.65, y, z]])
        permutation.extend([2 * i + 1, 2 * i])
    for i, z in enumerate([-0.43, -0.31, -0.11, 0.07, 0.26, 0.45]):
        points.append([0.0, 0.65 if i % 3 else -0.65, z])
        permutation.append(len(permutation))
    return np.array(points), np.array(permutation)


def landmark_meshes(count=3):
    P, permutation = known_landmarks()
    meshes = []
    for i in range(count):
        world = n.rigid(translation=[0.0, 0.0, 1.02 * i])
        mesh, _ = wedge(f"v{i + 1:02d}", world=world)
        mesh.landmarks = n.Landmarks(
            n.transform(world, P), [f"export_{i}-{k + 1}" for k in range(len(P))], "LPS"
        )
        meshes.append(mesh)
    return meshes, permutation


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_landmarks_alone_recover_known_plane_in_any_world_coordinates(scale):
    P, permutation = known_landmarks()
    R = Rotation.from_euler("xyz", [32.0, -61.0, 17.0], degrees=True).as_matrix()
    world = n.rigid(R, [13.0, -7.0, 9.0])
    observed = n.transform(world, P * scale)
    candidates = lm.discover_candidates(observed, R[:, 2])
    assert candidates and np.array_equal(candidates[0]["permutation"], permutation)
    normal, center, evidence = lm.fit_paired_plane(observed, permutation, scale, candidates[0]["normal"])
    assert lm.angle(normal, R[:, 0]) < 1e-6
    assert abs((center - world[:3, 3]) @ normal) < 1e-8 * scale
    assert evidence["converged"]


def test_pairing_is_independent_of_landmark_row_order():
    P, permutation = known_landmarks()
    order = np.random.default_rng(12).permutation(len(P))
    inverse = np.argsort(order)
    candidates = lm.discover_candidates(P[order], np.array([0.0, 0.0, 1.0]))
    assert np.array_equal(candidates[0]["permutation"], inverse[permutation[order]])


def test_one_misplaced_annotation_is_downweighted_as_a_complete_pair():
    P, permutation = known_landmarks()
    P[0] += [0.35, 0.15, -0.1]
    normal, center, fit = lm.fit_paired_plane(P, permutation, 1.0, [1.0, 0.02, -0.01])
    assert lm.angle(normal, np.array([1.0, 0.0, 0.0])) < 0.1
    assert abs(center @ normal) < 0.001
    assert fit["weights"][0] == pytest.approx(fit["weights"][1])
    assert set(fit["downweighted_indices"]) == {0, 1}


def test_non_involutive_correspondence_is_rejected():
    P, permutation = known_landmarks()
    permutation[:3] = [1, 2, 0]
    with pytest.raises(ValueError, match="invalid_bilateral_correspondence"):
        lm.fit_paired_plane(P, permutation)


def test_consensus_handles_prefixes_reordered_rows_and_partial_files():
    meshes, _ = landmark_meshes()
    # Delete one paired point: its surviving partner must be omitted as well.
    meshes[2].landmarks.points = meshes[2].landmarks.points[1:]
    meshes[2].landmarks.labels = meshes[2].landmarks.labels[1:]
    order = np.random.default_rng(4).permutation(28)
    meshes[1].landmarks.points = meshes[1].landmarks.points[order]
    meshes[1].landmarks.labels = [meshes[1].landmarks.labels[i] for i in order]
    result = lm.prepare_landmark_planes(meshes)
    assert result["landmark_plane_count"] == 3
    assert result["surface_fallback_count"] == 0
    assert len(result["schemas"]) == 1
    assert result["schemas"][0]["agreement_fraction"] == 1.0
    last = result["bones"][meshes[2].name]["quality"]
    assert last["omitted_unpaired_landmarks"] == ["export_2-2"]
    assert len(last["landmark_pairs"]) == 10
    for mesh in meshes:
        normal, point, quality = mesh._automatic_bilateral_plane
        assert lm.angle(normal, np.array([1.0, 0.0, 0.0])) < 1e-6
        assert not quality["surface_check_refines_plane"]


def test_accepted_landmark_plane_does_not_run_surface_plane_optimization(monkeypatch):
    meshes, _ = landmark_meshes(2)
    tilt = Rotation.from_euler("y", 2.0, degrees=True).as_matrix()
    for mesh in meshes:
        mesh.landmarks.points = (mesh.landmarks.points - mesh.origin) @ tilt.T + mesh.origin
    result = lm.prepare_landmark_planes(meshes)
    assert result["landmark_plane_count"] == 2

    def forbidden(*args, **kwargs):
        pytest.fail("accepted landmarks must not be replaced by surface optimization")

    monkeypatch.setattr(anatomy, "symmetry_plane", forbidden)
    normal, _, evidence = anatomy.intrinsic_symmetry_plane(meshes[0], np.array([0.0, 0.0, 1.0]))
    assert evidence["source"] == "robust_paired_landmarks_with_surface_check"
    # The mesh has exact reflection symmetry at x=0. The accepted landmark
    # plane is deliberately tilted and must retain its measured orientation.
    assert lm.angle(normal, tilt[:, 0]) < 1e-6
    assert lm.angle(normal, np.array([1.0, 0.0, 0.0])) == pytest.approx(2.0, abs=1e-6)


def test_landmark_plane_that_disagrees_with_the_mesh_has_an_explicit_fallback():
    meshes, _ = landmark_meshes(2)
    for mesh in meshes:
        mesh.landmarks.points += [0.35, 0.0, 0.0]
    result = lm.prepare_landmark_planes(meshes)
    assert result["landmark_plane_count"] == 0
    assert result["surface_fallback_count"] == 2
    assert all("landmark_plane_fails_surface_check" in x["reasons"] for x in result["bones"].values())
    assert all(not hasattr(mesh, "_automatic_bilateral_plane") for mesh in meshes)


def test_ambiguous_landmark_pattern_has_an_explicit_fallback():
    P = np.array([[x, y, z] for x in [-1.0, 1.0] for y in [-1.0, 1.0] for z in [-1.0, 1.0]])
    meshes = [
        SimpleNamespace(
            name=f"v{i}",
            origin=np.array([0.0, 0.0, 3.0 * i]),
            scale=1.0,
            landmarks=n.Landmarks(P + [0.0, 0.0, 3.0 * i], [f"F-{k + 1}" for k in range(8)], "RAS"),
        )
        for i in range(2)
    ]
    result = lm.prepare_landmark_planes(meshes)
    assert result["landmark_plane_count"] == 0
    assert result["surface_fallback_count"] == 2
    assert not result["schemas"]
    assert all("unresolved_landmark_pairing_consensus" in x["reasons"] for x in result["bones"].values())


def test_missing_landmarks_clear_any_previously_cached_plane():
    meshes, _ = landmark_meshes(2)
    lm.prepare_landmark_planes(meshes)
    for mesh in meshes:
        mesh.landmarks = None
    result = lm.prepare_landmark_planes(meshes)
    assert result["surface_fallback_count"] == 2
    assert all(not hasattr(mesh, "_automatic_bilateral_plane") for mesh in meshes)
    assert all(x["reasons"] == ["no_landmarks"] for x in result["bones"].values())


def test_semantic_label_prefixes_are_not_collapsed_into_numbered_roles():
    keys, method = lm.landmark_keys(["left-1", "right-1", "midline-1"])
    assert keys == ["left-1", "right-1", "midline-1"]
    assert method == "exact_labels"
    with pytest.raises(ValueError, match="duplicate_or_empty"):
        lm.landmark_keys(["F-01", "F-1"])
