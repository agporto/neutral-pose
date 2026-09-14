"""Exact reuse must preserve residuals, sparse dependencies and collision terms."""

from types import SimpleNamespace

import numpy as np
import pytest

from neutral_pose import core as n
from test_neutral_pose import fast_options, fixture_chain, wedge
from test_realistic import marching_cubes_vertebra


class NoReuse:
    def __init__(self, *args):
        pass

    def get(self, key):
        return None

    def put(self, key, value):
        pass


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("open_mesh", [False, True])
def test_refinement_reuse_preserves_every_residual_and_sparse_dependency(monkeypatch, constrained, open_mesh):
    meshes, cfg, _, _ = fixture_chain(count=4)
    if constrained:
        cfg["neutral_frames"] = {
            m.name: {"axes": np.eye(3).tolist(), "center": m.origin.tolist()} for m in meshes
        }
    if open_mesh:
        meshes[1].closed = False
    options = fast_options(smoothness_weight=0.17)
    joints = [n.Joint(a, b, cfg, options, i) for i, (a, b) in enumerate(zip(meshes, meshes[1:]))]
    # Deliberately overlapping nonadjacent bones exercise active collision terms.
    candidates = [
        n.Candidate(n.rigid(translation=[0.0, 0.03, 0.08]), 0.0, True, 1, "fixture", False, [])
        for joint in joints
    ]
    width = 3 if constrained else 6
    observed, counts = [], []
    original = n.Joint.residual

    def counted(self, *args, **kw):
        counts[-1] += 1
        return original(self, *args, **kw)

    def exercise(fun, x0, **kwargs):
        points = [x0.copy()]
        for bone in range(3):
            for axis in range(width):
                x = x0.copy()
                x[bone * width + axis] = 0.017
                points.extend([x, x0.copy()])
        # Exercise eviction, then revisit earlier poses. No approximate keys.
        for k in range(24):
            x = x0.copy()
            x[width - 1] = 0.001 * k
            points.append(x)
        points.extend([points[1].copy(), x0.copy()])
        outputs = []
        for x in points:
            r = fun(x)
            outputs.append(r.copy())
            r[:] = 123456.0  # A caller cannot corrupt future cached evaluations.
        observed.append((outputs, kwargs["jac_sparsity"].toarray(), kwargs["bounds"]))
        return SimpleNamespace(
            x=x0, fun=outputs[0].copy(), success=True, nfev=len(points), message="controlled evaluations"
        )

    monkeypatch.setattr(n.Joint, "residual", counted)
    monkeypatch.setattr(n, "least_squares", exercise)
    counts.append(0)
    poses, report = n.refine_column(meshes, joints, candidates, options)
    monkeypatch.setattr(n, "_ExactMemo", NoReuse)
    counts.append(0)
    fresh_poses, fresh_report = n.refine_column(meshes, joints, candidates, options)
    assert counts[0] < counts[1] / 2
    assert report == fresh_report
    assert np.array_equal(poses, fresh_poses)
    for reused, fresh in zip(observed[0][0], observed[1][0]):
        assert reused.tobytes() == fresh.tobytes()
    assert np.array_equal(observed[0][1], observed[1][1])
    assert np.array_equal(observed[0][2], observed[1][2])
    collision_terms = [r[-6:] for r in observed[0][0]]
    assert np.any(collision_terms[0] > 0)
    assert any(not np.array_equal(r, collision_terms[0]) for r in collision_terms[1:])


def test_exact_memo_uses_all_bits_and_bounded_lru_storage():
    memo = n._ExactMemo(2)
    a = np.eye(4)
    b = a.copy()
    b[0, 3] = np.nextafter(0.0, 1.0)
    c = a.copy()
    c[1, 3] = 0.1
    memo.put(a.tobytes(), "a")
    memo.put(b.tobytes(), "b")
    assert memo.get(a.tobytes()) == "a"
    assert memo.get(b.tobytes()) == "b"
    assert memo.get(a.tobytes()) == "a"
    memo.put(c.tobytes(), "c")
    assert memo.get(b.tobytes()) is None
    assert memo.get(a.tobytes()) == "a"
    assert memo.get(c.tobytes()) == "c"


def test_broad_phase_bounds_follow_geometry_between_calls():
    meshes, _, _, _ = fixture_chain(count=3)
    mesh = meshes[0]
    pose = n.rigid(translation=[1.0, -2.0, 3.0])
    before = n.pair_aabb(mesh, pose)
    mesh.vertices[0, 0] -= 10.0
    after = n.pair_aabb(mesh, pose)
    assert after[0][0] < before[0][0] - 9.0


@pytest.mark.parametrize(
    "shape,scale",
    [
        ("convex", 0.001),
        ("convex", 1.0),
        ("convex", 1000.0),
        ("concave", 0.1),
        ("concave", 1.0),
        ("concave", 10.0),
    ],
)
def test_penetration_broad_phase_matches_full_signed_distance_bit_for_bit(shape, scale):
    if shape == "convex":
        mesh, _ = wedge("closed", scale=scale)
    else:
        original = marching_cubes_vertebra("concave", np.eye(4), voxel=0.08, noise=0.0, seed=9)
        mesh = n.Mesh("concave_scaled", scale * original.vertices, original.faces)
    assert mesh.closed and mesh.self_intersection_checked
    rng = np.random.default_rng(7151)
    box = np.asarray(mesh.poly.GetBounds()).reshape(3, 2)
    midpoint = box.mean(axis=1)
    P = rng.uniform(-2.0, 2.0, (1200, 3)) * np.ptp(box, axis=1) + midpoint
    # Include the actual surface and points just inside/outside the AABB.
    boundary = np.tile(midpoint, (12, 1))
    for axis in range(3):
        for side in range(2):
            for direction in range(2):
                boundary[4 * axis + 2 * side + direction, axis] = np.nextafter(
                    box[axis, side], np.inf if direction else -np.inf
                )
    P = np.vstack([P, mesh.vertices, mesh.centers, boundary])
    for tolerance in [0.0, 0.007]:
        expected = np.maximum(-mesh.signed_distance(P) / scale - tolerance, 0)
        actual = mesh.penetration(P, scale, tolerance)
        assert actual.tobytes() == expected.tobytes()
        order = rng.permutation(len(P))
        assert mesh.penetration(P[order], scale, tolerance).tobytes() == expected[order].tobytes()


@pytest.mark.parametrize("uncertified", ["open", "unchecked"])
def test_penetration_never_culls_when_surface_signs_are_unverified(monkeypatch, uncertified):
    mesh, _ = wedge("unverified")
    if uncertified == "open":
        mesh.closed = False
    else:
        mesh.self_intersection_checked = False
    P = np.array([[200.0, 300.0, 400.0], [-2.0, -3.0, -4.0]])
    queried = []

    def uncertain_distance(points):
        queried.append(points.copy())
        return np.array([-0.25, 0.7])

    monkeypatch.setattr(mesh, "signed_distance", uncertain_distance)
    assert np.array_equal(mesh.penetration(P, 2.0, 0.03), [0.095, 0.0])
    assert len(queried) == 1 and np.array_equal(queried[0], P)


def test_penetration_keeps_boundary_band_and_skips_only_provably_exterior_points(monkeypatch):
    mesh, _ = wedge("closed")
    P = np.array([[0.0, 0.0, 0.0], [0.65 + 1e-14, 0.0, 0.0], [200.0, 300.0, 400.0]])
    original = mesh.signed_distance
    queried = []
    expected = np.maximum(-original(P), 0)

    def tracked(points):
        queried.append(points.copy())
        return original(points)

    monkeypatch.setattr(mesh, "signed_distance", tracked)
    assert mesh.penetration(P).tobytes() == expected.tobytes()
    assert len(queried) == 1 and np.array_equal(queried[0], P[:2])
    assert np.array_equal(mesh.penetration(P[2:]), [0.0])
    assert len(queried) == 1


def test_shared_internal_config_keeps_public_joint_configs_independent():
    meshes, cfg, _, _ = fixture_chain(count=3)
    _, _, report, joints = n.fit_column(meshes, cfg, fast_options())
    assert joints[0]._config is None and joints[1]._config is None
    assert joints[0]._config_snapshot is joints[1]._config_snapshot
    cfg["joints"][0]["patches"][0]["weight"] = 99.0
    report["configuration"]["joints"][0]["patches"][0]["weight"] = 88.0
    first = joints[0].config
    assert "weight" not in first["joints"][0]["patches"][0]
    first["joints"][0]["patches"][0]["weight"] = 77.0
    assert "weight" not in joints[1].config["joints"][0]["patches"][0]
    assert joints[0]._fitting_config() is first
    replacement = {"mesh_coordinate_system": "LPS"}
    joints[0].config = replacement
    assert joints[0].config is replacement
    assert joints[0]._fitting_config() is replacement


def test_standalone_joint_copies_its_config_before_caller_can_mutate_it():
    meshes, cfg, _, _ = fixture_chain(count=3)
    joint = n.Joint(meshes[0], meshes[1], cfg, fast_options())
    cfg["mesh_coordinate_system"] = "LPS"
    cfg["joints"][0]["patches"][0]["weight"] = 99.0
    assert joint.config["mesh_coordinate_system"] == "RAS"
    assert "weight" not in joint.config["joints"][0]["patches"][0]
