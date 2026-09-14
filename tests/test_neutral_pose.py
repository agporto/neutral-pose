"""Behavioral and geometric regression tests; fixtures are synthetic, not vertebrae."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from neutral_pose import core as n


def wedge(name, angle=0.0, scale=1.0, world=None):
    """Closed wedge with a tiled anterior and posterior face, in millimeters."""
    grid = np.linspace(-0.65, 0.65, 5)
    vertices = []
    for layer in range(2):
        for y in grid:
            for x in grid:
                z = -0.5 if layer == 0 else 0.5 + np.tan(angle) * y
                vertices.append([x, y, z])
    faces, bottom, top = [], [], []
    N = len(grid)
    plane = N * N
    for layer in range(2):
        for iy in range(N - 1):
            for ix in range(N - 1):
                a = layer * plane + iy * N + ix
                ids = [[a, a + 1, a + N + 1], [a, a + N + 1, a + N]]
                if layer == 0:
                    ids = [t[::-1] for t in ids]
                for triangle in ids:
                    (bottom if layer == 0 else top).append(len(faces))
                    faces.append(triangle)
    perimeter = (
        list(range(N))
        + [k * N + N - 1 for k in range(1, N)]
        + list(range(plane - 2, plane - N - 1, -1))
        + [k * N for k in range(N - 2, 0, -1)]
    )
    for a, b in zip(perimeter, perimeter[1:] + perimeter[:1]):
        faces.extend([[a, b + plane, b], [a, a + plane, b + plane]])
    V = np.asarray(vertices) * scale
    if world is not None:
        V = n.transform(world, V)
    mesh = n.Mesh(name, V, np.asarray(faces))
    # Three separate patches provide centering constraints at noncollinear points.
    patches = []
    for faceids in (bottom, top):
        centers = np.asarray(vertices)[np.asarray(faces)[faceids]].mean(axis=1)
        regions = [
            np.asarray(faceids)[centers[:, 1] < -0.1],
            np.asarray(faceids)[(centers[:, 1] >= -0.1) & (centers[:, 0] < 0)],
            np.asarray(faceids)[(centers[:, 1] >= -0.1) & (centers[:, 0] >= 0)],
        ]
        patches.append(regions)
    return mesh, patches


def fixture_pair(angle=0.0, scale=1.0, rotation=(9, -7, 5), translation=(0.10, -0.07, 0.05)):
    a, pa = wedge("v01", angle, scale)
    # The bottom plane of B is parallel to the top plane of A at a known gap.
    R = Rotation.from_euler("x", angle).as_matrix()
    normal = R @ np.array([0.0, 0.0, 1.0])
    truth = n.rigid(R, np.array([0.0, 0.0, 0.5 * scale]) + (0.5 + 0.02) * scale * normal)
    perturb = n.rigid(
        Rotation.from_euler("xyz", rotation, degrees=True).as_matrix(), np.asarray(translation) * scale
    )
    b, pb = wedge("v02", angle, scale, perturb @ truth)
    specs = []
    for i in range(3):
        specs.append(
            {
                "name": ["centrum", "left_facet", "right_facet"][i],
                "kind": "centrum" if i == 0 else "facet",
                "a": {"faces": pa[1][i].tolist()},
                "b": {"faces": pb[0][i].tolist()},
                "gap_fraction": 0.02,
                "gap_tolerance_fraction": 0.0,
            }
        )
    config = {
        "mesh_coordinate_system": "RAS",
        "mesh_scales": {"v01": scale, "v02": scale},
        "joints": [{"a": "v01", "b": "v02", "patches": specs}],
    }
    return a, b, config, n.inverse(perturb)


def fast_options(**kw):
    # Coarse hand-built wedges are noise-free; the edge-length noise heuristic
    # is for marching-cubes meshes, so it is disabled here (see test_realistic.py).
    opts = dict(
        samples=48,
        target_samples=512,
        collision_samples=96,
        starts=2,
        max_nfev=80,
        global_max_nfev=45,
        sensitivity=False,
        center_weight=2.0,
        normal_weight=0.2,
        auto_noise_floor=False,
    )
    opts.update(kw)
    return n.Options(**opts)


def test_reflection_rejected_and_round_trip():
    reflected = np.eye(4)
    reflected[0, 0] = -1
    with pytest.raises(ValueError, match="proper rigid"):
        n.validate_rigid(reflected)
    T = n.rigid(Rotation.from_euler("xyz", [43, -37, 11], degrees=True).as_matrix(), [10, -20, 3])
    P = np.random.default_rng(4).normal(size=(10, 3))
    assert np.allclose(n.transform(n.inverse(T), n.transform(T, P)), P)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_pose_recovery_and_scale_invariance(scale):
    a, b, cfg, expected = fixture_pair(scale=scale)
    matrices, poses, report, joints = n.fit_column([a, b], cfg, fast_options())
    difference = n.pose_difference(matrices[1], expected, scale)
    assert np.rad2deg(np.linalg.norm(difference[:3])) < 0.15
    assert np.linalg.norm(difference[3:]) < 0.004
    assert np.allclose(matrices[0], np.eye(4), atol=1e-10)
    assert all(np.isclose(np.linalg.det(M[:3, :3]), 1) for M in matrices)
    before = np.linalg.norm(b.vertices[1:] - b.vertices[:-1], axis=1)
    after = np.linalg.norm(np.diff(n.transform(poses[1], b.vertices), axis=0), axis=1)
    assert np.allclose(before, after, atol=scale * 1e-10)
    assert not report["joints"][0]["metrics"]["intersection_detected"]
    assert report["joints"][0]["metrics"]["max_sampled_penetration"] < scale * 1e-5


def test_curved_reference_is_recovered():
    a, b, cfg, expected = fixture_pair(angle=np.deg2rad(14))
    matrices, _, report, _ = n.fit_column([a, b], cfg, fast_options())
    difference = n.pose_difference(matrices[1], expected, 1.0)
    assert np.rad2deg(np.linalg.norm(difference[:3])) < 0.3
    assert np.linalg.norm(difference[3:]) < 0.008


def test_numeric_lps_fcsv_and_quoted_label(tmp_path):
    f = tmp_path / "test.fcsv"
    f.write_text(
        "# Markups fiducial file version = 4.11\n# CoordinateSystem = 1\n"
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
        '\n1,1,2,3,0,0,0,1,1,1,0,"left, facet",,\n'
    )
    lm = n.read_landmarks(f, "RAS")
    assert lm.labels == ["left, facet"]
    assert np.array_equal(lm.points, [[-1.0, -2.0, 3.0]])
    out = tmp_path / "out.mrk.json"
    n.write_landmarks(out, lm.points, lm.labels, "RAS")
    again = n.read_landmarks(out, "LPS")
    assert np.array_equal(again.points, [[1.0, 2.0, 3.0]])


def test_missing_or_duplicate_landmarks_fail(tmp_path):
    f = tmp_path / "bad.fcsv"
    f.write_text("label,1,2,3\n")
    with pytest.raises(ValueError, match="columns"):
        n.read_landmarks(f, "RAS")
    f.write_text("# columns = label,x,y,z\np,1,2,3\np,2,3,4\n")
    with pytest.raises(ValueError, match="duplicate"):
        n.read_landmarks(f, "RAS")


def test_closed_signed_distance_and_containment():
    outer, _ = wedge("outer", scale=2.0)
    inner, _ = wedge("inner", scale=0.1)
    assert outer.closed and inner.closed
    assert outer.signed_distance(np.array([[0.0, 0.0, 0.0]]))[0] < 0
    assert not n.intersects(outer, inner, np.eye(4))
    assert np.all(outer.signed_distance(inner.vertices) < 0)


def test_open_mesh_cannot_pass_as_verified():
    a, b, cfg, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    # Remove an unrelated side triangle; keep patch IDs unchanged by replacing
    # the final triangle only.
    b = n.Mesh(b.name, b.vertices + b.origin, b.faces[:-1])
    _, _, report, _ = n.fit_column([a, b], cfg, fast_options(starts=1, max_nfev=12))
    assert not report["joints"][0]["metrics"]["signed_distance_reliable"]
    assert "open_or_nonmanifold_mesh_sign_unverified" in report["joints"][0]["review_reasons"]


def test_anatomy_required_is_not_silently_guessed():
    a, b, cfg, _ = fixture_pair()
    with pytest.raises(ValueError, match="required"):
        n.Joint(a, b, {"mesh_coordinate_system": "RAS", "require_anatomy": True}, fast_options())


def test_geometry_only_is_provisional():
    a, b, _, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    _, _, report, _ = n.fit_column(
        [a, b], {"mesh_coordinate_system": "RAS"}, fast_options(starts=1, max_nfev=15)
    )
    assert "geometry_only_patch_identity" in report["joints"][0]["review_reasons"]


def test_explicit_patch_centers_seed_proper_rotation():
    a, b, cfg, _ = fixture_pair(rotation=(35, -28, 90), translation=(2, 1, -0.3))
    joint = n.Joint(a, b, cfg, fast_options())
    seed = joint.landmark_seed()
    assert seed is not None and np.isclose(np.linalg.det(seed[:3, :3]), 1.0)
    candidate = n.optimize_joint(joint, seed)
    assert not joint.metrics(candidate.transform, final=True)["intersection_detected"]


def test_sensitivity_is_recorded():
    a, b, cfg, _ = fixture_pair(rotation=(2, -1, 1), translation=(0, 0, 0))
    _, _, report, _ = n.fit_column([a, b], cfg, fast_options(sensitivity=True))
    sensitivity = report["joints"][0]["uncertainty"]["sensitivity"]
    assert {x["parameter"] for x in sensitivity} == {"clearance", "centering"}
    assert len(sensitivity) == 3
    assert all(x["translation_change_fraction"] > 0.001 for x in sensitivity if x["parameter"] == "clearance")


def test_batch_round_trip_and_atomic_overwrite(tmp_path):
    a, b, cfg, _ = fixture_pair(rotation=(3, -2, 1))
    source = tmp_path / "input" / "specimen"
    source.mkdir(parents=True)
    for mesh in [a, b]:
        n.write_mesh(source / f"{mesh.name}.ply", mesh, n.rigid(translation=mesh.origin))
    (source / "neutral_config.json").write_text(json.dumps(cfg))
    lm_path = source / "LMKs" / "v02.mrk.json"
    n.write_landmarks(lm_path, [b.vertices[0] + b.origin], ["check"], "RAS")
    out = tmp_path / "results"
    result = n.process_root(source.parent, out, {}, fast_options(starts=1))
    assert result[0]["status"] != "failed", result
    dest = out / "specimen"
    M = np.load(dest / "neutral_transforms.npy")
    lm = n.read_landmarks(dest / "LMKs_json" / "neutral_v02.mrk.json", "RAS")
    assert np.allclose(lm.points[0], n.transform(M[1], b.vertices[0] + b.origin), atol=1e-6)
    loaded = n.Mesh.read(dest / "neutral_v02.ply")
    # Cleaning may renumber vertices; compare as point sets.
    got = np.array(sorted(map(tuple, np.round(loaded.vertices + loaded.origin, 5))))
    want = np.array(sorted(map(tuple, np.round(n.transform(M[1], b.vertices + b.origin), 5))))
    assert np.allclose(got, want, atol=1e-5)
    assert (dest / "patches" / "joint_001_centrum_a_centers.npy").exists()
    with pytest.raises(FileExistsError):
        n.process_specimen(source, dest, {}, fast_options())
    # A failed overwrite must not destroy the last successful result.
    (source / "neutral_config.json").write_text(
        json.dumps({"mesh_coordinate_system": "RAS", "require_anatomy": True})
    )
    with pytest.raises(ValueError):
        n.process_specimen(source, dest, {}, fast_options(), overwrite=True)
    assert np.array_equal(M, np.load(dest / "neutral_transforms.npy"))


def test_input_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(ValueError, match="input"):
        n.process_root(tmp_path, tmp_path, {})


def fixture_chain(count=3, angle_deg=8.0, seed=44):
    rng = np.random.default_rng(seed)
    theta = np.deg2rad(angle_deg)
    R = Rotation.from_euler("x", theta).as_matrix()
    local = n.rigid(R, np.array([0.0, 0.0, 0.5]) + 0.52 * (R @ [0.0, 0.0, 1.0]))
    truth = np.eye(4)
    meshes, regions, expected, truth_poses = [], [], [], []
    for i in range(count):
        change = (
            np.eye(4)
            if i == 0
            else n.rigid(Rotation.from_rotvec(rng.normal(0, 0.1, 3)).as_matrix(), rng.normal(0, 0.06, 3))
        )
        m, patches = wedge(f"v{i + 1:02d}", theta, world=change @ truth)
        labels = ["anterior_centrum", "posterior_centrum", "dorsal"]
        m.landmarks = n.Landmarks(
            n.transform(change @ truth, [[0, 0, -0.5], [0, 0, 0.5], [0, 0.5, 0]]), labels, "RAS"
        )
        meshes.append(m)
        regions.append(patches)
        expected.append(n.inverse(change))
        truth_poses.append(truth.copy())
        truth = truth @ local
    config = {"mesh_coordinate_system": "RAS", "mesh_scales": {m.name: 1.0 for m in meshes}, "joints": []}
    for i in range(count - 1):
        config["joints"].append(
            {
                "a": meshes[i].name,
                "b": meshes[i + 1].name,
                "region": "trunk",
                "patches": [
                    {
                        "name": str(k),
                        "kind": "centrum" if k == 0 else "facet",
                        "a": {"faces": regions[i][1][k].tolist()},
                        "b": {"faces": regions[i + 1][0][k].tolist()},
                        "gap_fraction": 0.02,
                        "gap_tolerance_fraction": 0.0,
                    }
                    for k in range(3)
                ],
            }
        )
    return meshes, config, np.stack(expected), truth_poses


def test_global_refinement_preserves_curved_chain():
    meshes, cfg, expected, _ = fixture_chain()
    M, poses, report, _ = n.fit_column(meshes, cfg, fast_options())
    assert report["global_refinement"]["performed"]
    assert report["global_refinement"]["accepted"]
    assert report["global_refinement"]["final_score"] <= report["global_refinement"]["initial_score"] + 1e-10
    for estimate, truth in zip(M, expected):
        d = n.pose_difference(estimate, truth, 1.0)
        assert np.rad2deg(np.linalg.norm(d[:3])) < 0.5
        assert np.linalg.norm(d[3:]) < 0.015
    # The end vertebra retains the 16-degree cumulative anatomical bend.
    relative = report["joints"][0]["anatomical_orientation_b_in_a"]
    assert np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())) > 5
    assert not report["nonadjacent_contacts"]


def test_landmark_patches_and_boundary_sensitivity():
    a, b, cfg, expected = fixture_pair(rotation=(5, -4, 2))
    for side, mesh in [("a", a), ("b", b)]:
        labels, points = [], []
        for k, spec in enumerate(cfg["joints"][0]["patches"]):
            ids = spec[side]["faces"]
            point = np.average(mesh.centers[ids], axis=0, weights=mesh.areas[ids]) + mesh.origin
            label = f"surface_{k}"
            labels.append(label)
            points.append(point)
            spec[side] = {"label": label, "radius_fraction": 0.24, "normal_angle_deg": 45}
        mesh.landmarks = n.Landmarks(np.asarray(points), labels, "RAS")
    _, _, report, _ = n.fit_column([a, b], cfg, fast_options(sensitivity=True))
    trials = report["joints"][0]["uncertainty"]["sensitivity"]
    assert {x["parameter"] for x in trials} == {"clearance", "patch_radius", "centering"}
    assert len(trials) == 5


def test_self_intersection_is_rejected():
    a, _ = wedge("v01")
    V = np.vstack([a.vertices, a.vertices + [0.3, 0.2, 0.1]])
    F = np.vstack([a.faces, a.faces + len(a.vertices)])
    with pytest.raises(ValueError, match="intersect"):
        n.Mesh("invalid_overlap", V, F)


def test_far_disarticulated_unlabeled_meshes_fail():
    a, _ = wedge("v01")
    b, _ = wedge("v02", world=n.rigid(translation=[100.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="No nearby opposing surfaces"):
        n.Joint(a, b, {"mesh_coordinate_system": "RAS"}, fast_options())


def test_unit_and_coordinate_system_are_explicit():
    a, b, _, _ = fixture_pair()
    with pytest.raises(ValueError, match="coordinate"):
        n.fit_column([a, b], {}, fast_options())
    with pytest.raises(ValueError, match="millimeters"):
        n.fit_column([a, b], {"mesh_coordinate_system": "RAS", "length_unit": "cm"}, fast_options())


def test_cli_anatomy_requirement_overrides_local_config(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    a, b, _, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    for mesh in (a, b):
        n.write_mesh(source / f"{mesh.name}.ply", mesh, n.rigid(translation=mesh.origin))
    (source / "neutral_config.json").write_text(
        json.dumps({"mesh_coordinate_system": "RAS", "require_anatomy": False})
    )
    code = n.main([str(source), "--specimen", "--out", str(tmp_path / "out"), "--require-anatomy"])
    assert code == 1


def test_already_articulated_pair_is_preserved():
    a, b, cfg, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    M, _, _, _ = n.fit_column([a, b], cfg, fast_options())
    assert np.allclose(M, np.repeat(np.eye(4)[None], 2, axis=0), atol=1e-7)


def test_sphere_fit_accepts_curvature_and_rejects_planes():
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.5)
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(24)
    sphere.Update()
    p = sphere.GetOutput()
    mesh = n.Mesh(
        "sphere",
        vtk_to_numpy(p.GetPoints().GetData()),
        n.faces_of(p),
    )
    patch = np.where(mesh.centers[:, 2] > 0.2)[0]
    center, radius, error = n.fit_sphere(mesh, patch)
    assert center is not None and np.linalg.norm(center) < 0.005
    assert abs(radius - 0.5) < 0.005
    plane, regions = wedge("plane")
    assert n.fit_sphere(plane, regions[1][0])[0] is None


def test_global_refinement_corrects_biased_pairwise_initialization():
    meshes, cfg, _, _ = fixture_chain()
    options = fast_options()
    joints = [n.Joint(a, b, cfg, options, i) for i, (a, b) in enumerate(zip(meshes[:-1], meshes[1:]))]
    candidates = [n.fit_joint(j)[0] for j in joints]
    candidates[1].transform = n.perturb(candidates[1].transform, [0.04, -0.03, 0.02, 0.04, 0.02, -0.03], 1.0)
    _, result = n.refine_column(meshes, joints, candidates, options)
    assert result["nfev"] > 1
    assert result["final_score"] < result["initial_score"] * 0.2


def test_noisy_surface_recovery():
    a, b, cfg, truth = fixture_pair()
    rng = np.random.default_rng(550)
    # Independent segmentation noise on the two meshes, at 0.1% of local scale.
    a = n.Mesh(a.name, a.vertices + a.origin + rng.normal(0, 0.001, a.vertices.shape), a.faces)
    b = n.Mesh(b.name, b.vertices + b.origin + rng.normal(0, 0.001, b.vertices.shape), b.faces)
    M, _, report, _ = n.fit_column([a, b], cfg, fast_options())
    error = n.pose_difference(M[1], truth, 1.0)
    assert np.rad2deg(np.linalg.norm(error[:3])) < 0.6
    assert np.linalg.norm(error[3:]) < 0.01
    assert report["joints"][0]["metrics"]["max_sampled_penetration"] < 0.001
