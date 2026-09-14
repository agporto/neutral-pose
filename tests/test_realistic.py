"""Recovery tests on marching-cubes geometry with segmentation-like noise.

The synthetic vertebra has a spherical condyle (posterior) and cotyle
(anterior), an arch, and two zygapophyseal pads per end tilted +-20 degrees
about the column axis so the facets are not coplanar. Bones are meshed
independently by vtkFlyingEdges3D at different voxel pitches, with Gaussian
noise added to the implicit field, so tessellations are unrelated, surfaces
are bumpy, and patch boundaries grown from landmarks never match exactly.
Landmark seeds are analytic points on the articular surfaces, as an anatomist
or landmark transfer would supply them. These are not vertebrae; they
exercise spherical centering, planar facets
with a free yaw, noise-widened tolerances, and non-homologous patch growth.
"""

import numpy as np
import pytest
import vtk
import vtk.util.numpy_support
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import numpy_to_vtk

from neutral_pose import core as n

CENTRUM_LENGTH = 1.0
GAP = 0.02
FACET_TILT = np.deg2rad(20.0)
BALL = dict(center=np.array([0.0, 0.0, 0.40]), radius=0.40)  # posterior condyle
SOCKET = dict(
    center=np.array([0.0, 0.0, -0.62]), radius=0.42
)  # anterior cotyle (concentric with previous bone's ball)
POST_PAD = dict(center_x=0.55, center_z=0.34, half=np.array([0.22, 0.18, 0.15]))  # postzygapophyses


def rot_y(angle):
    return Rotation.from_euler("y", angle).as_matrix()


def facet_normal(side):
    return rot_y(side * FACET_TILT) @ np.array([0.0, 0.0, 1.0])


def pad_centers(side):
    """Postzygapophysis pad center, and the prezygapophysis pad center placed so
    that the previous bone's postzygapophysis face mates with it at GAP along
    the tilted normal (bone-local coordinates)."""
    post = np.array([side * POST_PAD["center_x"], 0.70, POST_PAD["center_z"]])
    pre = (
        post
        + 2 * POST_PAD["half"][2] * facet_normal(side)
        + GAP * facet_normal(side)
        - np.array([0.0, 0.0, CENTRUM_LENGTH + GAP])
    )
    return post, pre


def sdf_box(P, center, half, R=None):
    q = P - center
    if R is not None:
        q = q @ R
    d = np.abs(q) - half
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0)


def sdf_sphere(P, center, radius):
    return np.linalg.norm(P - center, axis=-1) - radius


def sdf_cylinder_z(P, radius, z0, z1):
    dr = np.hypot(P[..., 0], P[..., 1]) - radius
    dz = np.maximum(z0 - P[..., 2], P[..., 2] - z1)
    d = np.stack([dr, dz], axis=-1)
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(axis=-1), 0)


def vertebra_sdf(P):
    # Non-articular surfaces stand off further than the articular gap, as the
    # centrum rim and laminae do on real vertebrae; only ball/socket and the
    # zygapophyseal pads mate at GAP.
    body = sdf_cylinder_z(P, 0.45, -0.5, 0.46)
    body = np.minimum(body, sdf_sphere(P, BALL["center"], BALL["radius"]))
    body = np.maximum(body, -sdf_sphere(P, SOCKET["center"], SOCKET["radius"]))
    arch = sdf_box(P, np.array([0.0, 0.55, 0.0]), np.array([0.48, 0.12, 0.44]))
    d = np.minimum(body, arch)
    for side in (-1, 1):
        post, pre = pad_centers(side)
        R = rot_y(side * FACET_TILT)
        d = np.minimum(d, sdf_box(P, post, POST_PAD["half"], R))
        d = np.minimum(d, sdf_box(P, pre, POST_PAD["half"], R))
    return d


def analytic_landmarks():
    """Seeds on the articular surfaces in bone-local coordinates."""
    marks = {
        "posterior_centrum": BALL["center"] + [0, 0, BALL["radius"]],
        "anterior_centrum": SOCKET["center"] + [0, 0, SOCKET["radius"]],
        "dorsal": np.array([0.0, 0.67, 0.0]),
    }
    for side, name in ((-1, "left"), (1, "right")):
        post, pre = pad_centers(side)
        nrm = facet_normal(side)
        marks[f"{name}_postzygapophysis"] = post + POST_PAD["half"][2] * nrm
        marks[f"{name}_prezygapophysis"] = pre - POST_PAD["half"][2] * nrm
    return marks


def marching_cubes_vertebra(name, world, voxel=0.04, noise=0.25, seed=0, offset=0.371):
    """Mesh the implicit vertebra on a voxel grid with field noise, then pose it."""
    rng = np.random.default_rng(seed)
    lo, hi = np.array([-1.1, -0.65, -0.85]), np.array([1.1, 1.05, 1.0])
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    axes = [lo[k] + offset * voxel + voxel * np.arange(dims[k]) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1)
    field = vertebra_sdf(P) + rng.normal(0, noise * voxel, P.shape[:3])
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetOrigin(*(lo + offset * voxel))
    image.SetSpacing(voxel, voxel, voxel)
    arr = numpy_to_vtk(np.ascontiguousarray(field.transpose(2, 1, 0)).ravel(), deep=True)
    arr.SetName("sdf")
    image.GetPointData().SetScalars(arr)
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(image)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOff()
    fe.Update()
    tf = vtk.vtkTransform()
    tf.SetMatrix(np.asarray(world, float).ravel())
    tpd = vtk.vtkTransformFilter()
    tpd.SetInputConnection(fe.GetOutputPort())
    tpd.SetTransform(tf)
    tpd.Update()
    marks = analytic_landmarks()
    landmarks = n.Landmarks(n.transform(world, np.array(list(marks.values()))), list(marks), "RAS")
    return n.Mesh.from_polydata(name, tpd.GetOutput(), landmarks)


def articulated_truth(count):
    """World pose of bone i in a straight chain that mates at GAP."""
    return [n.rigid(translation=[0.0, 0.0, i * (CENTRUM_LENGTH + GAP)]) for i in range(count)]


def realistic_config():
    return {
        "mesh_coordinate_system": "RAS",
        "require_anatomy": True,
        "patch_defaults": {
            "centrum": {"radius_fraction": 0.30, "normal_angle_deg": 85},
            "facet": {"radius_fraction": 0.14, "normal_angle_deg": 40},
        },
        "joint_gaps": {"centrum": GAP, "facet": GAP},
    }


def realistic_options(**kw):
    opts = dict(
        samples=96,
        target_samples=1024,
        collision_samples=192,
        starts=3,
        max_nfev=80,
        global_max_nfev=45,
        sensitivity=False,
        noise_floor_mm=0.02,
    )  # Declared fixture allowance, independent of tessellation.
    opts.update(kw)
    return n.Options(**opts)


@pytest.fixture(scope="module")
def noisy_pair():
    rng = np.random.default_rng(7)
    truth = articulated_truth(2)
    perturbation = n.rigid(
        Rotation.from_euler("xyz", [8, -6, 11], degrees=True).as_matrix(), [0.12, -0.08, 0.07]
    )
    a = marching_cubes_vertebra("v01", truth[0], voxel=0.04, noise=0.25, seed=1, offset=0.371)
    b = marching_cubes_vertebra("v02", perturbation @ truth[1], voxel=0.046, noise=0.25, seed=2, offset=0.618)
    return a, b, n.inverse(perturbation)


def test_fixture_is_a_plausible_segmentation(noisy_pair):
    a, b, _ = noisy_pair
    assert a.closed and b.closed
    assert 3000 < len(a.faces) < 60000
    assert len(a.faces) != len(b.faces)  # unrelated tessellations
    assert 0.02 < a.resolution < 0.08  # sampling resolution of this fixture
    assert a.cleaning["connected_components"] >= 1


def test_noisy_ball_and_socket_pair_is_recovered(noisy_pair):
    a, b, expected = noisy_pair
    matrices, poses, report, joints = n.fit_column([a, b], realistic_config(), realistic_options())
    joint = report["joints"][0]
    # The centrum is recognized as spherical on both bones despite the noise.
    centrum = next(p for p in joint["metrics"]["patches"] if p["kind"] == "centrum")
    assert centrum["spherical_centrum"]
    assert all(src == "seed" for p in joint["metrics"]["patches"] for src in p["anchor_source"])
    # Tolerances widened to the noise floor rather than the 0.3% default.
    assert joint["effective_tolerances"]["noise_floor_mm"] == 0.02
    assert joint["effective_tolerances"]["noise_floor_source"] == "explicit"
    d = n.pose_difference(matrices[1], expected, 1.0)
    assert np.rad2deg(np.linalg.norm(d[:3])) < 1.0, d
    assert np.linalg.norm(d[3:]) < 0.02, d
    # Noisy surfaces at their nominal gap may touch; that is reported, not flagged.
    assert "triangle_intersections_require_review" not in joint["review_reasons"]
    assert "penetration_exceeds_tolerance" not in joint["review_reasons"]
    assert "poor_surface_fit" not in joint["review_reasons"]
    assert "insufficient_surface_coverage" not in joint["review_reasons"]


def test_recovery_does_not_depend_on_the_seed_alone(noisy_pair):
    """The optimizer must move from a deliberately wrong start to the truth;
    the shipped v1 validation only ever confirmed its own Kabsch seed."""
    a, b, expected = noisy_pair
    joint = n.Joint(a, b, realistic_config(), realistic_options(), 0)
    H_true = n.rigid(translation=-a.origin) @ expected @ n.rigid(translation=b.origin)
    wrong = n.perturb(H_true, [np.deg2rad(6), -np.deg2rad(5), np.deg2rad(7), 0.06, -0.05, 0.04], joint.scale)
    candidate = n.optimize_joint(joint, wrong, max_nfev=150)
    d = n.pose_difference(candidate.transform, H_true, joint.scale)
    # Yaw about the column axis is the weakly determined direction (ball gives
    # none, facets little); at this noise level it resolves to about a degree.
    assert np.rad2deg(np.linalg.norm(d[:3])) < 1.5, d
    assert np.linalg.norm(d[3:]) < 0.03, d


def test_noisy_three_bone_chain(noisy_pair):
    rng = np.random.default_rng(11)
    truth = articulated_truth(3)
    meshes, expected = [], []
    for i, T in enumerate(truth):
        change = (
            np.eye(4)
            if i == 0
            else n.rigid(Rotation.from_rotvec(rng.normal(0, 0.08, 3)).as_matrix(), rng.normal(0, 0.05, 3))
        )
        meshes.append(
            marching_cubes_vertebra(
                f"v{i + 1:02d}",
                change @ T,
                voxel=0.04 + 0.004 * i,
                noise=0.25,
                seed=20 + i,
                offset=0.2 + 0.3 * i,
            )
        )
        expected.append(n.inverse(change))
    matrices, poses, report, joints = n.fit_column(meshes, realistic_config(), realistic_options())
    # Each joint's relative pose is what the method estimates; errors then
    # accumulate along the chain (the first bone is fixed).
    for i in range(len(meshes) - 1):
        got = n.inverse(matrices[i]) @ matrices[i + 1]
        want = n.inverse(expected[i]) @ expected[i + 1]
        d = n.pose_difference(got, want, 1.0)
        assert np.rad2deg(np.linalg.norm(d[:3])) < 1.5, d
        assert np.linalg.norm(d[3:]) < 0.03, d
    for M, E in zip(matrices, expected):
        d = n.pose_difference(M, E, 1.0)
        assert np.rad2deg(np.linalg.norm(d[:3])) < 2.5, d
        assert np.linalg.norm(d[3:]) < 0.05, d
    assert not report["nonadjacent_contacts"]
    assert report["global_refinement"]["performed"]


def test_yaw_is_not_set_by_patch_growth_tie_breaks():
    """Regression for the 5-degree error found in review: identical planar
    wedges, landmark seeds exactly on a grid line, so patch growth picks
    different triangles on the two bones. Centering now acts on the seeds."""
    from test_neutral_pose import fast_options

    def wedge_grid(name, world, ngrid):
        grid = np.linspace(-0.65, 0.65, ngrid)
        V = []
        F = []
        for layer in range(2):
            for y in grid:
                for x in grid:
                    V.append([x, y, -0.5 if layer == 0 else 0.5])
        N = ngrid
        plane = N * N
        for layer in range(2):
            for iy in range(N - 1):
                for ix in range(N - 1):
                    a0 = layer * plane + iy * N + ix
                    ids = [[a0, a0 + 1, a0 + N + 1], [a0, a0 + N + 1, a0 + N]]
                    if layer == 0:
                        ids = [t[::-1] for t in ids]
                    F.extend(ids)
        per = (
            list(range(N))
            + [k * N + N - 1 for k in range(1, N)]
            + list(range(plane - 2, plane - N - 1, -1))
            + [k * N for k in range(N - 2, 0, -1)]
        )
        for p, q in zip(per, per[1:] + per[:1]):
            F.extend([[p, q + plane, q], [p, p + plane, q + plane]])
        return n.Mesh(name, n.transform(world, np.asarray(V)), np.asarray(F))

    truth = n.rigid(np.eye(3), [0, 0, 1.02])
    pert = n.rigid(Rotation.from_euler("xyz", [9, -7, 5], degrees=True).as_matrix(), [0.10, -0.07, 0.05])
    seeds_a = {"centrum": [0, -0.39, 0.5], "lf": [-0.3, 0.2, 0.5], "rf": [0.3, 0.2, 0.5]}
    seeds_b = {"centrum": [0, -0.39, -0.5], "lf": [-0.3, 0.2, -0.5], "rf": [0.3, 0.2, -0.5]}
    a = wedge_grid("v01", np.eye(4), 13)
    b = wedge_grid("v02", pert @ truth, 13)
    a.landmarks = n.Landmarks(np.asarray(list(seeds_a.values()), float), list(seeds_a), "RAS")
    b.landmarks = n.Landmarks(
        n.transform(pert @ truth, np.asarray(list(seeds_b.values()), float)), list(seeds_b), "RAS"
    )
    patches = [
        {
            "name": k,
            "kind": "centrum" if k == "centrum" else "facet",
            "a": {"label": k, "radius_fraction": 0.24, "normal_angle_deg": 45},
            "b": {"label": k, "radius_fraction": 0.24, "normal_angle_deg": 45},
            "gap_fraction": 0.02,
            "gap_tolerance_fraction": 0.0,
        }
        for k in seeds_a
    ]
    cfg = {
        "mesh_coordinate_system": "RAS",
        "mesh_scales": {"v01": 1.0, "v02": 1.0},
        "joints": [{"a": "v01", "b": "v02", "patches": patches}],
    }
    M, _, report, joints = n.fit_column([a, b], cfg, fast_options())
    sizes = [(len(p.a.ids), len(p.b.ids)) for p in joints[0].pairs]
    assert any(sa != sb for sa, sb in sizes), "fixture no longer reproduces asymmetric growth"
    d = n.pose_difference(M[1], n.inverse(pert), 1.0)
    assert np.rad2deg(np.linalg.norm(d[:3])) < 0.5, d
    assert np.linalg.norm(d[3:]) < 0.01, d


def test_preflight_scales_and_is_reported():
    """The vectorized self-intersection preflight must handle a mid-size mesh in
    seconds, and skipping it above the cap must be visible, not silent."""
    import time

    s = vtk.vtkSphereSource()
    s.SetRadius(3.0)
    s.SetThetaResolution(180)
    s.SetPhiResolution(180)
    s.Update()
    t = time.perf_counter()
    mesh = n.Mesh.from_polydata("sphere", s.GetOutput())
    assert len(mesh.faces) > 60000
    assert time.perf_counter() - t < 30
    assert mesh.self_intersection_checked and mesh.self_intersection is None
    capped = n.Mesh.from_polydata("sphere_capped", s.GetOutput(), preflight_triangles=1000)
    assert not capped.self_intersection_checked
    # And it still catches a genuine overlap between nonadjacent shells.
    V = np.vstack([mesh.vertices, mesh.vertices + [0.7, 0.4, 0.1]])
    F = np.vstack([mesh.faces, mesh.faces + len(mesh.vertices)])
    with pytest.raises(ValueError, match="intersect"):
        n.Mesh("overlap", V, F)


def test_unshared_vertices_are_merged(tmp_path):
    """Triangle soups (three private vertices per triangle, as in raw STL or
    some exporters) have only boundary edges; without merging, patch growth
    cannot cross triangles and closedness is never certified."""
    s = vtk.vtkSphereSource()
    s.SetThetaResolution(24)
    s.SetPhiResolution(16)
    s.Update()
    V = vtk.util.numpy_support.vtk_to_numpy(s.GetOutput().GetPoints().GetData())
    F = n.faces_of(s.GetOutput())
    soup = n.polydata(V[F].reshape(-1, 3), np.arange(3 * len(F)).reshape(-1, 3))
    mesh = n.Mesh.from_polydata("soup", soup)
    assert mesh.closed
    assert mesh.cleaning["merged_points"] > 0
    # Reading the same geometry from STL also yields a closed shell.
    w = vtk.vtkSTLWriter()
    w.SetInputData(s.GetOutput())
    w.SetFileName(str(tmp_path / "s.stl"))
    w.Write()
    assert n.Mesh.read(tmp_path / "s.stl").closed


def test_small_islands_are_removed_only_when_requested_and_reported():
    s = vtk.vtkSphereSource()
    s.SetRadius(1.0)
    s.Update()
    t = vtk.vtkSphereSource()
    t.SetRadius(0.05)
    t.SetCenter(3.0, 0, 0)
    t.Update()
    app = vtk.vtkAppendPolyData()
    app.AddInputData(s.GetOutput())
    app.AddInputData(t.GetOutput())
    app.Update()
    mesh = n.Mesh.from_polydata("two_shells", app.GetOutput(), keep_largest_component=True)
    assert mesh.cleaning["connected_components"] == 2
    assert mesh.cleaning["small_components_removed"] == 1
    kept = n.Mesh.from_polydata("two_shells_kept", app.GetOutput())
    assert kept.cleaning["small_components_removed"] == 0
