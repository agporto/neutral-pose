"""Regression cases independently reproducing the three v1.1 review defects."""

import copy
import json

import numpy as np
import pytest
import vtk
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import vtk_to_numpy

from neutral_pose import core as n
from test_neutral_pose import fast_options, fixture_pair


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_elongated_intersecting_triangles_are_not_culled(scale):
    # Their centroids lie farther apart than the sum of bbox half-diagonals.
    # A half-diagonal only encloses a triangle when centered at its bbox center.
    vertices = (
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 0.1, 0.0],
                [0.9, 0.001, 0.0],
            ]
        )
        * scale
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    assert vtk.vtkTriangle.TrianglesIntersect(*vertices[faces[0]], *vertices[faces[1]])
    with pytest.raises(ValueError, match="intersect"):
        n.Mesh("elongated_overlap", vertices, faces)


@pytest.mark.parametrize("seed", [3, 19, 81])
def test_preflight_matches_exhaustive_triangle_oracle(seed):
    rng = np.random.default_rng(seed)
    # Include separated inputs as well as intersecting triangles, multiple
    # length scales, and arbitrary orientations. No shared vertex IDs.
    for spread in (0.1, 2.0, 30.0):
        triangles = []
        for _ in range(25):
            triangle = np.array(
                [[0.0, 0.0, 0.0], [rng.uniform(0.5, 3), 0.0, 0.0], [0.0, 10 ** rng.uniform(-3, 0), 0.0]]
            )
            triangles.append(
                triangle @ Rotation.random(random_state=rng).as_matrix().T + rng.normal(0.0, spread, 3)
            )
        mesh = n.Mesh(
            "oracle",
            np.asarray(triangles).reshape(-1, 3),
            np.arange(75).reshape(-1, 3),
            preflight_triangles=1,
        )
        tri = mesh.vertices[mesh.faces]
        exact = any(
            vtk.vtkTriangle.TrianglesIntersect(*tri[i], *tri[j])
            for i in range(len(tri))
            for j in range(i + 1, len(tri))
        )
        hit = mesh.check_self_intersections()
        assert (hit is not None) == exact
        if hit is not None:
            assert vtk.vtkTriangle.TrianglesIntersect(*tri[hit[0]], *tri[hit[1]])


def body_with_dense_island():
    body = vtk.vtkCubeSource()
    body.SetXLength(4)
    body.SetYLength(4)
    body.SetZLength(4)
    body.Update()
    island = vtk.vtkSphereSource()
    island.SetRadius(0.1)
    island.SetCenter(10, 0, 0)
    island.SetThetaResolution(32)
    island.SetPhiResolution(24)
    island.Update()
    app = vtk.vtkAppendPolyData()
    app.AddInputData(body.GetOutput())
    app.AddInputData(island.GetOutput())
    app.Update()
    return app.GetOutput()


def test_default_loading_preserves_main_body_and_dense_island(tmp_path):
    source = body_with_dense_island()
    mesh = n.Mesh.from_polydata("two_components", source)
    assert mesh.connected_components == 2
    assert mesh.cleaning["components_removed"] == 0
    assert mesh.cleaning["component_selection"] == "preserve_all"
    assert np.ptp(mesh.vertices, axis=0)[0] > 12.0
    writer = vtk.vtkXMLPolyDataWriter()
    path = tmp_path / "two_components.vtp"
    writer.SetFileName(str(path))
    writer.SetInputData(source)
    assert writer.Write() == 1
    loaded = n.Mesh.read(path)
    assert loaded.connected_components == 2
    assert loaded.areas.sum() == pytest.approx(mesh.areas.sum())


def test_opt_in_component_selection_uses_area_not_cell_count():
    mesh = n.Mesh.from_polydata("body", body_with_dense_island(), keep_largest_component=True)
    assert mesh.connected_components == 1
    assert np.allclose(np.ptp(mesh.vertices, axis=0), [4.0, 4.0, 4.0])
    assert np.allclose(mesh.origin, 0.0)
    assert len(mesh.faces) == 12
    stats = mesh.cleaning
    assert stats["component_selection"] == "largest_surface_area"
    assert stats["components_removed"] == 1
    assert stats["component_triangles_removed"] > len(mesh.faces)
    assert 0 < stats["component_area_removed_mm2"] < 0.13
    assert 0 < stats["component_area_removed_fraction"] < 0.002
    assert sum(c["kept"] for c in stats["components"]) == 1


@pytest.fixture(scope="module")
def remeshed_pair():
    a, b, config, _ = fixture_pair()
    # Surface seeds identify the same physical patches on either tessellation.
    for pair in config["joints"][0]["patches"]:
        for side, mesh in (("a", a), ("b", b)):
            ids = pair[side]["faces"]
            point = np.average(mesh.centers[ids], axis=0, weights=mesh.areas[ids]) + mesh.origin
            pair[side] = {"point": point.tolist(), "radius_fraction": 0.35}
    refined = []
    for mesh in (a, b):
        sub = vtk.vtkLinearSubdivisionFilter()
        sub.SetInputData(n.polydata(mesh.vertices + mesh.origin, mesh.faces))
        sub.SetNumberOfSubdivisions(3)
        sub.Update()
        poly = sub.GetOutput()
        fine = n.Mesh(mesh.name, vtk_to_numpy(poly.GetPoints().GetData()), n.faces_of(poly))
        assert np.max(np.abs(mesh.signed_distance(fine.vertices + fine.origin - mesh.origin))) < 1e-6
        assert fine.areas.sum() == pytest.approx(mesh.areas.sum(), rel=1e-6)
        refined.append(fine)
    return (a, b), refined, config


@pytest.mark.parametrize("noise_mm", [None, 0.0, 0.001, 0.02])
def test_remeshing_does_not_change_default_or_explicit_tolerances(remeshed_pair, noise_mm):
    coarse, fine, cfg = remeshed_pair
    # Explicit values must also win when automatic estimation is enabled.
    opts = fast_options(noise_floor_mm=noise_mm, auto_noise_floor=noise_mm is not None)
    before = n.Joint(*coarse, cfg, opts)
    after = n.Joint(*fine, cfg, opts)
    assert before.tolerances == after.tolerances
    assert before.tolerances["noise_floor_mm"] == (noise_mm or 0.0)
    assert before.noise_source == ("disabled" if noise_mm is None else "explicit")
    assert before.tolerances["penetration_tolerance_mm"] == pytest.approx(max(0.001, 2 * (noise_mm or 0.0)))


def test_edge_heuristic_is_opt_in_and_reported_for_review(remeshed_pair):
    coarse, fine, cfg = remeshed_pair
    opts = fast_options(auto_noise_floor=True)
    before = n.Joint(*coarse, cfg, opts)
    after = n.Joint(*fine, cfg, opts)
    assert before.tolerances["noise_floor_mm"] > after.tolerances["noise_floor_mm"]
    assert before.noise_source == after.noise_source == "mesh_edge_heuristic"
    assert before.tolerances["automatic_noise_floor_capped"]
    _, _, report, _ = n.fit_column(coarse, cfg, opts)
    assert "noise_floor_estimated_from_mesh" in report["joints"][0]["review_reasons"]
    assert report["status"] == "needs_review"


def test_edge_resolution_is_independent_of_triangle_vertex_order():
    a, _, _, _ = fixture_pair()
    reordered = n.Mesh(a.name, a.vertices + a.origin, np.roll(a.faces, 1, axis=1))
    assert reordered.resolution == pytest.approx(a.resolution)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True, "0.01"])
def test_invalid_noise_floor_is_rejected(value):
    with pytest.raises(ValueError, match="noise_floor_mm"):
        n.Options(noise_floor_mm=value).validate()


@pytest.mark.parametrize("noise_mm", [0.0, 0.001])
def test_cli_noise_override_wins_over_specimen_and_auto_settings(tmp_path, noise_mm):
    a, b, cfg, _ = fixture_pair()
    source = tmp_path / "input"
    source.mkdir()
    for mesh in (a, b):
        n.write_mesh(source / f"{mesh.name}.ply", mesh, n.rigid(translation=mesh.origin))
    cfg["options"] = {
        "auto_noise_floor": True,
        "noise_floor_mm": 0.05,
        "samples": 32,
        "target_samples": 128,
        "collision_samples": 32,
        "starts": 1,
        "sensitivity": False,
    }
    (source / "neutral_config.json").write_text(json.dumps(cfg))
    output = tmp_path / "result"
    assert n.main([str(source), "--specimen", "--out", str(output), "--noise-floor-mm", str(noise_mm)]) == 0
    report = json.loads((output / "neutral_report.json").read_text())
    assert report["joints"][0]["effective_tolerances"]["noise_floor_mm"] == noise_mm
    assert report["joints"][0]["effective_tolerances"]["noise_floor_source"] == "explicit"
    assert "noise_floor_estimated_from_mesh" not in report["joints"][0]["review_reasons"]


@pytest.mark.parametrize("remove", [False, True])
def test_component_decisions_reach_fit_quality_report(remove):
    a, b, cfg, _ = fixture_pair()
    island = vtk.vtkSphereSource()
    island.SetCenter(4, 0, 0)
    island.SetRadius(0.05)
    island.Update()
    app = vtk.vtkAppendPolyData()
    app.AddInputData(n.polydata(a.vertices + a.origin, a.faces))
    app.AddInputData(island.GetOutput())
    app.Update()
    a = n.Mesh.from_polydata(a.name, app.GetOutput(), keep_largest_component=remove)
    _, _, report, _ = n.fit_column([a, b], cfg, fast_options())
    reason = "mesh_components_removed" if remove else "disconnected_mesh_components"
    assert reason in report["joints"][0]["review_reasons"]
    assert report["status"] == "needs_review"
    assert report["mesh_quality"][0]["connected_components"] == (1 if remove else 2)


def test_sphere_noise_allowance_does_not_depend_on_joint_length(monkeypatch):
    a, b, cfg, _ = fixture_pair()
    observed = []
    original = n.fit_sphere

    def record(mesh, ids, noise_fraction=0.0):
        observed.append(noise_fraction * mesh.scale)
        return original(mesh, ids, noise_fraction)

    monkeypatch.setattr(n, "fit_sphere", record)
    opts = fast_options(noise_floor_mm=0.012)
    n.Joint(a, b, cfg, opts)
    changed = copy.deepcopy(cfg)
    changed["mesh_scales"] = {a.name: 10.0, b.name: 5.0}
    n.Joint(a, b, changed, opts)
    assert observed == pytest.approx([0.012] * 4)
