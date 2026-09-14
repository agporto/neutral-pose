"""Behavioral tests for the one-command specimen workflow."""

import json
import zipfile

import numpy as np
import pytest

from neutral_pose import auto, discovery, surfaces
from neutral_pose import core as n
from test_neutral_pose import fast_options, fixture_pair


def write_pair(folder, gap=0.02, coordinate="LPS"):
    folder.mkdir(parents=True)
    a, b, _, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    b = n.Mesh(b.name, b.vertices + b.origin + [0, 0, gap - 0.02], b.faces)
    # Translation breaks artificial RAS/LPS symmetry and tests geometric selection.
    for mesh in (a, b):
        T = n.rigid(translation=mesh.origin + [10.0, -7.0, 3.0])
        n.write_mesh(folder / f"{mesh.name}.ply", mesh, T)
        points = n.transform(T, mesh.vertices[[0, 10, 24, 25, 35, 49]])
        n.write_landmarks(
            folder / "LMKs" / f"{mesh.name}.mrk.json", points, [f"F-{i + 1}" for i in range(6)], coordinate
        )


@pytest.mark.parametrize("coordinate", ["RAS", "LPS"])
def test_coordinate_system_is_inferred_from_landmark_surface_match(tmp_path, coordinate):
    folder = tmp_path / "specimen"
    write_pair(folder, coordinate=coordinate)
    paths, lms = auto.input_files(folder)
    meshes = [n.Mesh.read(p) for p in paths]
    result = auto.infer_coordinates(meshes, lms)
    assert result["coordinate_system"] == coordinate
    assert result["geometrically_resolved"]
    assert all(m.landmarks is not None for m in meshes)


@pytest.mark.parametrize("gap", [0.02, 0.08])
def test_spacing_comes_from_input_surfaces_not_a_fixed_gap(tmp_path, gap):
    folder = tmp_path / "specimen"
    write_pair(folder, gap=gap)
    paths, lms = auto.input_files(folder)
    meshes = [n.Mesh.read(p) for p in paths]
    auto.infer_coordinates(meshes, lms)
    definition, evidence = discovery.infer_joint(*meshes, 0)
    config = {"mesh_coordinate_system": "LPS", "joints": [definition]}
    options = fast_options()
    patches = evidence["patches"]
    assert patches
    for patch in patches:
        assert patch["estimated_gap_mm"] == pytest.approx(gap, abs=1e-5)
        assert patch["spacing_source"] == "median_signed_normal_separation_in_input_arrangement"
    pair = n.Joint(*meshes, config, options).pairs[0]
    smaller = n.Joint(*meshes, config, options, radius_factor=0.85).pairs[0]
    assert len(smaller.a.ids) < len(pair.a.ids)
    assert np.allclose(pair.a.anchor, smaller.a.anchor)
    assert pair.a.anchor_source == "inferred_surface_center"


def test_automatic_zip_run_needs_no_config_and_preserves_landmarks(tmp_path):
    from test_realistic import marching_cubes_vertebra

    source = tmp_path / "input" / "specimen"
    source.mkdir(parents=True)
    for i in range(2):
        mesh = marching_cubes_vertebra(
            f"v{i + 1:02d}", n.rigid(translation=[10.0, -7.0, 3.0 + 1.02 * i]), voxel=0.055, noise=0.0, seed=i
        )
        n.write_mesh(source / f"{mesh.name}.ply", mesh, n.rigid(translation=mesh.origin))
        n.write_landmarks(
            source / "LMKs" / f"{mesh.name}.mrk.json",
            mesh.landmarks.points,
            [f"F-{k + 1}" for k in range(len(mesh.landmarks.labels))],
            "LPS",
        )
    # Nested analysis tables are not additional specimens.
    tables = source / "LMKs" / "analysis"
    tables.mkdir()
    (tables / "meanShape.csv").write_text("x,y,z\n0,0,0\n")
    archive = tmp_path / "bones.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in source.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(source.parent))
    assert auto.main([str(archive)]) == 0
    output = tmp_path / "bones_neutral"
    report = json.loads((output / "neutral_report.json").read_text())
    assert report["mesh_order"] == ["v01", "v02"]
    assert report["mesh_coordinate_system"] == "LPS"
    assert "automatically_inferred_contact_reference" in report["joints"][0]["review_reasons"]
    assert (output / "automatic_setup.json").is_file()
    assert (output / "automatic_inference.json").is_file()
    assert (output / "articulation_preview.png").is_file()
    assert report["anatomical_neutrality"]["output_max_absolute_lateral_center_offset_mm"] < 1e-8
    assert report["anatomical_neutrality"]["output_max_symmetry_plane_misalignment_deg"] < 1e-6
    assert report["automatic_pose_selection"]["input_joints_retained"] == 0
    matrices = np.load(output / "neutral_transforms.npy")
    assert np.allclose(np.linalg.det(matrices[:, :3, :3]), 1.0)
    assert np.allclose(matrices[0], np.eye(4))
    for i, name in enumerate(report["mesh_order"]):
        original = n.read_landmarks(source / "LMKs" / f"{name}.mrk.json", "LPS")
        result = n.read_landmarks(output / "LMKs_json" / f"neutral_{name}.mrk.json", "LPS")
        assert result.labels == original.labels
        assert np.allclose(result.points, n.transform(matrices[i], original.points))
    with pytest.raises(FileExistsError):
        auto.run(archive)


def test_automatic_preview_reports_self_intersections_instead_of_certifying_signs():
    a, b, _, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    V = np.vstack([a.vertices, b.vertices + [0.25, 0.1, 0.05]])
    F = np.vstack([a.faces, b.faces + len(a.vertices)])
    with pytest.raises(ValueError, match="intersect"):
        n.Mesh("overlap", V, F)
    mesh = n.Mesh("overlap", V, F, reject_self_intersections=False)
    assert mesh.self_intersection_checked
    assert mesh.self_intersection is not None
    assert mesh.topologically_closed
    assert not mesh.closed


def test_zip_cannot_write_outside_extraction_folder(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../outside.txt", "bad")
    with pytest.raises(ValueError, match="outside"):
        auto.safe_extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.txt").exists()


def test_numbered_landmark_prefixes_do_not_imply_anatomical_roles():
    assert surfaces.canonical_label("F_4-21") == "F-21"
    assert surfaces.canonical_label("F-21") == "F-21"
    assert surfaces.canonical_label("other_21") == "other_21"


def test_incompatible_declared_units_are_rejected(tmp_path):
    p = tmp_path / "mesh.ply"
    p.write_text("ply\ncomment [Unit:Meter]\nend_header\n")
    with pytest.raises(ValueError, match="millimeters"):
        auto.mesh_units([p])


def test_duplicate_mesh_formats_are_rejected(tmp_path):
    (tmp_path / "v01.ply").touch()
    (tmp_path / "v01.stl").touch()
    with pytest.raises(ValueError, match="same basename"):
        auto.input_files(tmp_path)


def test_automatic_run_cannot_overwrite_its_source_zip(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("original.txt", "keep")
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="original input"):
        auto.run(archive, archive, overwrite=True)
    assert archive.read_bytes() == original


def test_existing_output_folders_are_not_rediscovered(tmp_path):
    specimen = tmp_path / "specimen"
    specimen.mkdir()
    output = tmp_path / "old_result"
    output.mkdir()
    for folder in (specimen, output):
        (folder / "v01.ply").touch()
        (folder / "v02.ply").touch()
    (output / "neutral_report.json").write_text("{}")
    assert auto.specimen_folders(tmp_path) == [specimen]


def test_ambiguous_pose_change_retains_original_local_articulation():
    a, b, cfg, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    _, poses, report, joints = n.fit_column([a, b], cfg, fast_options())
    poses[1] = n.perturb(poses[1], [0, 0, np.deg2rad(11), 0.1, 0, 0], joints[0].scale)
    report["joints"][0]["uncertainty"]["ambiguous"] = True
    matrices, selected, updated = auto.select_supported_poses([a, b], poses, report, joints)
    assert np.allclose(n.inverse(selected[0]) @ selected[1], joints[0].initial)
    assert np.allclose(matrices, np.tile(np.eye(4), (2, 1, 1)))
    assert updated["automatic_pose_selection"]["input_joints_retained"] == 1
    assert "input_pose_retained_due_to_uncertain_automatic_fit" in updated["joints"][0]["review_reasons"]
    assert not np.allclose(
        updated["joints"][0]["proposed_fit"]["transform_b_local_to_a_local"], joints[0].initial
    )


def test_pose_selection_rejects_increased_penetration_even_without_ambiguity():
    a, b, cfg, _ = fixture_pair(rotation=(0, 0, 0), translation=(0, 0, 0))
    _, poses, report, joints = n.fit_column([a, b], cfg, fast_options())
    poses[1] = n.rigid(translation=[0, 0, -0.15]) @ poses[1]
    report["joints"][0]["metrics"] = joints[0].metrics(n.inverse(poses[0]) @ poses[1], final=True)
    for key in ("ambiguous", "sensitive", "centering_dependent", "poorly_identified"):
        report["joints"][0]["uncertainty"][key] = False
    matrices, _, updated = auto.select_supported_poses([a, b], poses, report, joints)
    decision = updated["automatic_pose_selection"]["decisions"][0]
    assert "automatic_fit_increased_penetration" in decision["reasons"]
    assert np.allclose(matrices, np.tile(np.eye(4), (2, 1, 1)))
