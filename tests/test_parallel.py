"""Parallel execution must reproduce the serial results exactly, not approximately."""

import copy
import json
import multiprocessing
import pickle
import zipfile

import numpy as np
import pytest

from neutral_pose import auto, parallel
from neutral_pose import core as n
from test_realistic import marching_cubes_vertebra


def test_mesh_pickle_roundtrip_rebuilds_identical_geometry_queries():
    mesh = marching_cubes_vertebra("v", n.rigid(translation=[1.0, 2.0, 3.0]), voxel=0.06, noise=0.2, seed=3)
    clone = pickle.loads(pickle.dumps(mesh))
    assert set(clone.__dict__) == set(mesh.__dict__)
    assert np.array_equal(clone.vertices, mesh.vertices) and np.array_equal(clone.faces, mesh.faces)
    probes = np.random.default_rng(1).uniform(-1.2, 1.2, (4000, 3))
    assert np.array_equal(clone.signed_distance(probes), mesh.signed_distance(probes))
    assert clone.poly.GetBounds() == mesh.poly.GetBounds()
    assert np.array_equal(n.faces_of(clone.poly), n.faces_of(mesh.poly))


def test_resolve_workers():
    assert parallel.resolve_workers(1) == 1
    assert parallel.resolve_workers("3") == 3
    assert parallel.resolve_workers("auto") >= 1
    assert parallel.resolve_workers(None) >= 1
    with pytest.raises(ValueError):
        parallel.resolve_workers(0)


def _square(shared, index):
    return shared[index] ** 2


def test_parallel_map_preserves_order_and_matches_serial():
    data = list(range(7))
    assert list(parallel.parallel_map(_square, len(data), data, 1)) == [x * x for x in data]
    assert list(parallel.parallel_map(_square, len(data), data, 3)) == [x * x for x in data]


def _write_specimen(tmp_path, count=3):
    source = tmp_path / "input" / "specimen"
    source.mkdir(parents=True)
    for i in range(count):
        mesh = marching_cubes_vertebra(
            f"v{i + 1:02d}", n.rigid(translation=[10.0, -7.0, 3.0 + 1.02 * i]), voxel=0.06, noise=0.0, seed=i
        )
        n.write_mesh(source / f"{mesh.name}.ply", mesh, n.rigid(translation=mesh.origin))
        n.write_landmarks(
            source / "LMKs" / f"{mesh.name}.mrk.json",
            mesh.landmarks.points,
            [f"F-{k + 1}" for k in range(len(mesh.landmarks.labels))],
            "LPS",
        )
    archive = tmp_path / "bones.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in source.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(source.parent))
    return archive


def _outputs(folder):
    report = json.loads((folder / "neutral_report.json").read_text())
    report["software"]["version"] = None
    setup = json.loads((folder / "automatic_setup.json").read_text())
    return (
        np.load(folder / "neutral_transforms.npy"),
        report,
        setup,
        (folder / "joint_metrics.csv").read_bytes(),
        sorted((p.name, p.read_bytes()) for p in (folder / "meshes_vtp").iterdir()),
    )


@pytest.mark.parametrize("start_method", ["default", "spawn"])
def test_parallel_run_is_bitwise_identical_to_serial(tmp_path, monkeypatch, start_method):
    archive = _write_specimen(tmp_path)
    if start_method == "spawn":
        monkeypatch.setattr(parallel, "_context", lambda: multiprocessing.get_context("spawn"))
    assert auto.run(archive, tmp_path / "serial", workers=1)[0]["status"] in {"accepted", "needs_review"}
    assert auto.run(archive, tmp_path / "parallel", workers=2)[0]["status"] in {"accepted", "needs_review"}
    serial, parallel_ = _outputs(tmp_path / "serial"), _outputs(tmp_path / "parallel")
    assert np.array_equal(serial[0], parallel_[0])
    assert serial[1:] == parallel_[1:]


def test_discovery_stays_serial_when_symmetry_planes_are_not_landmark_fixed(tmp_path, monkeypatch):
    """Adjacent discoveries share cached surface planes, so they must not be parallelised."""
    archive = _write_specimen(tmp_path, count=3)
    tmp = tmp_path / "x"
    auto.safe_extract(archive, tmp)
    folder = auto.specimen_folders(tmp)[0]
    seen = []
    original = parallel.parallel_map

    def spy(task, count, shared, workers):
        seen.append((task.__name__, parallel.resolve_workers(workers)))
        return original(task, count, shared, workers)

    monkeypatch.setattr(auto, "parallel_map", spy)
    meshes, config, options, inference = auto.prepare_specimen(folder, workers=4)
    assert inference["landmark_symmetry"]["surface_fallback_count"] == len(meshes)
    assert seen == [("_discover_joint_task", 1)]
    # The serial pass left every mesh with the plane its neighbour seeded.
    assert all(getattr(m, "_automatic_bilateral_plane", None) is not None for m in meshes)


def test_fit_column_and_selection_parallel_match_serial(tmp_path):
    archive = _write_specimen(tmp_path)
    tmp = tmp_path / "x"
    auto.safe_extract(archive, tmp)
    meshes, config, options, inference = auto.prepare_specimen(auto.specimen_folders(tmp)[0])
    serial = n.fit_column(meshes, copy.deepcopy(config), options, workers=1)
    parallel_ = n.fit_column(meshes, copy.deepcopy(config), options, workers=2)
    assert np.array_equal(serial[0], parallel_[0])
    assert serial[2] == parallel_[2]
    matrices, poses, report, joints = serial
    a = auto.select_supported_poses(meshes, poses, report, joints, workers=1)
    b = auto.select_supported_poses(meshes, poses, report, joints, workers=2)
    assert np.array_equal(a[0], b[0])
    assert a[2] == b[2]
