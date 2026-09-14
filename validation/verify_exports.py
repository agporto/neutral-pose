"""Independently check mesh, landmark, matrix, precision and contact exports."""

import argparse
import json
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parent
from neutral_pose import auto, support
from neutral_pose import core as n


def vtp(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    p = reader.GetOutput()
    if p.GetNumberOfPoints() == 0 or p.GetNumberOfPolys() == 0:
        raise ValueError(f"Empty/unreadable export: {path}")
    return vtk_to_numpy(p.GetPoints().GetData()).copy(), n.faces_of(p).copy()


def ply_points(path):
    with Path(path).open("rb") as f:
        header = []
        for _ in range(100):
            line = f.readline()
            header.append(line.decode("ascii").strip())
            if line.strip() == b"end_header":
                break
        else:
            raise ValueError("Invalid PLY header")
        count = int(next(x for x in header if x.startswith("element vertex ")).split()[-1])
        assert all("property double " + axis in header for axis in "xyz")
        data = np.fromfile(f, dtype=[("xyz", "<f8", (3,)), ("rgb", "u1", (3,))], count=count)
        return data["xyz"].copy()


def verify(source, output):
    report = json.loads((output / "neutral_report.json").read_text())
    info = json.loads((output / "transforms.json").read_text())
    matrices = np.load(output / "neutral_transforms.npy")
    paths, lm_paths = auto.input_files(source)
    assert len(paths) == len(matrices) == len(info["meshes"])
    assert np.allclose(matrices[0], np.eye(4), rtol=0, atol=1e-9)
    errors = []
    landmark_errors = []
    records = []
    combined = []
    offset = 0
    combined_faces = []
    origins = []
    for path, lmpath, M, entry in zip(paths, lm_paths, matrices, info["meshes"]):
        n.validate_rigid(M)
        # Reread/clean the source independently using the public mesh reader.
        mesh = n.Mesh.read(path, reject_self_intersections=False)
        origins.append(mesh.origin.copy())
        assert entry["name"] == mesh.name
        assert np.array_equal(M, np.array(entry["matrix_input_to_neutral"]))
        assert np.allclose(n.inverse(M), entry["matrix_neutral_to_input"], rtol=0, atol=1e-9)
        V, F = vtp(output / "meshes_vtp" / f"neutral_{mesh.name}.vtp")
        expected = n.transform(M, mesh.vertices + mesh.origin)
        assert V.shape == expected.shape
        error = float(np.max(np.abs(V - expected)))
        assert error < max(1e-8, 1e-10 * mesh.scale)
        assert np.array_equal(np.sort(F, axis=1), np.sort(mesh.faces, axis=1))
        assert np.array_equal(ply_points(output / f"neutral_{mesh.name}.ply"), V)
        assert np.isclose(np.linalg.det(M[:3, :3]), 1.0, atol=1e-10)
        lm = n.read_landmarks(lmpath, info["coordinate_system"])
        posed = n.read_landmarks(
            output / "LMKs_json" / f"neutral_{mesh.name}.mrk.json", info["coordinate_system"]
        )
        assert lm.labels == posed.labels
        lmerror = float(np.max(np.abs(posed.points - n.transform(M, lm.points))))
        assert lmerror < 1e-9
        errors.append(error)
        landmark_errors.append(lmerror)
        combined.append(V)
        combined_faces.append(F + offset)
        offset += len(V)
        records.append(dict(mesh=mesh.name, vertices=len(V), triangles=len(F), landmarks=len(lm.points)))
    scene, faces = vtp(output / "column_neutral.vtp")
    assert np.array_equal(scene, np.vstack(combined))
    assert np.array_equal(np.sort(faces, axis=1), np.sort(np.vstack(combined_faces), axis=1))
    assert np.array_equal(ply_points(output / "column_neutral.ply"), scene)
    frames = report["configuration"]["neutral_frames"]
    anchors = np.array(
        [n.transform(M, np.array(frames[e["name"]]["center"])) for M, e in zip(matrices, info["meshes"])]
    )
    lateral = np.array(frames[info["meshes"][0]["name"]]["axes"])[:, 0]
    offset_error = float(np.max(np.abs((anchors - anchors[0]) @ lateral)))
    assert offset_error < 1e-8
    for M, entry in zip(matrices, info["meshes"]):
        axis = M[:3, :3] @ np.array(frames[entry["name"]]["axes"])[:, 0]
        assert np.linalg.norm(np.cross(lateral, axis)) < 1e-8 and lateral @ axis > 0
    for i, joint in enumerate(report["joints"]):
        # Translation of the local origins is independent of the report H.
        poses = [matrices[k] @ n.rigid(translation=origins[k]) for k in (i, i + 1)]
        assert np.allclose(
            n.inverse(poses[0]) @ poses[1], joint["transform_b_local_to_a_local"], rtol=0, atol=2e-8
        )
    result = dict(
        bones=len(records),
        landmarks=sum(r["landmarks"] for r in records),
        max_vertex_matrix_error_mm=max(errors),
        max_landmark_matrix_error_mm=max(landmark_errors),
        first_bone_fixed=True,
        proper_rigid_transforms=True,
        topology_preserved=True,
        landmark_labels_and_order_preserved=True,
        double_ply_equals_vtp=True,
        combined_vtp_equals_concatenated_individuals=True,
        relative_transforms_verified=True,
        maximum_lateral_center_offset_mm=offset_error,
        sagittal_reference_joints=report["automatic_pose_selection"]["sagittal_reference_joints"],
        penetration_exceeds_tolerance=[
            i + 1
            for i, j in enumerate(report["joints"])
            if "penetration_exceeds_tolerance" in j["review_reasons"]
        ],
        contacts_below_support_threshold=[
            i + 1
            for i, j in enumerate(report["joints"])
            if support.contact_support(j["metrics"])["unsupported"]
        ],
        nonadjacent_intersections=sum(x["intersection_detected"] for x in report["nonadjacent_contacts"]),
        unverified_mesh_signs=sum(not x["signed_distance_reliable"] for x in report["mesh_quality"]),
        status=report["status"],
        records=records,
    )
    (output / "export_verification.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted original specimen folder")
    parser.add_argument("output", type=Path, help="Corresponding automatic result folder")
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.output), indent=2))
