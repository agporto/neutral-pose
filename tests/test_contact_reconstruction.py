"""Regressions for the reconstructed contact policy and precision guarantees."""

from types import SimpleNamespace

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from neutral_pose import auto, contacts, recovery, support
from neutral_pose import core as n
from test_anatomical_neutral import framed_pair
from test_neutral_pose import fast_options, wedge


def metrics(coverage=(0.8, 0.8, 0.8), penetration=0.0, names=("centrum", "left", "right")):
    return dict(
        max_sampled_penetration=penetration,
        patches=[
            dict(
                name=name,
                kind="centrum" if i == 0 else "facet",
                a_to_b={"coverage_fraction": value},
                b_to_a={"coverage_fraction": value},
            )
            for i, (name, value) in enumerate(zip(names, coverage))
        ],
    )


def test_coverage_average_cannot_hide_loss_or_exchange_of_named_contacts():
    # The average improves, but one specific articulation is lost.
    assert "automatic_fit_lost_required_contact" in support.selection_reasons(
        metrics((0.4, 0.4, 0.4)), metrics((1.0, 1.0, 0.0)), 0.01, 1e-6
    )
    # Equal numbers of missing contacts are not equivalent anatomy.
    assert "automatic_fit_lost_required_contact" in support.selection_reasons(
        metrics((1.0, 0.0, 1.0)), metrics((1.0, 1.0, 0.0)), 0.01, 1e-6
    )
    assert support.selection_reasons(metrics(), metrics(names=("centrum", "left", "new")), 0.01, 1e-6)


def test_feasible_supported_fit_beats_colliding_reference_with_higher_coverage():
    assert (
        support.selection_reasons(
            metrics((1.0, 1.0, 1.0), 0.09), metrics((0.7, 0.7, 0.7), 0.009), 0.01, 0.001
        )
        == []
    )
    assert support.selection_reasons(
        metrics((0.7, 0.7, 0.7), 0.009), metrics((1.0, 1.0, 1.0), 0.09), 0.01, 0.001
    )


def test_absence_threshold_does_not_change_the_qc_coverage_requirement():
    assert support.contact_support(metrics((0.3, 0.3, 0.3)))["unsupported"] == 0
    assert n.Options().minimum_coverage == 0.6
    assert support.contact_support(metrics((0.3, 0.09, 0.3)))["missing_contacts"] == ["left"]


def training_fixture(conflicting=False):
    # Arbitrary semantic labels: there are no numbered anatomical roles.
    labels = ["socket", "ball", "alpha", "beta", "gamma", "delta"]
    meshes = [
        SimpleNamespace(
            name=f"b{i}",
            scale=1.0,
            landmarks=n.Landmarks(np.zeros((6, 3)), labels, "LPS"),
            _landmark_symmetry_status={"status": "landmark_plane"},
        )
        for i in range(6)
    ]
    definitions, evidence = [], []
    for i in range(5):
        assignments = [("ball", "socket"), ("alpha", "gamma"), ("beta", "delta")]
        if conflicting and i in (1, 3):
            assignments[1:] = [("alpha", "delta"), ("beta", "gamma")]
        definition = dict(a=f"b{i}", b=f"b{i + 1}", patches=[])
        receipt = dict(patches=[], bilateral_contact_selection={"supported": True})
        for k, (a, b) in enumerate(assignments):
            kind, name = ("centrum", "c") if k == 0 else ("facet", str(k))
            boundary = dict(radius_mm=0.3, normal=[0.0, 0.0, 1.0])
            definition["patches"].append(
                dict(
                    name=name,
                    kind=kind,
                    gap_fraction=0.01,
                    a={"automatic_boundary": boundary},
                    b={"automatic_boundary": boundary},
                )
            )
            receipt["patches"].append(
                dict(
                    name=name,
                    inferred_kind=kind,
                    landmark_support_a=[{"label": a}],
                    landmark_support_b=[{"label": b}],
                )
            )
        definitions.append(definition)
        evidence.append(receipt)
    return meshes, {"joints": definitions}, {"joints": evidence}


def test_roles_are_learned_from_arbitrary_identities(monkeypatch):
    meshes, cfg, receipt = training_fixture()
    monkeypatch.setattr(
        contacts.anatomy,
        "infer_frames",
        lambda meshes, config: {m.name: {"axes": np.eye(3).tolist()} for m in meshes},
    )
    model = contacts.learn_model(meshes, cfg, receipt, list(range(5)))
    assert model["supporting_joints"] == 5
    assert model["roles"][0]["a"] == ["ball"]
    assert model["roles"][0]["b"] == ["socket"]
    assert {(r["a"][0], r["b"][0]) for r in model["roles"][1:]} == {("alpha", "gamma"), ("beta", "delta")}


def test_conflicting_correspondences_are_rejected():
    meshes, cfg, receipt = training_fixture(conflicting=True)
    with pytest.raises(ValueError, match="correspondence votes"):
        contacts.learn_model(meshes, cfg, receipt, list(range(5)))


def test_incomplete_contact_reference_requires_supported_training():
    cfg = {"joints": [dict(a="one", b="two", patches=[])]}
    inference = {"joints": [{"patches": []}]}
    with pytest.raises(ValueError, match="three complete"):
        recovery.complete_contacts([], cfg, inference, n.Options())


@pytest.mark.parametrize("values,expected", [([0.01] * 5 + [0.12], 1), ([0.12] * 6, 0), ([0.01, 0.12], 0)])
def test_spacing_uses_joint_votes_and_preserves_consistently_wide_gaps(values, expected):
    cfg, inference = {"joints": []}, {"joints": []}
    for value in values:
        cfg["joints"].append(
            {
                "patches": [dict(name="c", kind="centrum", gap_fraction=0.01)]
                + [dict(name=name, kind="facet", gap_fraction=value) for name in ("left", "right")]
            }
        )
        inference["joints"].append(
            {
                "local_scale_mm": 2.0,
                "patches": [dict(name=name, estimated_gap_mm=2 * value) for name in ("c", "left", "right")],
            }
        )
    ids = list(range(len(values)))
    stats = contacts.spacing_statistics(cfg, inference, ids)
    changed = contacts.adjust_spacing(cfg, inference, stats, ids)
    assert len(changed) == 2 * expected
    assert stats["facet"]["supporting_joints"] == len(values)
    if expected:
        assert cfg["joints"][-1]["patches"][1]["gap_fraction"] == pytest.approx(0.01)
        assert inference["joints"][-1]["patches"][1]["original_pooled_gap_mm"] == pytest.approx(0.24)


def test_contact_policy_keeps_known_curvature_despite_uncertainty():
    a, b, cfg, expected, truth, observed = framed_pair(angle=14.0)
    cfg["automatic_contact_policy"] = contacts.POLICY
    _, poses, report, joints = n.fit_column([a, b], cfg, fast_options())
    report["joints"][0]["uncertainty"]["ambiguous"] = True
    M, selected, result = auto.select_supported_poses([a, b], poses, report, joints)
    actual = M[1] @ observed
    assert np.rad2deg(np.arctan2(actual[2, 1], actual[1, 1])) == pytest.approx(14.0, abs=0.4)
    assert np.allclose(actual[:3, 0], [1.0, 0.0, 0.0], atol=1e-9)
    assert result["joints"][0]["uncertainty"]["ambiguous"]
    assert result["automatic_pose_selection"]["sagittal_reference_joints"] == 0


def penetrating_joint():
    a, b, cfg, _, _, _ = framed_pair()
    joint = n.Joint(a, b, cfg, fast_options())
    H = n.optimize_joint(joint, joint.initial).transform
    q = joint.neutral_plane.encode(H, joint.scale)
    q[2] -= 0.08
    return joint, joint.neutral_plane.decode(q, joint.scale)


def test_dense_constraint_repairs_penetration_without_releasing_anatomical_plane():
    joint, H = penetrating_joint()
    before = joint.metrics(H, final=True)["max_sampled_penetration"]
    assert before > joint.penetration_tolerance * joint.scale
    fixed, receipt = support.refine_dense_feasibility(joint, H)
    assert receipt["accepted"]
    after = joint.metrics(fixed, final=True)["max_sampled_penetration"]
    assert after <= joint.penetration_tolerance * joint.scale
    assert joint.neutral_plane.diagnostics(fixed)["lateral_axis_misalignment_deg"] < 1e-6
    assert abs(joint.neutral_plane.diagnostics(fixed)["lateral_center_offset_mm"]) < 1e-9
    assert receipt["tolerance_mm"] == joint.penetration_tolerance * joint.scale


def test_optimizer_success_cannot_replace_independent_dense_validation(monkeypatch):
    joint, H = penetrating_joint()
    monkeypatch.setattr(
        support,
        "minimize",
        lambda objective, x, **kwargs: SimpleNamespace(
            x=np.zeros_like(x),
            fun=objective(np.zeros_like(x)),
            success=True,
            message="claimed success",
            nit=1,
        ),
    )
    fixed, receipt = support.refine_dense_feasibility(joint, H)
    assert not receipt["accepted"]
    assert np.array_equal(fixed, H)
    assert receipt["reason"] == "no_candidate_passed_full_dense_validation"


def test_double_ply_and_vtp_preserve_small_geometry_at_large_world_offsets(tmp_path):
    mesh, _ = wedge("bone")
    T = n.rigid(translation=[-67816.123456789, 35591.312345679, -101797.512345679])
    expected = n.transform(T, mesh.vertices)
    ply, vtp = tmp_path / "bone.ply", tmp_path / "bone.vtp"
    n.write_mesh(ply, mesh, T)
    n.write_mesh(vtp, mesh, T)
    with ply.open("rb") as f:
        while f.readline().strip() != b"end_header":
            pass
        data = np.fromfile(f, dtype=[("xyz", "<f8", (3,)), ("rgb", "u1", (3,))], count=len(expected))
    assert np.array_equal(data["xyz"], expected)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp))
    reader.Update()
    assert np.array_equal(vtk_to_numpy(reader.GetOutput().GetPoints().GetData()), expected)
    assert np.array_equal(n.faces_of(reader.GetOutput()), mesh.faces)


def test_vtp_connectivity_survives_binary_block_boundary(tmp_path):
    # A power-of-two number of offsets exposed a compressed writer/reader error.
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.tile([0, 1, 2], (65535, 1))
    path = tmp_path / "boundary.vtp"
    n.write_vtp(path, n.polydata(points, faces))
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(path))
    r.Update()
    assert r.GetOutput().GetNumberOfPolys() == len(faces)
    assert np.array_equal(n.faces_of(r.GetOutput()), faces)
