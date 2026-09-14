"""Regressions against independent anatomical-frame and articulation ground truth."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from neutral_pose import anatomy, auto
from neutral_pose import core as n
from test_neutral_pose import fast_options, fixture_pair, wedge
from test_realistic import marching_cubes_vertebra


def framed_pair(angle=14.0, scale=1.0):
    a, b, cfg, expected = fixture_pair(
        angle=np.deg2rad(angle), scale=scale, rotation=(6.0, -13.0, 11.0), translation=(0.10, -0.07, 0.05)
    )
    rotation = Rotation.from_euler("x", angle, degrees=True).as_matrix()
    truth = n.rigid(rotation, scale * (np.array([0.0, 0.0, 0.5]) + 0.52 * rotation[:, 2]))
    observed = n.inverse(expected) @ truth
    cfg["neutral_frames"] = {
        "v01": {"axes": np.eye(3).tolist(), "center": [0.0, 0.0, 0.0]},
        "v02": {"axes": observed[:3, :3].tolist(), "center": observed[:3, 3].tolist()},
    }
    return a, b, cfg, expected, truth, observed


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0])
def test_lateral_bend_and_twist_removed_while_sagittal_curvature_is_recovered(scale):
    a, b, cfg, expected, truth, observed = framed_pair(scale=scale)
    M, poses, report, joints = n.fit_column([a, b], cfg, fast_options())
    actual = M[1] @ observed
    assert np.allclose(actual[:3, :3][:, 0], [1.0, 0.0, 0.0], atol=1e-10)
    assert abs(actual[0, 3]) < 1e-10 * scale
    # The independent synthetic articulation has 14 degrees sagittal bend.
    angle = np.rad2deg(np.arctan2(actual[2, 1], actual[1, 1]))
    assert angle == pytest.approx(14.0, abs=0.4)
    error = n.pose_difference(M[1], expected, scale)
    assert np.linalg.norm(error[3:]) < 0.009
    assert np.allclose(M[0], np.eye(4))
    assert all(np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-10) for T in M)
    assert len(report["joints"][0]["uncertainty"]["jacobian_singular_values"]) == 3


@pytest.mark.parametrize("failure", ["ambiguous", "penetration"])
def test_uncertain_or_worse_fit_fallback_cannot_restore_lateral_bend_or_twist(failure):
    a, b, cfg, _, _, observed = framed_pair()
    _, poses, report, joints = n.fit_column([a, b], cfg, fast_options())
    if failure == "ambiguous":
        report["joints"][0]["uncertainty"]["ambiguous"] = True
    else:
        joint = joints[0]
        q = joint.neutral_plane.encode(n.inverse(poses[0]) @ poses[1], joint.scale)
        q[2] -= 0.3
        poses[1] = poses[0] @ joint.neutral_plane.decode(q, joint.scale)
        entry = report["joints"][0]
        entry["metrics"] = joint.metrics(n.inverse(poses[0]) @ poses[1], final=True)
        for key in ("ambiguous", "sensitive", "centering_dependent", "poorly_identified"):
            entry["uncertainty"][key] = False
    M, selected, report = auto.select_supported_poses([a, b], poses, report, joints)
    actual = M[1] @ observed
    assert np.allclose(actual[:3, 0], [1.0, 0.0, 0.0], atol=1e-10)
    assert abs(actual[0, 3]) < 1e-10
    assert not np.allclose(M[1], np.eye(4))
    assert report["automatic_pose_selection"]["input_joints_retained"] == 0
    assert report["automatic_pose_selection"]["sagittal_reference_joints"] == 1
    entry = report["joints"][0]
    assert entry["automatic_selection"]["choice"] == "neutral_sagittal_reference"
    if failure == "penetration":
        assert "automatic_fit_increased_penetration" in entry["automatic_selection"]["reasons"]
    assert np.allclose(entry["transform_b_local_to_a_local"], n.inverse(selected[0]) @ selected[1])
    assert np.allclose(entry["anatomical_orientation_b_in_a"], actual[:3, :3])


def test_global_refinement_and_fallback_keep_all_bones_in_one_anatomical_plane():
    angle = np.deg2rad(8.0)
    R = Rotation.from_euler("x", angle).as_matrix()
    step = n.rigid(R, np.array([0.0, 0.0, 0.5]) + 0.52 * R[:, 2])
    world = n.rigid(
        Rotation.from_euler("xyz", [31.0, -24.0, 67.0], degrees=True).as_matrix(), [7.0, -4.0, 12.0]
    )
    truth = np.eye(4)
    meshes = []
    patches = []
    observations = []
    frames = {}
    for i in range(4):
        distortion = Rotation.from_euler("yz", [9 * i, -6 * i], degrees=True).as_matrix()
        observed = world @ n.rigid(distortion @ truth[:3, :3], truth[:3, 3] + [0.05 * i, 0.02 * i, 0.0])
        mesh, patch = wedge(f"v{i + 1:02d}", angle=angle, world=observed)
        frames[mesh.name] = {"axes": observed[:3, :3].tolist(), "center": observed[:3, 3].tolist()}
        meshes.append(mesh)
        patches.append(patch)
        observations.append(observed)
        truth = truth @ step
    cfg = {"mesh_coordinate_system": "RAS", "neutral_frames": frames, "joints": []}
    for i in range(3):
        cfg["joints"].append(
            {
                "a": meshes[i].name,
                "b": meshes[i + 1].name,
                "patches": [
                    {
                        "name": str(k),
                        "kind": "centrum" if k == 0 else "facet",
                        "a": {"faces": patches[i][1][k].tolist()},
                        "b": {"faces": patches[i + 1][0][k].tolist()},
                        "gap_fraction": 0.02,
                        "gap_tolerance_fraction": 0.0,
                    }
                    for k in range(3)
                ],
            }
        )
    M, poses, report, joints = n.fit_column(meshes, cfg, fast_options())
    assert report["global_refinement"]["free_parameters_per_moving_bone"] == 3
    for matrix, observed in zip(M, observations):
        recovered = n.inverse(world) @ matrix @ observed
        assert np.allclose(recovered[:3, 0], [1.0, 0.0, 0.0], atol=1e-9)
        assert abs(recovered[0, 3]) < 1e-9
    assert np.rad2deg(np.arctan2(recovered[2, 1], recovered[1, 1])) == pytest.approx(24.0, abs=1.0)
    report["joints"][1]["uncertainty"]["ambiguous"] = True
    M, poses, report = auto.select_supported_poses(meshes, poses, report, joints)
    for matrix, observed in zip(M, observations):
        recovered = n.inverse(world) @ matrix @ observed
        assert np.allclose(recovered[:3, 0], [1.0, 0.0, 0.0], atol=1e-9)
        assert abs(recovered[0, 3]) < 1e-9
    assert np.allclose(M[0], np.eye(4), atol=1e-10)


def test_whole_surface_symmetry_recovers_known_plane_from_biased_contact_seed():
    world = n.rigid(
        Rotation.from_euler("xyz", [28.0, -39.0, 63.0], degrees=True).as_matrix(), [13.0, -8.0, 19.0]
    )
    mesh = marching_cubes_vertebra("bone", world, voxel=0.055, noise=0.0, seed=9)
    seed = world[:3, :3] @ Rotation.from_euler("yz", [7.0, -5.0], degrees=True).as_matrix()[:, 0]
    normal, center, evidence = anatomy.symmetry_plane(
        mesh, seed, world[:3, 3] - mesh.origin + 0.025 * world[:3, 0], samples=768
    )
    error = np.rad2deg(np.arccos(np.clip(abs(normal @ world[:3, 0]), 0.0, 1.0)))
    assert error < 0.3
    assert abs((center + mesh.origin - world[:3, 3]) @ world[:3, 0]) < 0.003
    assert not evidence["review_reasons"]


def test_bilateral_spacing_pools_signed_evidence_before_clamping():
    definition = {"patches": [{"name": "left", "kind": "facet"}, {"name": "right", "kind": "facet"}]}
    receipt = {
        "local_scale_mm": 2.0,
        "patches": [
            {"name": "left", "estimated_gap_mm": 0.0, "input_signed_gap_quantiles_mm": [-0.1, -0.04, 0.0]},
            {"name": "right", "estimated_gap_mm": 0.08, "input_signed_gap_quantiles_mm": [0.0, 0.08, 0.1]},
        ],
    }
    anatomy.pool_bilateral_spacing({"joints": [definition]}, {"joints": [receipt]})
    assert all(p["gap_fraction"] == pytest.approx(0.01) for p in definition["patches"])
    assert all(p["estimated_gap_mm"] == pytest.approx(0.02) for p in receipt["patches"])


def test_unidentified_anatomy_is_reported_instead_of_exporting_a_curved_neutral_pose():
    a, b, cfg, _ = fixture_pair()
    cfg = copy.deepcopy(cfg)
    cfg["joints"][0]["patches"] = cfg["joints"][0]["patches"][:1]
    with pytest.raises(ValueError, match="cannot infer a neutral plane"):
        anatomy.infer_frames([a, b], cfg)
