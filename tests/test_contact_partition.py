"""Contact identity and merged-region regressions independent of specimen labels."""

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from neutral_pose import contacts, recovery
from neutral_pose import core as n


def test_joint_assignment_resolves_duplicate_nearest_choices():
    # The second role has only one eligible surface. The first must take its
    # alternative; each assigned surface remains inside the same distance bound.
    distances = np.array([[0.04, 0.06, 0.5], [0.041, 0.5, 0.5], [0.5, 0.5, 0.01]])
    chosen = contacts.assign_contact_roles(distances, ["body", "alpha", "beta"])
    assert chosen.tolist() == [1, 0, 2]


def test_joint_evidence_can_resolve_a_locally_ambiguous_match():
    distances = np.array([[0.01, 0.012, 0.5], [0.09, 0.012, 0.5], [0.5, 0.5, 0.01]])
    assert contacts.assign_contact_roles(distances, ["body", "alpha", "beta"]).tolist() == [0, 1, 2]


def test_globally_ambiguous_identities_are_still_rejected():
    distances = np.array([[0.01, 0.017, 0.5], [0.018, 0.01, 0.5], [0.5, 0.5, 0.01]])
    with pytest.raises(ValueError, match="near-equivalent one-to-one"):
        contacts.assign_contact_roles(distances, ["body", "alpha", "beta"])


@pytest.mark.parametrize(
    "distances,message",
    [
        ([[0.01], [0.02], [0.03]], "Only 1 candidate"),
        ([[0.01, 0.3, 0.4], [0.02, 0.5, 0.3], [0.4, 0.5, 0.01]], "No distinct contact assignment"),
        ([[0.11, 0.4, 0.5], [0.4, 0.01, 0.5], [0.4, 0.5, 0.01]], "No distinct contact assignment"),
    ],
)
def test_missing_or_distant_contacts_are_not_forced(distances, message):
    with pytest.raises(ValueError, match=message):
        contacts.assign_contact_roles(distances, ["body", "alpha", "beta"])


def test_assignment_is_equivariant_to_role_and_candidate_order():
    distances = np.array([[0.04, 0.06, 0.5, 0.7], [0.041, 0.5, 0.5, 0.6], [0.5, 0.5, 0.01, 0.6]])
    rows, columns = np.array([2, 0, 1]), np.array([3, 2, 0, 1])
    chosen = contacts.assign_contact_roles(distances[np.ix_(rows, columns)], ["b", "c", "a"])
    assert columns[chosen].tolist() == np.array([1, 0, 2])[rows].tolist()


def merged_surface():
    x, y = np.meshgrid(np.linspace(-2.0, 2.0, 17), np.linspace(-1.0, 1.0, 9))
    points = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    faces = []
    for row in range(8):
        for col in range(16):
            a = row * 17 + col
            faces += [[a, a + 1, a + 18], [a, a + 18, a + 17]]
    labels = ["central_seed", "seed_green", "seed_violet"]
    landmarks = n.Landmarks(np.array([[0.1, 0.2, 0.0], [-1.35, -0.3, 0.0], [1.4, 0.45, 0.0]]), labels, "LPS")
    mesh = n.Mesh.from_polydata("merged", n.polydata(points, np.array(faces)), landmarks=landmarks)
    model = {"roles": [dict(name=label, a=[label], b=[label]) for label in labels]}
    return mesh, model


def test_geodesic_partition_separates_roles_without_inventing_surface():
    mesh, model = merged_surface()
    vertices, faces = mesh.vertices.copy(), mesh.faces.copy()
    mask = np.ones(len(mesh.faces), bool)
    diagnostics = []
    parts = contacts.split_contact_components(mesh, mask, model, "a", diagnostics)
    assert len(parts) == 3
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(mesh.faces)))
    for ids in parts:
        assert connected_components(mesh.adjacency()[ids][:, ids], directed=False)[0] == 1
    assert np.array_equal(vertices, mesh.vertices) and np.array_equal(faces, mesh.faces)
    assert diagnostics[0]["parts_retained"] == [True, True, True]
    assert sum(diagnostics[0]["part_face_counts"]) == len(mesh.faces)


def test_partition_cannot_grow_outside_opposing_surface_mask():
    mesh, model = merged_surface()
    mask = (mesh.centers[:, 1] > -0.6) & (mesh.centers[:, 1] < 0.7)
    parts = contacts.split_contact_components(mesh, mask, model, "a")
    assert len(parts) == 3
    assert np.array_equal(np.sort(np.concatenate(parts)), np.flatnonzero(mask))


def test_absent_landmark_support_does_not_create_extra_contacts():
    mesh, model = merged_surface()
    mesh.landmarks.points[1:, 2] = 10.0
    mask = np.ones(len(mesh.faces), bool)
    parts = contacts.split_contact_components(mesh, mask, model, "a")
    assert len(parts) == 1
    assert np.array_equal(parts[0], np.arange(len(mesh.faces)))


def test_coincident_anatomical_seeds_remain_unresolved():
    mesh, model = merged_surface()
    mesh.landmarks.points[1] = mesh.landmarks.points[0]
    with pytest.raises(ValueError, match="coincident landmark seeds"):
        contacts.split_contact_components(mesh, np.ones(len(mesh.faces), bool), model, "a")


def test_recovery_retries_after_neighbor_provides_anatomical_frame(monkeypatch):
    config = {"joints": [{"a": "a", "b": "b", "patches": []}, {"a": "b", "b": "c", "patches": []}]}
    inference = {"joints": [{"patches": []}, {"patches": []}]}
    model = {"supporting_joints": 5}
    monkeypatch.setattr(contacts, "learn_model", lambda *args: model)
    calls = []

    def recover(meshes, cfg, evidence, index, learned, *args):
        calls.append(index)
        assert learned is model  # Recovered joints never retrain the landmark model.
        if index == 0 and not contacts.complete(cfg["joints"][1]):
            raise ValueError("Neighbor frame is not yet supported")
        return dict(
            cfg["joints"][index], patches=[{"kind": kind} for kind in ("centrum", "facet", "facet")]
        ), {"patches": []}

    monkeypatch.setattr(recovery, "recover_joint", recover)
    recovery.complete_contacts([], config, inference, n.Options())
    assert calls == [0, 1, 0]
    assert all(contacts.complete(d) for d in config["joints"])
    assert len(inference["contact_consensus"]["recovery_passes"]) == 2


def test_unresolvable_recovery_stops_without_repeating_failed_pass(monkeypatch):
    config = {"joints": [{"a": "a", "b": "b", "patches": []}]}
    inference = {"joints": [{"patches": []}]}
    monkeypatch.setattr(contacts, "learn_model", lambda *args: {"supporting_joints": 5})
    calls = []

    def recover(*args):
        calls.append(args[3])
        raise ValueError("Absent anatomical evidence")

    monkeypatch.setattr(recovery, "recover_joint", recover)
    with pytest.raises(ValueError, match="made no progress.*Absent anatomical evidence"):
        recovery.complete_contacts([], config, inference, n.Options())
    assert calls == [0]
    assert not contacts.complete(config["joints"][0])
