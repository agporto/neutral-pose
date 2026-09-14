"""Surface queries and contact-patch descriptors shared by discovery and recovery.

These helpers operate on cleaned :class:`~neutral_pose.core.Mesh` geometry only.
They never assign anatomical roles; labels are opaque identifiers here.
"""

from __future__ import annotations

import math
import re

import numpy as np
import vtk
from scipy.sparse.csgraph import dijkstra

from . import core as n


def canonical_label(label):
    """Normalize exporter prefixes for auditing, without assigning anatomical roles."""
    match = re.fullmatch(r"F(?:_\d+)?-(\d+)", label, re.I)
    return "F-" + match.group(1) if match else label


def surface_queries(mesh, ids, points):
    """Closest points on specified triangles, retaining their original face normals."""
    poly = n.polydata(mesh.vertices, mesh.faces[ids])
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(poly)
    locator.BuildLocator()
    Q = np.empty_like(points, dtype=float)
    normals = np.empty_like(points, dtype=float)
    cell = vtk.reference(0)
    sub = vtk.reference(0)
    dist2 = vtk.reference(0.0)
    for i, point in enumerate(points):
        closest = [0.0, 0.0, 0.0]
        locator.FindClosestPoint(point, closest, cell, sub, dist2)
        Q[i] = closest
        normals[i] = mesh.normals[ids[int(cell)]]
    return Q, normals


def surface_roughness(mesh, ids):
    """Physical residual around a robust quadratic surface; not a voxel/noise measurement."""
    P = mesh.sample(512, ids, seed=219).points
    center = P.mean(axis=0)
    _, _, axes = np.linalg.svd(P - center, full_matrices=False)
    q = (P - center) @ axes.T
    scale = max(float(np.std(q[:, :2], axis=0).max()), 1e-12)
    u, v, z = (q / scale).T
    X = np.c_[np.ones(len(u)), u, v, u * u, u * v, v * v]
    weights = np.ones(len(u))
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(X * np.sqrt(weights[:, None]), z * np.sqrt(weights), rcond=None)
        residual = z - X @ coef
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        weights = np.minimum(1.0, 2.5 * max(mad, 1e-12) / np.maximum(np.abs(residual), 1e-12))
    return float(scale * mad)


def boundary_spec(mesh, ids):
    """Serialisable description of a face set: faces, anchor, geodesic radius and mean normal."""
    center = np.average(mesh.centers[ids], axis=0, weights=mesh.areas[ids])
    anchor, _ = surface_queries(mesh, ids, center[None])
    seed = int(ids[np.argmin(np.linalg.norm(mesh.centers[ids] - anchor[0], axis=1))])
    distance = dijkstra(mesh.adjacency(), directed=False, indices=seed)
    normal = n.unit(np.average(mesh.normals[ids], axis=0, weights=mesh.areas[ids]))
    return {
        "faces": ids.tolist(),
        "inferred_anchor": (anchor[0] + mesh.origin).tolist(),
        "automatic_boundary": {
            "seed_face": seed,
            "radius_mm": float(distance[ids].max()),
            "normal": normal.tolist(),
            "normal_cosine": float(max(-0.1, np.quantile(mesh.normals[ids] @ normal, 0.1))),
        },
    }


def landmark_support(mesh, ids):
    """Landmarks lying close to a patch, with distances, for evidence and learning."""
    if mesh.landmarks is None:
        return []
    Q, _ = surface_queries(mesh, ids, mesh.landmarks.points - mesh.origin)
    distances = np.linalg.norm(Q - (mesh.landmarks.points - mesh.origin), axis=1)
    cutoff = 0.12 * math.sqrt(float(mesh.areas[ids].sum()))
    order = np.argsort(distances)
    return [
        {
            "label": mesh.landmarks.labels[i],
            "canonical_label": canonical_label(mesh.landmarks.labels[i]),
            "distance_to_patch_mm": float(distances[i]),
        }
        for i in order
        if distances[i] <= cutoff
    ]
