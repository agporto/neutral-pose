#!/usr/bin/env python3
"""Estimate an osteological neutral articulation from segmented vertebrae.

Standalone replacement for the supplied straightening script. Requires numpy,
scipy and vtk. Coordinates and dimensions are preserved; only proper rigid
transforms are fitted. See README.md for the anatomical input contract.

The fitted energy is an anatomical scoring function, not a physical energy.
An estimated osteological reference is not a measurement of resting posture.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import platform
import re
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import vtk
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

from .parallel import parallel_map
from .version import __version__

LOG = logging.getLogger(__name__)

# Fitting-objective and QC thresholds that are not user options. They are
# dimensionless (fractions of the joint scale or of the sample spacing) unless
# stated. Changing any of them changes the fitted model; they are named here so
# the objective is fully described in one place rather than scattered inline.

#: Tangential slack, in units of the target patch's sample spacing, before the
#: coverage residual penalises a source sample for lying beyond the target patch.
COVERAGE_SLACK_SPACINGS = 2.5
#: Reduced anchor-centering weight applied when patches were discovered
#: automatically rather than declared anatomically (anchors are less trusted).
AUTOMATIC_CENTERING_WEIGHT_FACTOR = 0.15
#: Minimum normal-gap window (fraction of joint scale) counted as "near" when
#: estimating coverage in :meth:`Joint.metrics`.
COVERAGE_NEAR_FRACTION = 0.02
#: A sample counts as covered only within this many target sample spacings
#: tangentially.
COVERAGE_TANGENTIAL_SPACINGS = 3
#: Facing test for coverage: opposing normals must have a dot product below
#: ``COVERAGE_FACING_COSINE + min(COVERAGE_FACING_NOISE_CAP, 4 * noise_fraction)``.
COVERAGE_FACING_COSINE = -0.5
COVERAGE_FACING_NOISE_CAP = 0.3
#: Minimum number of independent area samples used by the final dense
#: penetration check, in addition to every vertex and triangle center.
DENSE_PENETRATION_MIN_SAMPLES = 2048
#: Fixed seed for the independent dense-validation sample so that reports are
#: reproducible and never share points with the optimisation sample.
DENSE_PENETRATION_SEED = 911


def unit(x: np.ndarray) -> np.ndarray:
    """Normalise vectors along the last axis; raise if any has negligible length."""
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("Cannot define an anatomical direction from coincident points.")
    return x / n


def rigid(rotation=None, translation=None):
    """Build a 4x4 homogeneous matrix from an optional rotation and translation."""
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation
    return T


def inverse(T):
    """Exact inverse of a rigid 4x4 transform (transpose, not a general solve)."""
    return rigid(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def transform(T, points):
    """Apply a rigid 4x4 transform to an (N, 3) array of points."""
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def validate_rigid(T):
    """Return ``T`` as float array after checking it is a proper rigid transform (det +1)."""
    T = np.asarray(T, float)
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9)
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-8)
    ):
        raise ValueError("A pose must be a proper rigid transform (determinant +1).")
    return T


def perturb(T, x, scale):
    """Rotation about the moving mesh's local origin; dimensionless translation."""
    return rigid(Rotation.from_rotvec(x[:3]).as_matrix() @ T[:3, :3], T[:3, 3] + scale * np.asarray(x[3:]))


def pose_difference(A, B, scale):
    """Rotation vector and scale-normalised translation taking pose ``B`` to pose ``A``."""
    return np.r_[Rotation.from_matrix(A[:3, :3] @ B[:3, :3].T).as_rotvec(), (A[:3, 3] - B[:3, 3]) / scale]


def robust_residual(x, delta):
    """Square-root Huber embedding, so robustification is independent of N."""
    x = np.asarray(x, float)
    a = np.abs(x)
    return np.sign(x) * np.sqrt(np.where(a <= delta, x * x, 2 * delta * a - delta * delta))


def coordinate_system(value):
    """Normalise ``'RAS'``/``'LPS'`` (or Slicer's ``0``/``1``) to the canonical string."""
    s = str(value).strip().upper()
    if s in ("0", "RAS"):
        return "RAS"
    if s in ("1", "LPS"):
        return "LPS"
    raise ValueError(f"Unknown coordinate system {value!r}; specify RAS or LPS.")


def convert_coordinates(points, source, target):
    """Convert points between RAS and LPS (flip x and y); identity if systems match."""
    P = np.asarray(points, float).copy()
    if coordinate_system(source) != coordinate_system(target):
        P[..., :2] *= -1
    return P


@dataclass
class Landmarks:
    """Landmark points with their original labels and declared coordinate system."""

    points: np.ndarray
    labels: list[str]
    coordinate_system: str


def read_landmarks(path, mesh_coordinate_system):
    """Read Slicer FCSV (including numeric 0/1 headers) or Markups JSON."""
    path = Path(path)
    if path.name.endswith(".mrk.json"):
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
        marks = doc.get("markups", [])
        if len(marks) != 1 or marks[0].get("type") != "Fiducial":
            raise ValueError(f"{path}: expected one Fiducial markup.")
        mark = marks[0]
        if mark.get("coordinateUnits", "mm") != "mm":
            raise ValueError(f"{path}: landmark coordinates must be in millimeters.")
        coord = coordinate_system(mark.get("coordinateSystem", "LPS"))
        cp = mark.get("controlPoints", [])
        if any(p.get("positionStatus", "defined") != "defined" for p in cp):
            raise ValueError(f"{path}: contains undefined landmarks.")
        labels = [p.get("label", f"P{i + 1}") for i, p in enumerate(cp)]
        points = [p["position"] for p in cp]
    else:
        coord, columns, rows = "RAS", None, []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                if "CoordinateSystem" in line and "=" in line:
                    coord = coordinate_system(line.split("=", 1)[1])
                if re.search(r"\bcolumns\s*=", line, re.I):
                    columns = [s.strip().lower() for s in next(csv.reader([line.split("=", 1)[1]]))]
                continue
            rows.append(next(csv.reader([line], skipinitialspace=True)))
        if columns is None:
            raise ValueError(f"{path}: FCSV must declare its columns; refusing to guess coordinates.")
        indices = [columns.index(c) for c in ("x", "y", "z")]
        li = columns.index("label") if "label" in columns else None
        points = [[float(row[i]) for i in indices] for row in rows]
        labels = [row[li] if li is not None else f"P{i + 1}" for i, row in enumerate(rows)]
    P = np.asarray(points, float).reshape(-1, 3)
    if not len(P) or not np.isfinite(P).all() or len(set(labels)) != len(labels):
        raise ValueError(f"{path}: empty, nonfinite, or duplicate-labeled landmarks.")
    return Landmarks(convert_coordinates(P, coord, mesh_coordinate_system), labels, coord)


def write_landmarks(path, points, labels, coord):
    """Write landmarks as Slicer ``.fcsv`` or ``.mrk.json`` depending on the suffix."""
    cp = [
        {
            "id": str(i + 1),
            "label": label,
            "description": "",
            "associatedNodeID": "",
            "position": list(map(float, p)),
            "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "defined",
        }
        for i, (p, label) in enumerate(zip(points, labels))
    ]
    doc = {
        "@schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json",
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": coordinate_system(coord),
                "coordinateUnits": "mm",
                "locked": False,
                "labelFormat": "%N-%d",
                "controlPoints": cp,
                "display": {"visibility": True},
            }
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")


def polydata(vertices, faces, colors=None):
    """Build a vtkPolyData from vertex and triangle arrays (optionally with RGB colours)."""
    p = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    p.SetPoints(pts)
    cells = vtk.vtkCellArray()
    faces = np.asarray(faces, dtype=np.int64)
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.ravel()), deep=True),
    )
    p.SetPolys(cells)
    if colors is not None:
        arr = numpy_to_vtk(np.asarray(colors, np.uint8), deep=True)
        arr.SetName("RGB")
        p.GetPointData().SetScalars(arr)
    return p


def oriented_poly(poly):
    """Return a copy with consistently oriented, outward-pointing triangle normals where determinable."""
    norm = vtk.vtkPolyDataNormals()
    norm.SetInputData(poly)
    norm.SplittingOff()
    norm.ConsistencyOn()
    norm.AutoOrientNormalsOn()
    norm.ComputeCellNormalsOn()
    norm.ComputePointNormalsOn()
    norm.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(norm.GetOutput())
    return result


@dataclass
class SurfaceSample:
    """Area-weighted surface sample: points, their face normals, face ids and mean spacing."""

    points: np.ndarray
    normals: np.ndarray
    faces: np.ndarray


def faces_of(poly):
    """Triangle connectivity of a vtkPolyData whose polys are all triangles."""
    cells = poly.GetPolys()
    conn = vtk_to_numpy(cells.GetConnectivityArray()).astype(np.int64)
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    if len(offsets) > 1 and not np.all(np.diff(offsets) == 3):
        raise ValueError("Mesh contains non-triangular polygons after triangulation.")
    return conn.reshape(-1, 3)


class Mesh:
    """Immutable fitting geometry centered locally, with loaded vertex indices.

    `preflight_triangles` caps the nonadjacent self-intersection scan (it is
    skipped and reported, not silently passed, above the cap).
    """

    def __init__(
        self,
        name,
        vertices,
        faces,
        colors=None,
        landmarks=None,
        source=None,
        preflight_triangles=300000,
        cleaning=None,
        reject_self_intersections=True,
    ):
        self.name, self.source, self.landmarks = name, source, landmarks
        self.cleaning = dict(cleaning or {})
        V, F = np.asarray(vertices, float), np.asarray(faces, np.int64)
        if (
            V.ndim != 2
            or V.shape[1] != 3
            or F.ndim != 2
            or F.shape[1] != 3
            or len(F) < 1
            or not np.isfinite(V).all()
            or F.min() < 0
            or F.max() >= len(V)
        ):
            raise ValueError(f"{name}: invalid or empty triangular mesh.")
        self.origin = V.mean(axis=0)
        self.vertices = V - self.origin
        p = oriented_poly(polydata(self.vertices, F))
        self.faces = faces_of(p).copy()
        tri = self.vertices[self.faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        if np.any(twice_area <= np.finfo(float).eps * max(np.ptp(V, axis=0).max() ** 2, 1e-30)):
            raise ValueError(
                f"{name}: contains zero-area triangles; load through Mesh.read (which cleans them) "
                "or repair the mesh."
            )
        self.areas = twice_area / 2
        self.normals = cross / twice_area[:, None]
        self.centers = tri.mean(axis=1)
        self.poly = p
        self.colors = (
            np.full((len(V), 3), 204, np.uint8) if colors is None else np.asarray(colors, np.uint8).copy()
        )
        if self.colors.shape != (len(V), 3):
            raise ValueError(f"{name}: invalid vertex colors.")
        edges = np.sort(
            np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]), axis=1
        )
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        self.boundary_edges = int(np.sum(counts == 1))
        self.nonmanifold_edges = int(np.sum(counts > 2))
        self.closed = bool(np.all(counts == 2))
        self.topologically_closed = self.closed
        # Sampling resolution, not an independent estimate of surface error.
        # Use every unique edge so cyclic triangle indexing cannot change it.
        self.resolution = float(
            np.median(
                np.linalg.norm(self.vertices[unique_edges[:, 1]] - self.vertices[unique_edges[:, 0]], axis=1)
            )
        )
        graph = coo_matrix(
            (np.ones(len(unique_edges)), (unique_edges[:, 0], unique_edges[:, 1])), shape=(len(V), len(V))
        ).tocsr()
        _, vertex_regions = connected_components(graph, directed=False)
        self.connected_components = len(np.unique(vertex_regions[self.faces[:, 0]]))
        self.sdf = vtk.vtkImplicitPolyDataDistance()
        self.sdf.SetInput(self.poly)
        self._graph = None
        self._smooth_normals = None
        sample = self.sample(4096, seed=11).points
        centered = sample - sample.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        ext = np.diff(np.quantile(centered @ vt.T, [0.05, 0.95], axis=0), axis=0)[0]
        self.scale = float(np.median(ext))
        if self.scale <= 1e-12:
            raise ValueError(f"{name}: mesh has negligible extent.")
        self.self_intersection_checked = len(self.faces) <= preflight_triangles
        self.self_intersection = self.check_self_intersections() if self.self_intersection_checked else None
        if self.self_intersection is not None:
            if reject_self_intersections:
                raise ValueError(
                    f"{name}: nonadjacent triangles {self.self_intersection} intersect; "
                    "repair the input mesh."
                )
            # Automatic preview can retain imperfect geometry, but its signed
            # distances must not act as verified inside/outside constraints.
            self.closed = False
            LOG.warning(
                "%s: input self-intersection detected; retaining geometry with unverified signs.", name
            )

    def __getstate__(self):
        # VTK objects are not picklable. They are pure functions of the stored
        # geometry and are rebuilt in __setstate__; polydata() reproduces the
        # oriented poly's points, connectivity and bounds exactly, so signed
        # distances, locators and collision checks are unchanged.
        state = self.__dict__.copy()
        del state["poly"], state["sdf"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.poly = polydata(self.vertices, self.faces)
        self.sdf = vtk.vtkImplicitPolyDataDistance()
        self.sdf.SetInput(self.poly)

    @classmethod
    def read(
        cls,
        path,
        landmarks=None,
        keep_largest_component=False,
        preflight_triangles=300000,
        reject_self_intersections=True,
    ):
        path = Path(path)
        reader_cls = {
            ".ply": vtk.vtkPLYReader,
            ".vtp": vtk.vtkXMLPolyDataReader,
            ".stl": vtk.vtkSTLReader,
            ".obj": vtk.vtkOBJReader,
        }.get(path.suffix.lower())
        if reader_cls is None:
            raise ValueError(f"Unsupported mesh format: {path.suffix}")
        r = reader_cls()
        r.SetFileName(str(path))
        r.Update()
        return cls.from_polydata(
            path.stem,
            r.GetOutput(),
            landmarks,
            str(path),
            keep_largest_component,
            preflight_triangles,
            reject_self_intersections,
        )

    @classmethod
    def from_polydata(
        cls,
        name,
        poly,
        landmarks=None,
        source=None,
        keep_largest_component=False,
        preflight_triangles=300000,
        reject_self_intersections=True,
    ):
        """Clean a segmentation export: merge duplicate points (STL has none
        shared), drop degenerate cells, triangulate, and optionally keep only the
        shell with greatest surface area. Components are preserved by default;
        surface area is a geometric selection rule, not anatomical identification.
        Everything removed is counted in `cleaning`."""
        stats = {"input_points": int(poly.GetNumberOfPoints()), "input_cells": int(poly.GetNumberOfCells())}
        clean = vtk.vtkCleanPolyData()
        # Marching-cubes exporters emit points coincident to float32 rounding;
        # merge within 1e-7 of the bounding-box diagonal so slivers collapse
        # into degenerate cells, which are then removed.
        clean.SetInputData(poly)
        clean.PointMergingOn()
        clean.ToleranceIsAbsoluteOff()
        clean.SetTolerance(1e-7)
        clean.ConvertPolysToLinesOn()
        clean.ConvertLinesToPointsOn()
        clean.ConvertStripsToPolysOn()
        clean.Update()
        tr = vtk.vtkTriangleFilter()
        tr.SetInputConnection(clean.GetOutputPort())
        tr.PassLinesOff()
        tr.PassVertsOff()
        tr.Update()
        p = tr.GetOutput()
        if p.GetNumberOfPoints() == 0 or p.GetNumberOfPolys() == 0:
            raise ValueError(f"{name}: no triangles after cleaning.")
        stats["merged_points"] = stats["input_points"] - int(p.GetNumberOfPoints())
        stats["removed_degenerate_cells"] = max(stats["input_cells"] - int(p.GetNumberOfPolys()), 0)
        if stats["merged_points"] or stats["removed_degenerate_cells"]:
            LOG.info(
                "%s: merged %d duplicate points, removed %d degenerate cells.",
                name,
                stats["merged_points"],
                stats["removed_degenerate_cells"],
            )
        colors = p.GetPointData().GetArray("RGB")
        if colors is None:
            colors = p.GetPointData().GetArray("RGBA")
        if colors is not None and colors.GetNumberOfComponents() >= 3:
            colors = vtk_to_numpy(colors)[:, :3]
            if np.issubdtype(colors.dtype, np.floating):
                colors = np.clip(np.rint(colors * 255 if colors.max() <= 1 else colors), 0, 255).astype(
                    np.uint8
                )
        else:
            colors = None
        V, F = vtk_to_numpy(p.GetPoints().GetData()).astype(float), faces_of(p)
        # Connectivity is by shared vertices, as in VTK. Select by physical
        # area rather than cell count, which changes under harmless subdivision.
        edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(V), len(V))).tocsr()
        _, vertex_regions = connected_components(graph, directed=False)
        _, regions = np.unique(vertex_regions[F[:, 0]], return_inverse=True)
        tri = V[F]
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        region_areas = np.bincount(regions, weights=areas)
        region_counts = np.bincount(regions)
        selected = int(np.argmax(region_areas)) if keep_largest_component else None
        keep = np.ones(len(F), dtype=bool) if selected is None else regions == selected
        stats.update(
            {
                "connected_components": len(region_areas),
                "component_selection": "largest_surface_area" if selected is not None else "preserve_all",
                "components_removed": len(region_areas) - 1 if selected is not None else 0,
                "component_triangles_removed": int(np.sum(~keep)),
                "component_area_removed_mm2": float(areas[~keep].sum()),
                "component_area_removed_fraction": float(areas[~keep].sum() / areas.sum())
                if areas.sum() > 0
                else 0.0,
                "components": [
                    {
                        "id": i,
                        "triangles": int(count),
                        "surface_area_mm2": float(area),
                        "kept": selected is None or i == selected,
                    }
                    for i, (area, count) in enumerate(zip(region_areas, region_counts))
                ],
            }
        )
        # Retain legacy report keys for clients reading 1.1 reports.
        stats["small_components_removed"] = stats["components_removed"]
        stats["small_component_triangles_removed"] = stats["component_triangles_removed"]
        F = F[keep]
        if stats["components_removed"]:
            LOG.warning(
                "%s: kept the shell with greatest surface area of %d components "
                "(%d triangles, %.6g mm^2 discarded); inspect the selected anatomy.",
                name,
                len(region_areas),
                stats["component_triangles_removed"],
                stats["component_area_removed_mm2"],
            )
        # Remove discarded/unreferenced points before deriving sliver thresholds.
        used, F = np.unique(F, return_inverse=True)
        F = F.reshape(-1, 3)
        V = V[used]
        if colors is not None:
            colors = colors[used]
        # Collapse sliver triangles (area below 1e-10 of the squared extent) by
        # merging their two closest vertices. Closedness is checked afterward;
        # collapse is a cleanup operation, not a guarantee of valid topology.
        collapsed = 0
        for _ in range(4):
            tri = V[F]
            area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            slivers = np.where(area2 <= 1e-10 * np.ptp(V, axis=0).max() ** 2)[0]
            if not len(slivers):
                break
            remap = np.arange(len(V))
            for f in F[slivers]:
                pairs = [(0, 1), (1, 2), (2, 0)]
                i, j = min(pairs, key=lambda ij: np.linalg.norm(V[f[ij[0]]] - V[f[ij[1]]]))
                a_, b_ = sorted((int(f[i]), int(f[j])))
                remap[remap == b_] = a_
            F = remap[F]
            distinct = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
            collapsed += int(np.sum(~distinct))
            F = F[distinct]
        stats["collapsed_sliver_triangles"] = collapsed
        used, F = np.unique(F, return_inverse=True)
        F = F.reshape(-1, 3)
        V = V[used]
        if colors is not None:
            colors = colors[used]
        if collapsed:
            LOG.info("%s: collapsed %d sliver triangles.", name, collapsed)
        mesh = cls(
            name,
            V,
            F,
            colors,
            landmarks,
            source,
            preflight_triangles,
            stats,
            reject_self_intersections=reject_self_intersections,
        )
        mesh.cleaning["output_connected_components"] = mesh.connected_components
        return mesh

    def smooth_normals(self):
        """One-ring averaged face normals: marching-cubes normals are too noisy
        for a cone test or a normal-compatibility term."""
        if self._smooth_normals is None:
            A = (self.adjacency() > 0).astype(float)
            N = self.normals * self.areas[:, None] + A @ (self.normals * self.areas[:, None])
            self._smooth_normals = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)
        return self._smooth_normals

    def sample(self, n, face_ids=None, seed=0):
        ids = np.arange(len(self.faces)) if face_ids is None else np.asarray(face_ids, int)
        rng = np.random.default_rng(seed)
        # Stratified area sampling avoids dependence on vertex tessellation density.
        cdf = np.cumsum(self.areas[ids])
        cdf /= cdf[-1]
        idx = ids[np.searchsorted(cdf, (np.arange(n) + rng.random(n)) / n).clip(max=len(ids) - 1)]
        uv = rng.random((n, 2))
        su = np.sqrt(uv[:, 0])
        bary = np.c_[1 - su, su * (1 - uv[:, 1]), su * uv[:, 1]]
        points = np.einsum("ni,nij->nj", bary, self.vertices[self.faces[idx]])
        return SurfaceSample(points, self.normals[idx], idx)

    def signed_distance(self, points):
        points = np.ascontiguousarray(points, dtype=np.float64)
        out = vtk.vtkDoubleArray()
        out.SetNumberOfTuples(len(points))
        self.sdf.FunctionValue(numpy_to_vtk(points, deep=False), out)
        return vtk_to_numpy(out).copy()

    def penetration(self, points, scale=1.0, tolerance=0.0):
        """Exact nonnegative penetration penalty, with a conservative broad phase.

        For a checked closed surface, a point outside its enclosing box cannot
        be inside the mesh: max(-signed_distance / scale - tolerance, 0) is
        zero there. Keep an expanded boundary band in the original VTK query.
        Open/unverified geometry and unsupported scale/tolerance values use
        the original full query. This does not replace signed_distance, whose
        positive exterior distances are still needed by anatomy and spacing.
        """
        P = np.ascontiguousarray(points, dtype=np.float64)
        if (
            not self.closed
            or not self.self_intersection_checked
            or not np.isfinite([scale, tolerance]).all()
            or scale <= 0
            or tolerance < 0
        ):
            return np.maximum(-self.signed_distance(P) / scale - tolerance, 0)
        bounds = np.asarray(self.poly.GetBounds()).reshape(3, 2)
        padding = max(
            self.sdf.GetTolerance(), 64 * np.finfo(float).eps * max(1.0, float(np.abs(bounds).max()))
        )
        active = np.all(P >= bounds[:, 0] - padding, axis=1) & np.all(P <= bounds[:, 1] + padding, axis=1)
        active |= ~np.isfinite(P).all(axis=1)
        result = np.zeros(len(P))
        if np.any(active):
            result[active] = np.maximum(-self.signed_distance(P[active]) / scale - tolerance, 0)
        return result

    def adjacency(self):
        if self._graph is not None:
            return self._graph
        edges, owners = [], []
        for pair in [(0, 1), (1, 2), (2, 0)]:
            edges.append(np.sort(self.faces[:, pair], axis=1))
            owners.append(np.arange(len(self.faces)))
        E, O = np.concatenate(edges), np.concatenate(owners)
        order = np.lexsort((E[:, 1], E[:, 0]))
        E, O = E[order], O[order]
        same = np.where(np.all(E[1:] == E[:-1], axis=1))[0]
        a, b = O[same], O[same + 1]
        w = np.maximum(np.linalg.norm(self.centers[a] - self.centers[b], axis=1), self.scale * 1e-12)
        self._graph = coo_matrix(
            (np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=(len(self.faces), len(self.faces))
        ).tocsr()
        return self._graph

    def label(self, label):
        if self.landmarks is None or label not in self.landmarks.labels:
            raise ValueError(f"{self.name}: missing landmark {label!r}.")
        return self.landmarks.points[self.landmarks.labels.index(label)] - self.origin

    def check_self_intersections(self):
        """Detect intersections of triangles without a shared topological vertex.

        Bounding-box centers and half-diagonals define enclosing spheres.
        Candidate batches bound memory even with very uneven triangle sizes;
        only surviving pairs reach the exact VTK triangle test.
        """
        tri = self.vertices[self.faces]
        lo, hi = tri.min(axis=1), tri.max(axis=1)
        centers = lo + (hi - lo) / 2
        half = np.linalg.norm(hi - lo, axis=1) / 2
        pad = 16 * np.finfo(float).eps * max(float(np.abs(tri).max()), np.finfo(float).tiny)
        tree = cKDTree(centers)
        radii = half + half.max() + 2 * pad
        counts = tree.query_ball_point(centers, radii, return_length=True)
        cumulative = np.cumsum(counts, dtype=np.int64)
        start = 0
        while start < len(tri):
            before = cumulative[start - 1] if start else 0
            end = max(start + 1, int(np.searchsorted(cumulative, before + 250000, side="right")))
            neighbours = tree.query_ball_point(centers[start:end], radii[start:end], return_sorted=True)
            I = np.repeat(np.arange(start, end), counts[start:end])
            J = np.concatenate(neighbours)
            keep = J > I
            I, J = I[keep], J[keep]
            keep = np.all(lo[I] <= hi[J] + pad, axis=1) & np.all(lo[J] <= hi[I] + pad, axis=1)
            I, J = I[keep], J[keep]
            shared = (self.faces[I][:, :, None] == self.faces[J][:, None, :]).any(axis=(1, 2))
            I, J = I[~shared], J[~shared]
            for i, j in zip(I.tolist(), J.tolist()):
                if vtk.vtkTriangle.TrianglesIntersect(*tri[i], *tri[j]):
                    return [int(i), int(j)]
            start = end
        return None


def write_precise_ply(path, vertices, faces, colors=None):
    """Write doubles without vtkPLYWriter's float32 coordinate conversion."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or len(faces) == 0
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or faces.max() > np.iinfo(np.int32).max
    ):
        raise ValueError("Invalid triangular mesh for PLY export")
    colors = np.full(vertices.shape, 204, np.uint8) if colors is None else np.asarray(colors, np.uint8)
    if colors.shape != vertices.shape:
        raise ValueError("Invalid vertex colors for PLY export")
    points = np.empty(
        len(vertices),
        dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for i, key in enumerate(("x", "y", "z")):
        points[key] = vertices[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        points[key] = colors[:, i]
    cells = np.empty(len(faces), dtype=[("count", "u1"), ("vertices", "<i4", (3,))])
    cells["count"] = 3
    cells["vertices"] = faces
    header = (
        f"ply\nformat binary_little_endian 1.0\ncomment Unit: mm\n"
        f"element vertex {len(vertices)}\nproperty double x\nproperty double y\nproperty double z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with Path(path).open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(points.tobytes())
        handle.write(cells.tobytes())


def write_vtp(path, poly):
    """Write vtkPolyData as uncompressed binary VTP so double coordinates survive unchanged."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(poly)
    writer.SetFileName(str(path))
    # Uncompressed UInt64 avoids a VTK compressed-offset boundary failure at
    # particular cell counts. Coordinates and connectivity are unchanged.
    writer.SetHeaderTypeToUInt64()
    writer.SetCompressorTypeToNone()
    if writer.Write() != 1:
        raise OSError(f"Failed to write {path}")


def write_mesh(path, mesh, T, colors=None):
    """Write ``mesh`` posed by ``T`` as double-precision PLY (and VTP alongside)."""
    validate_rigid(T)
    vertices = transform(T, mesh.vertices)
    colors = mesh.colors if colors is None else colors
    if Path(path).suffix.lower() == ".vtp":
        write_vtp(path, oriented_poly(polydata(vertices, mesh.faces, colors)))
    else:
        write_precise_ply(path, vertices, mesh.faces, colors)


@dataclass
class Options:
    """Fitting and QC settings. Fractions are relative to the joint's anatomical scale.

    Defaults are the validated release settings; call :meth:`validate` after
    constructing from user input. See README for the meaning of each field.
    """

    seed: int = 42
    samples: int = 160
    target_samples: int = 1600
    collision_samples: int = 384
    starts: int = 5
    max_nfev: int = 100
    global_max_nfev: int = 60
    rotation_bound_deg: float = 45.0
    translation_bound_fraction: float = 0.8
    gap_fraction: float = 0.01
    gap_tolerance_fraction: float = 0.003
    penetration_tolerance_fraction: float = 0.001
    huber_fraction: float = 0.025
    surface_weight: float = 1.0
    normal_weight: float = 0.04
    center_weight: float = 0.6
    penetration_weight: float = 50.0
    smoothness_weight: float = 0.01
    trusted_pose_weight: float = 0.0
    sensitivity: bool = True
    clearance_factors: tuple = (0.5, 1.5)
    patch_radius_factors: tuple = (0.85, 1.15)
    centering_factors: tuple = (0.25,)
    ambiguity_angle_deg: float = 3.0
    ambiguity_translation_fraction: float = 0.03
    acceptable_surface_error_fraction: float = 0.025
    minimum_coverage: float = 0.6
    # An explicit physical error allowance, including zero, takes precedence.
    # None leaves the configured fractions unchanged unless the user opts into
    # an edge-length heuristic; that heuristic always requires review.
    noise_floor_mm: float | None = None
    auto_noise_floor: bool = False
    noise_edge_factor: float = 0.5
    # Centering acts on anatomical anchors (landmark seeds), with a reduced pull
    # for landmark placement error, so patch-growth asymmetry cannot rotate the bone.
    centering_tolerance_fraction: float = 0.03
    centering_inner_weight: float = 0.4
    preflight_triangles: int = 300000
    keep_largest_component: bool = False

    def validate(self):
        if self.noise_floor_mm is not None and (
            isinstance(self.noise_floor_mm, bool)
            or not isinstance(self.noise_floor_mm, (float, int))
            or not np.isfinite(self.noise_floor_mm)
            or self.noise_floor_mm < 0
        ):
            raise ValueError("noise_floor_mm must be null or a finite nonnegative number in mm.")
        for key in ("sensitivity", "auto_noise_floor", "keep_largest_component"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a JSON boolean.")
        for key in (
            "samples",
            "target_samples",
            "collision_samples",
            "starts",
            "max_nfev",
            "global_max_nfev",
            "preflight_triangles",
        ):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer.")
        for key, value in asdict(self).items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and nonnegative.")
        if self.samples < 16 or self.target_samples < self.samples:
            raise ValueError("Use at least 16 samples and target_samples >= samples.")
        if self.rotation_bound_deg <= 0 or self.translation_bound_fraction <= 0:
            raise ValueError("Search bounds must be positive.")
        if self.huber_fraction <= 0 or not 0 <= self.minimum_coverage <= 1:
            raise ValueError("Invalid Huber threshold or minimum coverage.")
        if not 0 <= self.centering_inner_weight <= 1:
            raise ValueError("centering_inner_weight must lie in [0, 1].")
        for key in ("clearance_factors", "patch_radius_factors", "centering_factors"):
            if any(not np.isfinite(x) or x <= 0 for x in getattr(self, key)):
                raise ValueError(f"{key} must contain positive finite factors.")
        return self


@dataclass
class Patch:
    """A contact patch: triangle ids on a mesh plus its spec, kind and derived anchor/normal."""

    mesh: Mesh
    ids: np.ndarray
    spec: dict
    kind: str
    center: np.ndarray
    normal: np.ndarray
    # The anatomical anchor used for centering and seeding: the landmark/point
    # seed projected onto the surface, or the area centroid for explicit face lists.
    anchor: np.ndarray = None
    anchor_source: str = "centroid"
    sphere_center: np.ndarray | None = None
    sphere_radius: float | None = None
    sphere_relative_error: float | None = None
    query: SurfaceSample | None = None
    target: SurfaceSample | None = None
    tree: cKDTree | None = None
    spacing: float = 0.0

    def prepare(self, options, seed):
        self.query = self.mesh.sample(options.samples, self.ids, seed)
        self.target = self.mesh.sample(options.target_samples, self.ids, seed + 1009)
        self.tree = cKDTree(self.target.points)
        self.spacing = math.sqrt(self.mesh.areas[self.ids].sum() / options.target_samples)
        return self


def fit_sphere(mesh, ids, noise_fraction=0.0):
    """Accept a sphere only with adequate curvature and a well-conditioned fit.

    The residual threshold is widened by the surface noise level so a noisy
    but genuinely spherical condyle/cotyle still contributes its center."""
    P = mesh.sample(512, ids, seed=7).points
    center = P.mean(axis=0)
    local = (P - center) / mesh.scale
    A = np.c_[2 * local, np.ones(len(P))]
    if np.linalg.cond(A) > 100:
        return None, None, None
    coef, *_ = np.linalg.lstsq(A, (local * local).sum(axis=1), rcond=None)
    C = center + mesh.scale * coef[:3]
    rad = np.linalg.norm(P - C, axis=1)
    R = float(np.median(rad))
    err = float(np.sqrt(np.mean((rad - R) ** 2)) / mesh.scale)
    if R < 0.06 * mesh.scale or R > 2 * mesh.scale or err > max(0.008, 1.5 * noise_fraction):
        return None, None, None
    # A shallow cap provides a poorly identified center even when residuals are small.
    spread = np.linalg.svd(unit(P - C), compute_uv=False)
    if spread[-1] / spread[0] < 0.1:
        return None, None, None
    return C, R, err


def make_patch(mesh, spec, kind, coordinate, radius_factor=1.0, noise_fraction=0.0):
    """Build a :class:`Patch` from a spec: explicit faces, or a geodesic disc grown from a landmark/anchor."""
    spec = copy.deepcopy(spec)
    anchor, anchor_source = None, "centroid"
    if "faces" in spec:
        ids = np.asarray(spec["faces"], dtype=int)
        if ids.ndim != 1 or not len(ids) or ids.min() < 0 or ids.max() >= len(mesh.faces):
            raise ValueError(f"{mesh.name}: invalid patch triangle indices.")
        ids = np.unique(ids)
        if "automatic_boundary" in spec and radius_factor != 1.0:
            boundary = spec["automatic_boundary"]
            seed = int(boundary["seed_face"])
            radius = float(boundary["radius_mm"]) * radius_factor
            distance = dijkstra(mesh.adjacency(), directed=False, indices=seed, limit=radius)
            axis = unit(np.asarray(boundary["normal"], float))
            if radius_factor < 1.0:
                ids = ids[distance[ids] <= radius]
            else:
                allowed = (distance <= radius) & (mesh.normals @ axis >= float(boundary["normal_cosine"]))
                allowed[spec["faces"]] = True
                candidates = np.where(allowed)[0]
                _, labels = connected_components(mesh.adjacency()[candidates][:, candidates], directed=False)
                ids = candidates[labels == labels[np.where(candidates == seed)[0][0]]]
    else:
        if "label" in spec:
            point = mesh.label(spec["label"])
        elif "point" in spec:
            point = (
                convert_coordinates(
                    np.asarray(spec["point"], float), spec.get("coordinate_system", coordinate), coordinate
                )
                - mesh.origin
            )
        else:
            raise ValueError("A patch requires label, point, or faces.")
        if np.asarray(point).shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Patch seed must contain three finite coordinates.")
        # Use the closest triangle, not a vertex, for a landmark-to-surface query.
        loc = vtk.vtkStaticCellLocator()
        loc.SetDataSet(mesh.poly)
        loc.BuildLocator()
        cp = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(0)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        loc.FindClosestPoint(point, cp, cell_id, sub_id, dist2)
        seed = int(cell_id)
        if math.sqrt(float(dist2)) > float(spec.get("max_seed_distance_fraction", 0.12)) * mesh.scale:
            raise ValueError(f"{mesh.name}: landmark lies too far from the articular surface.")
        anchor, anchor_source = np.asarray(cp, float), "seed"
        radius = float(spec.get("radius_fraction", 0.22)) * mesh.scale * radius_factor
        angle = float(spec.get("normal_angle_deg", 80.0 if kind == "centrum" else 55.0))
        if not np.isfinite(radius) or radius <= 0 or not 0 < angle <= 180:
            raise ValueError("Patch radius must be positive and normal angle in (0, 180].")
        distance = dijkstra(mesh.adjacency(), directed=False, indices=seed, limit=radius)
        # One-ring smoothed normals tame marching-cubes noise; on coarse hand-made
        # meshes a one-ring spans a large fraction of the patch, so use raw normals.
        normals = mesh.smooth_normals() if mesh.resolution < 0.25 * radius else mesh.normals
        # The cone axis is the area-weighted normal of the seed's neighbourhood,
        # not the seed triangle itself, so one noisy triangle cannot shrink the patch.
        core = np.where(distance <= max(0.3 * radius, 2 * mesh.resolution))[0]
        axis = unit(np.average(normals[core], axis=0, weights=mesh.areas[core]))
        allowed = (distance <= radius) & (normals @ axis >= np.cos(np.deg2rad(angle)))
        allowed[seed] = True
        candidates = np.where(allowed)[0]
        _, labels = connected_components(mesh.adjacency()[candidates][:, candidates], directed=False)
        ids = candidates[labels == labels[np.where(candidates == seed)[0][0]]]
    if len(ids) < 3 or mesh.areas[ids].sum() < 1e-6 * mesh.scale**2:
        raise ValueError(f"{mesh.name}: articular patch is too small; adjust the seed or radius.")
    center = np.average(mesh.centers[ids], axis=0, weights=mesh.areas[ids])
    if "inferred_anchor" in spec:
        anchor = np.asarray(spec["inferred_anchor"], float) - mesh.origin
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("Inferred anchor must contain three finite coordinates.")
        anchor_source = "inferred_surface_center"
    normal = np.average(mesh.normals[ids], axis=0, weights=mesh.areas[ids])
    if np.linalg.norm(normal) < 0.1:
        raise ValueError(f"{mesh.name}: patch wraps around the bone; reduce its radius.")
    patch = Patch(
        mesh,
        ids,
        spec,
        kind,
        center,
        unit(normal),
        anchor=center if anchor is None else anchor,
        anchor_source=anchor_source,
    )
    if kind == "centrum":
        patch.sphere_center, patch.sphere_radius, patch.sphere_relative_error = fit_sphere(
            mesh, ids, noise_fraction
        )
    return patch


@dataclass
class PatchPair:
    """Two opposing patches with the assumed clearance, tolerance and weight used in fitting."""

    name: str
    kind: str
    a: Patch
    b: Patch
    gap_fraction: float
    gap_tolerance_fraction: float  # configured slack for cartilage uncertainty (fitting dead zone)
    weight: float = 1.0
    report_tolerance_fraction: float = 0.0  # max(configured, noise floor): used for acceptance, not fitting


def anatomical_roles(config, mesh):
    """Landmark labels for each declared anatomical role that this mesh actually has."""
    roles = dict(config.get("roles", {}))
    roles.update(config.get("mesh_roles", {}).get(mesh.name, {}))
    # Exact semantic labels may be used without a mapping; numbered labels are never guessed.
    if mesh.landmarks is not None:
        for role in (
            "anterior_centrum",
            "posterior_centrum",
            "left_prezygapophysis",
            "right_prezygapophysis",
            "left_postzygapophysis",
            "right_postzygapophysis",
            "zygosphene",
            "zygantrum",
            "dorsal",
        ):
            if role in mesh.landmarks.labels:
                roles.setdefault(role, role)
    return roles


def anatomical_scale(mesh, roles, config):
    """Local length scale for a bone: from declared centrum landmarks when present, else the mesh scale."""
    explicit = config.get("mesh_scales", {}).get(mesh.name)
    if explicit is not None:
        if not np.isfinite(explicit) or explicit <= 0:
            raise ValueError("mesh_scales values must be finite positive lengths.")
        return float(explicit)
    if "anterior_centrum" in roles and "posterior_centrum" in roles:
        length = np.linalg.norm(
            mesh.label(roles["anterior_centrum"]) - mesh.label(roles["posterior_centrum"])
        )
        if length > 1e-6 * mesh.scale:
            return float(length)
    return mesh.scale


def anatomical_frame(mesh, roles):
    """Lateral/dorsal/longitudinal axes from declared bilateral landmarks; ``known`` is False if inferred."""
    if not {"anterior_centrum", "posterior_centrum"} <= roles.keys():
        return np.eye(3), False
    anterior = mesh.label(roles["anterior_centrum"])
    posterior = mesh.label(roles["posterior_centrum"])
    z = unit(posterior - anterior)
    if "left_prezygapophysis" in roles and "right_prezygapophysis" in roles:
        x = mesh.label(roles["right_prezygapophysis"]) - mesh.label(roles["left_prezygapophysis"])
        x = unit(x - z * np.dot(x, z))
        y = unit(np.cross(z, x))
    elif "dorsal" in roles:
        y = mesh.label(roles["dorsal"]) - (anterior + posterior) / 2
        y = unit(y - z * np.dot(y, z))
        x = unit(np.cross(y, z))
    else:
        return np.eye(3), False
    F = np.column_stack([x, y, z])
    validate_rigid(rigid(F))
    return F, True


class NeutralPlane:
    """Rigid motions with coincident anatomical midsagittal planes.

    Frames have columns (lateral, dorsal, posterior). Only rotation about the
    lateral axis and the two in-plane translations remain free. No vertex is
    projected, scaled, reflected or deformed by this constraint.
    """

    def __init__(self, a, b, frames):
        self.fa, self.ca = self.read_frame(a, frames)
        self.fb, self.cb = self.read_frame(b, frames)

    @staticmethod
    def read_frame(mesh, frames):
        if mesh.name not in frames:
            raise ValueError(f"{mesh.name}: missing anatomical neutral-plane frame.")
        item = frames[mesh.name]
        axes = np.asarray(item["axes"], float)
        center = np.asarray(item["center"], float)
        validate_rigid(rigid(axes))
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Anatomical frame center must be a finite original-coordinate point.")
        return axes, center - mesh.origin

    def encode(self, H, scale):
        G = self.fa.T @ H[:3, :3] @ self.fb
        # Closest rotation about x in SO(3), not an Euler-angle deletion.
        angle = math.atan2(G[2, 1] - G[1, 2], G[1, 1] + G[2, 2])
        offset = self.fa.T @ (transform(H, self.cb) - self.ca)
        return np.r_[angle, offset[1:] / scale]

    def decode(self, parameters, scale):
        angle, y, z = np.asarray(parameters, float)
        R = self.fa @ Rotation.from_rotvec([angle, 0.0, 0.0]).as_matrix() @ self.fb.T
        t = self.ca + self.fa @ np.array([0.0, y * scale, z * scale]) - R @ self.cb
        return validate_rigid(rigid(R, t))

    def project(self, H, scale):
        return self.decode(self.encode(H, scale), scale)

    def diagnostics(self, H):
        G = self.fa.T @ H[:3, :3] @ self.fb
        offset = self.fa.T @ (transform(H, self.cb) - self.ca)
        angle = math.atan2(G[2, 1] - G[1, 2], G[1, 1] + G[2, 2])
        return {
            "lateral_axis_misalignment_deg": float(np.rad2deg(np.arctan2(np.linalg.norm(G[1:, 0]), G[0, 0]))),
            "lateral_center_offset_mm": float(offset[0]),
            "sagittal_rotation_deg": float(np.rad2deg(angle)),
            "free_parameters": ["sagittal_rotation", "dorsal_translation", "posterior_translation"],
        }


def candidate_components(mesh, mask):
    """Connected face groups of a boolean face mask, largest area first."""
    ids = np.where(mask)[0]
    if not len(ids):
        return []
    n, labels = connected_components(mesh.adjacency()[ids][:, ids], directed=False)
    comps = [ids[labels == k] for k in range(n)]
    comps = [x for x in comps if len(x) >= 3 and mesh.areas[x].sum() > 0.002 * mesh.scale**2]
    return sorted(comps, key=lambda x: -mesh.areas[x].sum())[:8]


def automatic_patch_specs(a, b, H, scale, *, component_splitter=None):
    """Initial-pose-dependent candidate discovery; never labels anatomical identity."""
    # Query triangle centers; combine neighborhood size with opposing normals.
    B = transform(H, b.centers)
    BN = b.normals @ H[:3, :3].T
    tree_a, tree_b = cKDTree(a.centers), cKDTree(B)

    def nearest(P, N, tree, QN):
        d, idx = tree.query(P, k=min(8, len(QN)))
        if d.ndim == 1:
            d, idx = d[:, None], idx[:, None]
        dots = np.einsum("ni,nki->nk", N, QN[idx])
        merit = d / scale + 0.2 * np.maximum(dots + 0.3, 0)
        take = np.argmin(merit, axis=1)
        rows = np.arange(len(P))
        return d[rows, take], idx[rows, take], dots[rows, take]

    da, ia, na = nearest(a.centers, a.normals, tree_b, BN)
    db, ib, nb = nearest(B, BN, tree_a, a.normals)
    finite = da[na < -0.3]
    if not len(finite) or np.min(finite) > 0.18 * scale:
        raise ValueError(
            "No nearby opposing surfaces. Supply anatomical landmarks/patches or an approximate articulation."
        )
    threshold = min(0.18 * scale, max(0.045 * scale, float(np.quantile(finite, 0.12)) + 0.025 * scale))
    components = candidate_components if component_splitter is None else component_splitter
    ca = components(a, (da <= threshold) & (na < -0.3))
    cb = components(b, (db <= threshold) & (nb < -0.3))
    pairs, used = [], set()
    for ids in ca:
        best = None
        for j, other in enumerate(cb):
            if j in used:
                continue
            score = np.mean(np.isin(ia[ids], other)) + np.mean(np.isin(ib[other], ids))
            if best is None or score > best[0]:
                best = score, j, other
        if best is not None and best[0] >= 0.4:
            _, j, other = best
            used.add(j)
            pairs.append(
                {
                    "name": f"candidate_{len(pairs) + 1}",
                    "kind": "unclassified",
                    "a": {"faces": ids.tolist()},
                    "b": {"faces": other.tolist()},
                }
            )
        if len(pairs) == 4 and component_splitter is None:
            break
    if not pairs:
        raise ValueError("Could not identify connected opposing patches; anatomical seeds are required.")
    return pairs


class Joint:
    """One adjacent bone pair with its contact patches, tolerances and fitting residual.

    ``a`` is the fixed (proximal) bone and ``b`` the moving one. All residuals
    are expressed in fractions of the joint scale ``sqrt(scale_a * scale_b)``.
    """

    def __init__(self, a, b, config, options, index=0, radius_factor=1.0, *, _config_snapshot=None):
        self.a, self.b, self.index, self.options = a, b, index, options
        self.name = f"{a.name}__{b.name}"
        # fit_column owns an isolated snapshot for internal read-only use.
        # Public access still gets an independent mutable deep copy, just as
        # standalone Joint construction does. Large columns need not copy all
        # other bones' patch lists for every joint and every sensitivity trial.
        self._config = copy.deepcopy(config) if _config_snapshot is None else None
        self._config_snapshot = _config_snapshot
        self.roles_a, self.roles_b = anatomical_roles(config, a), anatomical_roles(config, b)
        sa = anatomical_scale(a, self.roles_a, config)
        sb = anatomical_scale(b, self.roles_b, config)
        self.scale = math.sqrt(sa * sb)
        self.frame, self.frame_known = anatomical_frame(a, self.roles_a)
        frames = config.get("neutral_frames")
        self.neutral_plane = NeutralPlane(a, b, frames) if frames is not None else None
        if self.neutral_plane is not None:
            self.frame, self.frame_known = self.neutral_plane.fa, True
        self.initial = rigid(translation=b.origin - a.origin)
        overrides = [j for j in config.get("joints", []) if j.get("a") == a.name and j.get("b") == b.name]
        if len(overrides) > 1:
            raise ValueError(f"Duplicate joint definitions for {self.name}.")
        override = overrides[0] if overrides else {}
        self.region = override.get("region")
        if "initial_transform" in override:
            # Maps B's original mesh coordinates into A's original coordinate system.
            original = validate_rigid(override["initial_transform"])
            self.initial = rigid(translation=-a.origin) @ original @ rigid(translation=b.origin)
        specs = copy.deepcopy(override.get("patches", []))
        if not specs:
            role_pairs = [
                ("centrum", "centrum", "posterior_centrum", "anterior_centrum"),
                ("left_facet", "facet", "left_postzygapophysis", "left_prezygapophysis"),
                ("right_facet", "facet", "right_postzygapophysis", "right_prezygapophysis"),
                ("accessory", "accessory", "zygantrum", "zygosphene"),
            ]
            for name, kind, ra, rb in role_pairs:
                if ra in self.roles_a and rb in self.roles_b:
                    settings = config.get("patch_defaults", {}).get(kind, {})
                    specs.append(
                        {
                            "name": name,
                            "kind": kind,
                            "a": dict(settings, label=self.roles_a[ra]),
                            "b": dict(settings, label=self.roles_b[rb]),
                        }
                    )
        self.anatomical = bool(specs)
        if not specs:
            if config.get("require_anatomy", False):
                raise ValueError(f"{self.name}: anatomically identified patches are required.")
            specs = automatic_patch_specs(a, b, self.initial, self.scale)
        self.specs = specs
        # Physical settings override the optional tessellation-dependent fallback.
        capped = False
        supplied_noise = override.get("noise_floor_mm", options.noise_floor_mm)
        if supplied_noise is not None:
            noise_mm = float(supplied_noise)
            if not np.isfinite(noise_mm) or noise_mm < 0:
                raise ValueError("Joint noise_floor_mm must be finite and nonnegative.")
            self.noise_source = str(override.get("noise_floor_source", "explicit"))
        elif options.auto_noise_floor:
            edge_based = options.noise_edge_factor * max(a.resolution, b.resolution)
            capped = edge_based > 0.05 * self.scale
            if capped:
                LOG.warning(
                    "%s: median edge length is coarse relative to the bone; capping the automatic "
                    "noise floor at 5%% of local scale. Supply an independently justified noise_floor_mm.",
                    self.name,
                )
                edge_based = 0.05 * self.scale
            noise_mm = float(edge_based)
            self.noise_source = "mesh_edge_heuristic"
        else:
            noise_mm = 0.0
            self.noise_source = "disabled"
        self.noise_fraction = noise_mm / self.scale
        # A modeling allowance for two uncertain surfaces; not a statistical
        # confidence bound or an anatomical justification for interpenetration.
        self.penetration_tolerance = max(options.penetration_tolerance_fraction, 2 * self.noise_fraction)
        self.ambiguity_translation = max(options.ambiguity_translation_fraction, 1.5 * self.noise_fraction)
        self.huber = max(options.huber_fraction, 2 * self.noise_fraction)
        self.acceptable_surface_error = max(
            options.acceptable_surface_error_fraction, 2 * self.noise_fraction
        )
        self.tolerances = {
            "noise_floor_mm": noise_mm,
            "noise_fraction": self.noise_fraction,
            "noise_floor_source": self.noise_source,
            "automatic_noise_floor_capped": capped,
            "penetration_tolerance_mm": self.penetration_tolerance * self.scale,
            "huber_threshold_mm": self.huber * self.scale,
            "acceptable_rms_gap_error_mm": self.acceptable_surface_error * self.scale,
            "centering_tolerance_mm": options.centering_tolerance_fraction * self.scale,
            "ambiguity_translation_mm": self.ambiguity_translation * self.scale,
        }
        self.pairs = []
        for k, spec in enumerate(specs):
            kind = spec.get("kind", "unclassified")
            if kind not in ("centrum", "facet", "accessory", "unclassified"):
                raise ValueError(f"Unknown patch kind {kind!r}.")
            coord = config["mesh_coordinate_system"]
            # Sphere residuals are normalized by each mesh's robust scale,
            # whereas joint fitting residuals use the anatomical joint scale.
            pa = make_patch(a, spec["a"], kind, coord, radius_factor, noise_mm / a.scale).prepare(
                options, options.seed + 31 * k
            )
            pb = make_patch(b, spec["b"], kind, coord, radius_factor, noise_mm / b.scale).prepare(
                options, options.seed + 31 * k + 13
            )
            gap = float(
                spec.get("gap_fraction", config.get("joint_gaps", {}).get(kind, options.gap_fraction))
            )
            tol = float(spec.get("gap_tolerance_fraction", options.gap_tolerance_fraction))
            weight = float(spec.get("weight", 1.0))
            if not np.isfinite([gap, tol, weight]).all() or min(gap, tol) < 0 or weight <= 0:
                raise ValueError("Patch gaps/tolerances must be nonnegative and weights positive.")
            # Noise is averaged out by least squares, so it does not widen the
            # fitting dead zone (that would leave the pose undetermined by
            # ~noise/lever-arm); it only widens what counts as an acceptable fit.
            self.pairs.append(
                PatchPair(
                    spec.get("name", f"patch_{k + 1}"),
                    kind,
                    pa,
                    pb,
                    gap,
                    tol,
                    weight,
                    report_tolerance_fraction=max(tol, self.noise_fraction),
                )
            )
        if len({p.name for p in self.pairs}) != len(self.pairs):
            raise ValueError(f"{self.name}: patch names must be unique.")
        self.collision_a = a.sample(options.collision_samples, seed=options.seed + 17).points
        self.collision_b = b.sample(options.collision_samples, seed=options.seed + 29).points

    @property
    def config(self):
        if self._config is None:
            self._config = copy.deepcopy(self._config_snapshot)
            self._config_snapshot = None
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        self._config_snapshot = None

    def _fitting_config(self):
        return self._config_snapshot if self._config is None else self._config

    def landmark_seed(self):
        if len(self.pairs) < 3:
            return None
        source, target = [], []
        for p in self.pairs:
            if p.kind == "centrum" and p.a.sphere_center is not None and p.b.sphere_center is not None:
                source.append(p.b.sphere_center)
                target.append(p.a.sphere_center)
            else:
                source.append(p.b.anchor)
                target.append(p.a.anchor + p.gap_fraction * self.scale * p.a.normal)
        X, Y = np.asarray(source), np.asarray(target)
        xc, yc = X.mean(axis=0), Y.mean(axis=0)
        U, S, Vt = np.linalg.svd((X - xc).T @ (Y - yc))
        if S[1] < 1e-5 * self.scale**2:
            return None
        D = np.eye(3)
        D[-1, -1] = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ D @ U.T
        return rigid(R, yc - R @ xc)

    def matches(self, source, target, T):
        P = transform(T, source.query.points)
        N = source.query.normals @ T[:3, :3].T
        distance, ids = target.tree.query(P, k=min(8, len(target.target.points)))
        if distance.ndim == 1:
            distance, ids = distance[:, None], ids[:, None]
        dot = np.einsum("ni,nki->nk", N, target.target.normals[ids])
        # Fixed-count correspondences: unmatched or wrongly facing samples cannot
        # disappear from the objective and thereby improve its score.
        choice = np.argmin((distance / self.scale) ** 2 + 0.02 * (1 + dot) ** 2, axis=1)
        ids = ids[np.arange(len(P)), choice]
        Q, QN = target.target.points[ids], target.target.normals[ids]
        D = P - Q
        normal_distance = np.einsum("ni,ni->n", D, QN) / self.scale
        tangent = np.linalg.norm(D - self.scale * normal_distance[:, None] * QN, axis=1) / self.scale
        return normal_distance, tangent, N + QN, np.einsum("ni,ni->n", N, QN)

    def anchor_offset(self, pair, H):
        """Tangential offset (fractions of scale) between paired anchors, or
        between fitted sphere centers for a spherical centrum."""
        if pair.kind == "centrum" and pair.a.sphere_center is not None and pair.b.sphere_center is not None:
            return (transform(H, pair.b.sphere_center) - pair.a.sphere_center) / self.scale
        delta = (transform(H, pair.b.anchor) - pair.a.anchor) / self.scale
        # Centering is tangential; clearance handles the normal direction.
        return delta - np.dot(delta, pair.a.normal) * pair.a.normal

    def residual(self, H, gap_factor=1.0, include_collision=True, center_factor=1.0):
        opts, s, residuals = self.options, self.scale, []
        H_inverse = inverse(H)
        n_pairs = len(self.pairs)
        for pair in self.pairs:
            gap = pair.gap_fraction * gap_factor
            for source, target, T in [(pair.b, pair.a, H), (pair.a, pair.b, H_inverse)]:
                d, tangent, normals, dots = self.matches(source, target, T)
                factor = math.sqrt(pair.weight / (2 * n_pairs * len(d)))
                e = d - gap
                e = np.sign(e) * np.maximum(np.abs(e) - pair.gap_tolerance_fraction, 0)
                residuals.append(factor * math.sqrt(opts.surface_weight) * robust_residual(e, self.huber))
                coverage = np.maximum(tangent - COVERAGE_SLACK_SPACINGS * target.spacing / s, 0)
                residuals.append(
                    factor * math.sqrt(opts.surface_weight) * robust_residual(coverage, self.huber)
                )
                if source.mesh.closed and target.mesh.closed:
                    nr = normals
                else:
                    # Open surfaces have no certified outward side. Their normal
                    # alignment is sign-agnostic and their QC remains provisional.
                    nr = np.sqrt(np.maximum(1 - np.abs(dots), 0))[:, None]
                residuals.append((factor * math.sqrt(opts.normal_weight) * nr).ravel())
            center = self.anchor_offset(pair, H)
            spherical = (
                pair.kind == "centrum"
                and pair.a.sphere_center is not None
                and pair.b.sphere_center is not None
            )
            cw = (
                opts.center_weight
                * center_factor
                * (1.0 if self.anatomical else AUTOMATIC_CENTERING_WEIGHT_FACTOR)
            )
            if not spherical:
                # Anchors are anatomical claims with placement error. Inside the
                # tolerance a reduced pull remains, enough to resolve a
                # direction the surfaces leave free; beyond it the full weight
                # applies. Small non-homology therefore cannot override surface
                # evidence or rotate a bone about a planar facet's normal.
                norm = np.linalg.norm(center)
                outside = (
                    center * max(norm - opts.centering_tolerance_fraction, 0) / norm if norm > 0 else center
                )
                center = np.r_[
                    math.sqrt(opts.centering_inner_weight) * center,
                    math.sqrt(1 - opts.centering_inner_weight) * outside,
                ]
            residuals.append(math.sqrt(cw * pair.weight / n_pairs) * robust_residual(center, self.huber))
        if include_collision:
            factor = math.sqrt(opts.penetration_weight / opts.collision_samples)
            for mesh, points in [
                (self.a, transform(H, self.collision_b)),
                (self.b, transform(H_inverse, self.collision_a)),
            ]:
                depth = (
                    mesh.penetration(points, s, self.penetration_tolerance)
                    if mesh.closed
                    else np.zeros(len(points))
                )
                residuals.append(factor * depth)
        if opts.trusted_pose_weight:
            residuals.append(math.sqrt(opts.trusted_pose_weight) * pose_difference(H, self.initial, s))
        return np.concatenate(residuals)

    def metrics(self, H, gap_factor=1.0, final=False):
        data = []
        for pair in self.pairs:
            sides = []
            for source, target, T in [(pair.b, pair.a, H), (pair.a, pair.b, inverse(H))]:
                d, tangential, _, dots = self.matches(source, target, T)
                error = np.maximum(
                    np.abs(d - pair.gap_fraction * gap_factor) - pair.report_tolerance_fraction, 0
                )
                # Surface coverage is an area-sampling estimate, not polygonal overlap.
                near = np.abs(d - pair.gap_fraction * gap_factor) <= max(
                    COVERAGE_NEAR_FRACTION, 2 * pair.report_tolerance_fraction
                )
                # Noisy face normals: accept a looser facing test as noise grows.
                facing_cut = COVERAGE_FACING_COSINE + min(COVERAGE_FACING_NOISE_CAP, 4 * self.noise_fraction)
                facing = (
                    dots < facing_cut
                    if source.mesh.closed and target.mesh.closed
                    else np.abs(dots) > -facing_cut
                )
                covered = (
                    near & (tangential <= COVERAGE_TANGENTIAL_SPACINGS * target.spacing / self.scale) & facing
                )
                sides.append(
                    {
                        "rms_gap_error_fraction": float(np.sqrt(np.mean(error**2))),
                        "median_clearance": float(np.median(d) * self.scale),
                        "coverage_fraction": float(np.mean(covered)),
                        "median_normal_dot": float(np.median(dots)),
                    }
                )
            offset = self.anchor_offset(pair, H)
            data.append(
                {
                    "name": pair.name,
                    "kind": pair.kind,
                    "assumed_gap": pair.gap_fraction * gap_factor * self.scale,
                    "allowed_gap_tolerance": pair.gap_tolerance_fraction * self.scale,
                    "acceptance_gap_tolerance": pair.report_tolerance_fraction * self.scale,
                    "anchor_source": [pair.a.anchor_source, pair.b.anchor_source],
                    "spherical_centrum": bool(
                        pair.kind == "centrum"
                        and pair.a.sphere_center is not None
                        and pair.b.sphere_center is not None
                    ),
                    "anchor_tangential_offset_mm": float(np.linalg.norm(offset) * self.scale),
                    "a_to_b": sides[1],
                    "b_to_a": sides[0],
                }
            )
        # Independent dense validation includes every vertex and triangle center,
        # plus a separate area sample; optimization used a smaller fixed sample.
        depth = []
        for mesh, other, T in [(self.a, self.b, H), (self.b, self.a, inverse(H))]:
            if not mesh.closed:
                depth.append(None)
            else:
                pts = (
                    np.vstack(
                        [
                            other.vertices,
                            other.centers,
                            other.sample(
                                max(DENSE_PENETRATION_MIN_SAMPLES, self.options.collision_samples),
                                seed=DENSE_PENETRATION_SEED,
                            ).points,
                        ]
                    )
                    if final
                    else other.sample(self.options.collision_samples, seed=DENSE_PENETRATION_SEED).points
                )
                depth.append(float(mesh.penetration(transform(T, pts)).max()))
        result = {
            "patches": data,
            "max_sampled_penetration": max([x for x in depth if x is not None], default=None),
            "signed_distance_reliable": self.a.closed and self.b.closed,
            "intersection_detected": intersects(self.a, self.b, H) if final else None,
        }
        return result


def intersects(a, b, H):
    """Independent triangle/triangle intersection check (includes open surfaces)."""
    collision = vtk.vtkCollisionDetectionFilter()
    collision.SetInputData(0, a.poly)
    collision.SetInputData(1, b.poly)
    A, B = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
    A.Identity()
    B.DeepCopy(np.asarray(H, float).ravel())
    collision.SetMatrix(0, A)
    collision.SetMatrix(1, B)
    collision.SetCollisionModeToFirstContact()
    collision.SetBoxTolerance(0.0)
    collision.SetCellTolerance(0.0)
    collision.SetNumberOfCellsPerNode(2)
    collision.GenerateScalarsOff()
    collision.Update()
    return bool(collision.GetNumberOfContacts())


@dataclass
class Candidate:
    """A fitted joint pose with its score, convergence flag and start index."""

    transform: np.ndarray
    score: float
    converged: bool
    nfev: int
    reason: str
    boundary: bool
    singular_values: list[float]


def optimize_joint(joint, seed, gap_factor=1.0, max_nfev=None, center_factor=1.0, fixed_sagittal=False):
    """Least-squares fit of one joint from ``seed``; sagittal-constrained when a neutral plane exists."""
    o = joint.options
    if joint.neutral_plane is not None:
        base = joint.neutral_plane.encode(seed, joint.scale)
        if fixed_sagittal:
            bounds = np.full(2, o.translation_bound_fraction)
            decode = lambda x: joint.neutral_plane.decode(base + np.r_[0.0, x], joint.scale)
        else:
            bounds = np.r_[np.deg2rad(o.rotation_bound_deg), np.full(2, o.translation_bound_fraction)]
            decode = lambda x: joint.neutral_plane.decode(base + x, joint.scale)
    else:
        if fixed_sagittal:
            raise ValueError("A fixed sagittal angle requires anatomical neutral frames.")
        bounds = np.r_[np.full(3, np.deg2rad(o.rotation_bound_deg)), np.full(3, o.translation_bound_fraction)]
        decode = lambda x: perturb(seed, x, joint.scale)
    fit = least_squares(
        lambda x: joint.residual(decode(x), gap_factor, center_factor=center_factor),
        np.zeros(len(bounds)),
        bounds=(-bounds, bounds),
        method="trf",
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=max_nfev or o.max_nfev,
        ftol=2e-6,
        xtol=2e-6,
        gtol=2e-7,
    )
    H = validate_rigid(decode(fit.x))
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    return Candidate(
        H,
        float(np.dot(fit.fun, fit.fun)),
        bool(fit.success),
        int(fit.nfev),
        str(fit.message),
        bool(np.any(np.abs(fit.x) > 0.98 * bounds)),
        singular.tolist(),
    )


def fit_joint(joint):
    """Multistart fit of a joint; returns the best :class:`Candidate` and all candidates."""
    o = joint.options
    seeds = [joint.initial]
    anatomical_seed = joint.landmark_seed()
    if anatomical_seed is not None:
        seeds.insert(0, anatomical_seed)
    rng = np.random.default_rng(o.seed + joint.index * 997)
    while len(seeds) < o.starts:
        x = np.r_[rng.normal(0, np.deg2rad(12), 3), rng.normal(0, 0.05, 3)]
        seeds.append(perturb(seeds[0], x, joint.scale))
    candidates = [optimize_joint(joint, seed) for seed in seeds[: o.starts]]
    # Rank by a collision check independent of the fitting samples, then by energy.
    for c in candidates:
        m = joint.metrics(c.transform)
        penetration = m["max_sampled_penetration"]
        c._penetration = penetration if penetration is not None else 0.0
    candidates.sort(key=lambda c: (c._penetration > joint.penetration_tolerance * joint.scale, c.score))
    return candidates[0], candidates


def _aabb_corners(mesh):
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    return np.array([[x, y, z] for x in [lo[0], hi[0]] for y in [lo[1], hi[1]] for z in [lo[2], hi[2]]])


def _transformed_aabb(corners, T):
    P = transform(T, corners)
    return P.min(axis=0), P.max(axis=0)


def pair_aabb(mesh, T):
    """Axis-aligned bounding box of a mesh under transform ``T`` as ``(lo, hi)``."""
    return _transformed_aabb(_aabb_corners(mesh), T)


def aabb_overlap(a, b):
    """True if two ``(lo, hi)`` boxes intersect."""
    return bool(np.all(a[1] >= b[0]) and np.all(b[1] >= a[0]))


class _ExactMemo:
    """Bounded exact-key LRU; instances live only inside one refinement call.

    Keys contain all transform bytes, with no rounding or pose tolerance.
    Stored arrays are internal and must not be modified. The residual returned
    to SciPy is a fresh concatenation, so SciPy cannot modify cached entries.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.values = OrderedDict()

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
        return value

    def put(self, key, value):
        self.values[key] = value
        self.values.move_to_end(key)
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)


def refine_column(meshes, joints, pairwise, options):
    """Jointly refine all poses with a smoothness prior and nonadjacent collision terms.

    Returns ``(poses, report)`` where ``poses`` are world transforms of each
    bone's locally centred geometry. Joint residuals are memoised by exact
    transform bytes so sparse finite differences never recompute unchanged joints.
    """
    poses = [rigid(translation=meshes[0].origin)]
    for candidate in pairwise:
        poses.append(poses[-1] @ candidate.transform)
    if len(meshes) < 3:
        return poses, {"performed": False, "reason": "A single joint is already optimized."}
    scales = [m.scale for m in meshes]
    reference = [c.transform for c in pairwise]
    nonadjacent = [(i, j) for i in range(len(meshes)) for j in range(i + 2, len(meshes))]
    probes = [m.sample(options.collision_samples, seed=options.seed + 131).points for m in meshes]
    penetration_tolerance = max(j.penetration_tolerance for j in joints)
    constrained = joints[0].neutral_plane is not None
    width = 3 if constrained else 6
    if constrained:
        frames = joints[0]._fitting_config()["neutral_frames"]
        constraints = [NeutralPlane(meshes[0], m, frames) for m in meshes[1:]]
        base = [c.encode(inverse(poses[0]) @ T, s) for c, T, s in zip(constraints, poses[1:], scales[1:])]
    # Geometry, samples, references and options are fixed during this solve.
    # A sparse finite-difference step changes only a few joint transforms.
    # Reuse only byte-identical transforms; never approximate a new pose.
    joint_memo = [_ExactMemo(16) for joint in joints]
    collision_memo = _ExactMemo(256)
    corners = [_aabb_corners(m) for m in meshes]
    pair_i, pair_j = np.asarray(nonadjacent, dtype=int).T

    def decode(x):
        if constrained:
            return [poses[0]] + [
                poses[0] @ c.decode(p + x[k * 3 : k * 3 + 3], scales[k + 1])
                for k, (c, p) in enumerate(zip(constraints, base))
            ]
        return [poses[0]] + [
            perturb(poses[i], x[(i - 1) * 6 : i * 6], scales[i]) for i in range(1, len(meshes))
        ]

    def residual(x, structure=False):
        T = decode(x)
        values, dependencies = [], []
        deviations = []
        for i, joint in enumerate(joints):
            H = inverse(T[i]) @ T[i + 1]
            key = H.tobytes()
            cached = joint_memo[i].get(key)
            if cached is None:
                r = joint.residual(H)
                dev = pose_difference(H, reference[i], joint.scale)
                # Express rotational corrections in each anterior bone's anatomical
                # frame. Preserve the pairwise reference curvature itself.
                dev[:3] = joint.frame.T @ dev[:3]
                dev[3:] = joint.frame.T @ dev[3:]
                joint_memo[i].put(key, (r, dev))
            else:
                r, dev = cached
            values.append(r)
            if structure:
                dependencies.append(({i, i + 1}, len(r)))
            deviations.append(dev)
        for i in range(len(deviations) - 1):
            if (
                joints[i].frame_known
                and joints[i + 1].frame_known
                and options.smoothness_weight
                and joints[i].region == joints[i + 1].region
            ):
                r = math.sqrt(options.smoothness_weight) * (deviations[i + 1] - deviations[i])
                values.append(r)
                if structure:
                    dependencies.append(({i, i + 1, i + 2}, len(r)))
        boxes = np.asarray([_transformed_aabb(c, pose) for c, pose in zip(corners, T)])
        # The same >= tests as aabb_overlap, batched without changing pair order.
        overlapping = np.all(boxes[pair_i, 1] >= boxes[pair_j, 0], axis=1) & np.all(
            boxes[pair_j, 1] >= boxes[pair_i, 0], axis=1
        )
        collision_values = np.zeros((len(nonadjacent), 2))
        for pair_index in np.flatnonzero(overlapping):
            i, j = nonadjacent[pair_index]
            H = inverse(T[i]) @ T[j]
            key = (i, j, H.tobytes())
            r = collision_memo.get(key)
            if r is None:
                r = np.zeros(2)
                scale = math.sqrt(scales[i] * scales[j])
                for k, target, P in [
                    (0, meshes[i], transform(H, probes[j])),
                    (1, meshes[j], transform(inverse(H), probes[i])),
                ]:
                    if target.closed:
                        e = target.penetration(P, scale, penetration_tolerance)
                        r[k] = math.sqrt(options.penetration_weight * np.mean(e * e))
                collision_memo.put(key, r)
            collision_values[pair_index] = r
        values.append(collision_values.ravel())
        if structure:
            # Keep every dependency, including currently separated pairs: a
            # later step may bring those meshes into contact.
            dependencies.extend(({i, j}, 2) for i, j in nonadjacent)
        result = np.concatenate(values)
        return (result, dependencies) if structure else result

    x0 = np.zeros(width * (len(meshes) - 1))
    r0, dependencies = residual(x0, structure=True)
    sparsity = lil_matrix((len(r0), len(x0)), dtype=int)
    row = 0
    for indices, length in dependencies:
        for i in indices:
            if i:
                sparsity[row : row + length, (i - 1) * width : i * width] = 1
        row += length
    step_bounds = (
        np.r_[np.deg2rad(12), 0.25, 0.25]
        if constrained
        else np.r_[np.full(3, np.deg2rad(12)), np.full(3, 0.25)]
    )
    bounds = np.tile(step_bounds, len(meshes) - 1)
    result = least_squares(
        residual,
        x0,
        bounds=(-bounds, bounds),
        jac_sparsity=sparsity.tocsr(),
        diff_step=1e-4,
        max_nfev=options.global_max_nfev,
        ftol=2e-6,
        xtol=2e-6,
        gtol=2e-7,
    )
    improved = float(result.fun @ result.fun) <= float(r0 @ r0) + 1e-12
    answer = decode(result.x) if improved else poses
    for T in answer:
        validate_rigid(T)
    return answer, {
        "performed": True,
        "accepted": improved,
        "converged": bool(result.success),
        "free_parameters_per_moving_bone": width,
        "initial_score": float(r0 @ r0),
        "final_score": float(result.fun @ result.fun),
        "nfev": int(result.nfev),
        "reason": str(result.message),
        "search_boundary_reached": bool(np.any(np.abs(result.x) > 0.98 * bounds)),
    }


def evaluate_uncertainty(joint, best, candidates):
    """Stability diagnostics for a fitted joint: ambiguity between starts and sensitivity to settings."""
    o, s = joint.options, joint.scale
    alternatives = []
    threshold = best.score * 1.1 + 1e-6
    for c in candidates:
        d = pose_difference(c.transform, best.transform, s)
        alternatives.append(
            {
                "score": c.score,
                "converged": c.converged,
                "angle_difference_deg": float(np.rad2deg(np.linalg.norm(d[:3]))),
                "translation_difference_fraction": float(np.linalg.norm(d[3:])),
                "similarly_scoring": bool(c.score <= threshold),
                "transform_b_local_to_a_local": c.transform.tolist(),
            }
        )
    sensitivity = []
    if o.sensitivity:
        for factor in o.clearance_factors:
            c = optimize_joint(joint, best.transform, gap_factor=factor)
            d = pose_difference(c.transform, best.transform, s)
            sensitivity.append(
                {
                    "parameter": "clearance",
                    "factor": factor,
                    "angle_change_deg": float(np.rad2deg(np.linalg.norm(d[:3]))),
                    "translation_change_fraction": float(np.linalg.norm(d[3:])),
                    "converged": c.converged,
                    "score": c.score,
                }
            )
        for factor in o.centering_factors:
            c = optimize_joint(joint, best.transform, center_factor=factor)
            d = pose_difference(c.transform, best.transform, s)
            sensitivity.append(
                {
                    "parameter": "centering",
                    "factor": factor,
                    "angle_change_deg": float(np.rad2deg(np.linalg.norm(d[:3]))),
                    "translation_change_fraction": float(np.linalg.norm(d[3:])),
                    "converged": c.converged,
                    "score": c.score,
                }
            )
        if any(
            "faces" not in p.a.spec
            or "faces" not in p.b.spec
            or "automatic_boundary" in p.a.spec
            or "automatic_boundary" in p.b.spec
            for p in joint.pairs
        ):
            for factor in o.patch_radius_factors:
                try:
                    perturbed = Joint(
                        joint.a,
                        joint.b,
                        joint._fitting_config(),
                        o,
                        joint.index,
                        radius_factor=factor,
                        _config_snapshot=joint._config_snapshot,
                    )
                    c = optimize_joint(perturbed, best.transform)
                    d = pose_difference(c.transform, best.transform, s)
                    sensitivity.append(
                        {
                            "parameter": "patch_radius",
                            "factor": factor,
                            "angle_change_deg": float(np.rad2deg(np.linalg.norm(d[:3]))),
                            "translation_change_fraction": float(np.linalg.norm(d[3:])),
                            "converged": c.converged,
                            "score": c.score,
                        }
                    )
                except ValueError as error:
                    sensitivity.append(
                        {
                            "parameter": "patch_radius",
                            "factor": factor,
                            "converged": False,
                            "error": str(error),
                        }
                    )
    sv = np.asarray(best.singular_values)
    condition = float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 1e-12 else None
    ambiguous = any(
        x["similarly_scoring"]
        and (
            x["angle_difference_deg"] > o.ambiguity_angle_deg
            or x["translation_difference_fraction"] > joint.ambiguity_translation
        )
        for x in alternatives
    )

    def moved(x):
        return (
            not x["converged"]
            or x.get("angle_change_deg", 0) > o.ambiguity_angle_deg
            or x.get("translation_change_fraction", 0) > joint.ambiguity_translation
        )

    sensitive = any(moved(x) for x in sensitivity if x["parameter"] != "centering")
    centering_dependent = any(moved(x) for x in sensitivity if x["parameter"] == "centering")
    return {
        "multistart_candidates": alternatives,
        "sensitivity": sensitivity,
        "ambiguous": ambiguous,
        "sensitive": sensitive,
        "centering_dependent": centering_dependent,
        "jacobian_singular_values": best.singular_values,
        "jacobian_condition_number": condition,
        "poorly_identified": condition is None or condition > 1e5,
        "note": (
            "Stability diagnostics are not statistical confidence intervals or proof of a global optimum."
        ),
    }


def _fit_joint_task(joints, index):
    """Multistart fit and stability assessment of one joint (independent of all other joints)."""
    joint = joints[index]
    best, candidates = fit_joint(joint)
    return best, candidates, evaluate_uncertainty(joint, best, candidates)


def fit_column(meshes, config, options=None, workers=1):
    """Return original-coordinate transforms, locally centered poses, and report.

    ``workers`` > 1 fits joints in parallel processes. Each joint's fit is a
    pure function of that joint alone, so results are identical to the serial
    loop; only wall-clock time changes.
    """
    options = (options or Options()).validate()
    if len(meshes) < 2:
        raise ValueError("At least two ordered vertebrae are required.")
    if len({m.name for m in meshes}) != len(meshes):
        raise ValueError("Mesh names must be unique.")
    config = copy.deepcopy(config)
    if "mesh_coordinate_system" not in config:
        raise ValueError("Specify mesh_coordinate_system (RAS or LPS); PLY has no coordinate metadata.")
    config["mesh_coordinate_system"] = coordinate_system(config["mesh_coordinate_system"])
    if config.get("length_unit", "mm") != "mm":
        raise ValueError("Slicer export currently requires coordinates in millimeters (length_unit='mm').")
    valid_pairs = {(a.name, b.name) for a, b in zip(meshes[:-1], meshes[1:])}
    for entry in config.get("joints", []):
        if (entry.get("a"), entry.get("b")) not in valid_pairs:
            raise ValueError(
                "Joint definition is absent or not adjacent in the supplied order: "
                f"{entry.get('a')}, {entry.get('b')}"
            )
    # Separate from the mutable report configuration returned to the caller.
    config_snapshot = copy.deepcopy(config)
    joints = [
        Joint(a, b, config, options, i, _config_snapshot=config_snapshot)
        for i, (a, b) in enumerate(zip(meshes[:-1], meshes[1:]))
    ]
    bests, alternatives, uncertainty = [], [], []
    for i, (best, candidates, stability) in enumerate(
        parallel_map(_fit_joint_task, len(joints), joints, workers)
    ):
        LOG.info("Fitted joint %d/%d: %s", i + 1, len(joints), joints[i].name)
        bests.append(best)
        alternatives.append(candidates)
        uncertainty.append(stability)
    LOG.info("Refining the full column with the fitted joint constraints")
    poses, global_report = refine_column(meshes, joints, bests, options)
    LOG.info("Checking fitted surfaces, dense penetration samples and triangle intersections")
    report_joints = []
    for i, (joint, best, unc) in enumerate(zip(joints, bests, uncertainty)):
        H = inverse(poses[i]) @ poses[i + 1]
        metrics = joint.metrics(H, final=True)
        reasons = []
        if joint.noise_source == "mesh_edge_heuristic":
            reasons.append("noise_floor_estimated_from_mesh")
        if config.get("automatic_inference"):
            reasons.append("automatically_inferred_contact_reference")
        if any(m.self_intersection is not None for m in (joint.a, joint.b)):
            reasons.append("input_self_intersection")
        if any(m.connected_components > 1 for m in (joint.a, joint.b)):
            reasons.append("disconnected_mesh_components")
        if any(m.cleaning.get("components_removed", 0) for m in (joint.a, joint.b)):
            reasons.append("mesh_components_removed")
        if not joint.anatomical or any(p.kind == "unclassified" for p in joint.pairs):
            reasons.append("geometry_only_patch_identity")
        if (
            not any(p.kind == "centrum" for p in joint.pairs)
            or sum(p.kind == "facet" for p in joint.pairs) < 2
        ):
            reasons.append("incomplete_anatomical_neutral_reference")
        if not metrics["signed_distance_reliable"]:
            reasons.append("open_or_nonmanifold_mesh_sign_unverified")
        depth = metrics["max_sampled_penetration"]
        if depth is not None and depth > joint.penetration_tolerance * joint.scale:
            reasons.append("penetration_exceeds_tolerance")
        if metrics["intersection_detected"]:
            # Shallow crossings between two noisy surfaces at their nominal gap
            # are expected; only unmeasurable (open mesh) or deeper ones need review.
            if (
                not metrics["signed_distance_reliable"]
                or depth is None
                or depth > joint.penetration_tolerance * joint.scale
            ):
                reasons.append("triangle_intersections_require_review")
            else:
                metrics["contact_within_noise_tolerance"] = True
        for p in metrics["patches"]:
            for side in (p["a_to_b"], p["b_to_a"]):
                if side["rms_gap_error_fraction"] > joint.acceptable_surface_error:
                    reasons.append("poor_surface_fit")
                if side["coverage_fraction"] < options.minimum_coverage:
                    reasons.append("insufficient_surface_coverage")
        if not best.converged:
            reasons.append("pairwise_optimizer_not_converged")
        if best.boundary:
            reasons.append("pairwise_search_boundary_reached")
        if unc["ambiguous"]:
            reasons.append("multiple_similarly_scoring_poses")
        if unc["sensitive"]:
            reasons.append("sensitive_to_model_assumptions")
        if unc["centering_dependent"]:
            reasons.append("pose_depends_on_patch_centering")
        if any(
            p["anchor_tangential_offset_mm"] > options.centering_tolerance_fraction * joint.scale
            and not p["spherical_centrum"]
            for p in metrics["patches"]
        ):
            reasons.append("anchor_offset_exceeds_tolerance")
        if not joint.a.self_intersection_checked or not joint.b.self_intersection_checked:
            reasons.append("self_intersection_preflight_skipped")
        if unc["poorly_identified"]:
            reasons.append("weakly_identified_pose")
        if options.starts < 2:
            reasons.append("multiple_starts_not_evaluated")
        if not options.sensitivity:
            reasons.append("sensitivity_not_evaluated")
        final_change = pose_difference(H, best.transform, joint.scale)
        if (
            np.rad2deg(np.linalg.norm(final_change[:3])) > options.ambiguity_angle_deg
            or np.linalg.norm(final_change[3:]) > joint.ambiguity_translation
        ):
            reasons.append("global_refinement_changed_pairwise_reference")
        ra, rka = anatomical_frame(joint.a, joint.roles_a)
        rb, rkb = anatomical_frame(joint.b, joint.roles_b)
        report_joints.append(
            {
                "name": joint.name,
                "a": joint.a.name,
                "b": joint.b.name,
                "status": "needs_review" if reasons else "passed_geometric_checks",
                "review_reasons": sorted(set(reasons)),
                "local_scale_mm": joint.scale,
                "effective_tolerances": joint.tolerances,
                "transform_b_local_to_a_local": H.tolist(),
                "anatomical_orientation_b_in_a": (ra.T @ H[:3, :3] @ rb).tolist() if rka and rkb else None,
                "pairwise_score": best.score,
                "final_score": float(np.sum(joint.residual(H) ** 2)),
                "pairwise_converged": best.converged,
                "pairwise_nfev": best.nfev,
                "metrics": metrics,
                "uncertainty": unc,
            }
        )
        if joint.neutral_plane is not None:
            report_joints[-1]["neutral_plane"] = joint.neutral_plane.diagnostics(H)
            report_joints[-1]["input_neutral_plane"] = joint.neutral_plane.diagnostics(joint.initial)
            report_joints[-1]["anatomical_orientation_b_in_a"] = (
                joint.neutral_plane.fa.T @ H[:3, :3] @ joint.neutral_plane.fb
            ).tolist()
    nonadjacent = []
    boxes = [pair_aabb(m, T) for m, T in zip(meshes, poses)]
    for i in range(len(meshes)):
        for j in range(i + 2, len(meshes)):
            if not aabb_overlap(boxes[i], boxes[j]):
                continue
            H = inverse(poses[i]) @ poses[j]
            hit = intersects(meshes[i], meshes[j], H)
            depth = 0.0
            verified = meshes[i].closed and meshes[j].closed
            for target, other, T in [(meshes[i], meshes[j], H), (meshes[j], meshes[i], inverse(H))]:
                if target.closed:
                    P = np.vstack([other.vertices, other.centers, other.sample(2048, seed=71).points])
                    depth = max(depth, float(target.penetration(transform(T, P)).max()))
            if hit or depth > 0 or not verified:
                nonadjacent.append(
                    {
                        "a": meshes[i].name,
                        "b": meshes[j].name,
                        "intersection_detected": hit,
                        "max_sampled_penetration": depth,
                        "signed_distance_reliable": verified,
                    }
                )
    reasons = []
    if any(j["status"] == "needs_review" for j in report_joints):
        reasons.append("one_or_more_joints_require_review")
    if nonadjacent:
        reasons.append("nonadjacent_meshes_require_review")
    if global_report.get("performed") and (
        not global_report.get("converged") or global_report.get("search_boundary_reached")
    ):
        reasons.append("global_optimizer_requires_review")
    matrices = np.stack([T @ rigid(translation=-m.origin) for m, T in zip(meshes, poses)])
    for T in matrices:
        validate_rigid(T)
    report = {
        "schema_version": 1,
        "software": {
            "name": "neutral_pose",
            "version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "vtk": vtk.vtkVersion.GetVTKVersion(),
        },
        "definition": (
            "Estimated osteological neutral articulation under the recorded patch and clearance assumptions."
        ),
        "status": "needs_review" if reasons else "passed_geometric_checks",
        "review_reasons": reasons,
        "mesh_coordinate_system": config["mesh_coordinate_system"],
        "length_unit": "mm",
        "mesh_order": [m.name for m in meshes],
        "first_vertebra_fixed": True,
        "options": asdict(options),
        "configuration": config,
        "mesh_quality": [
            {
                "name": m.name,
                "vertices": len(m.vertices),
                "triangles": len(m.faces),
                "closed_manifold_edges": m.topologically_closed,
                "boundary_edges": m.boundary_edges,
                "signed_distance_reliable": m.closed,
                "nonmanifold_edges": m.nonmanifold_edges,
                "median_edge_mm": m.resolution,
                "connected_components": m.connected_components,
                "self_intersection_preflight": (
                    "intersections_detected"
                    if m.self_intersection is not None
                    else "passed"
                    if m.self_intersection_checked
                    else "skipped_too_many_triangles"
                ),
                "self_intersection_pair": m.self_intersection,
                "cleaning": m.cleaning,
            }
            for m in meshes
        ],
        "joints": report_joints,
        "global_refinement": global_report,
        "nonadjacent_contacts": nonadjacent,
        "limitations": [
            "Geometric checks do not establish physiological resting posture.",
            "Signed-distance interpretation assumes consistently oriented, non-self-intersecting boundaries.",
            "Penetration depths and coverage are sampled estimates; "
            "triangle intersection detection is independent.",
            "Sensitivity is assessed per joint about its pairwise fit, not as a joint statistical posterior.",
        ],
    }
    return matrices, poses, report, joints


def deep_merge(a, b):
    """Recursively merge dict ``b`` over ``a`` without mutating either."""
    result = copy.deepcopy(a)
    for k, v in b.items():
        result[k] = (
            deep_merge(result[k], v)
            if isinstance(v, dict) and isinstance(result.get(k), dict)
            else copy.deepcopy(v)
        )
    return result


def numeric_key(path):
    """Natural sort key: numbers inside a filename compare numerically."""
    parts = re.split(r"(\d+)", Path(path).name.lower())
    return tuple((0, int(x)) if x.isdigit() else (1, x) for x in parts)


def collect_specimen(directory, config):
    """Load the meshes and landmarks named by ``config['meshes']`` from ``directory``."""
    directory = Path(directory)
    supported = {".ply", ".vtp", ".stl", ".obj"}
    available = {p.name: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in supported}
    if "order" in config:
        names = config["order"]
        if not isinstance(names, list) or len(set(names)) != len(names):
            raise ValueError("order must be a list of unique mesh filenames.")
        if any(n not in available for n in names):
            raise ValueError(f"Missing ordered mesh files in {directory}.")
        paths = [available[n] for n in names]
    else:
        paths = sorted([p for p in available.values() if re.search(r"\d", p.stem)], key=numeric_key)
        dropped = sorted(set(available) - {p.name for p in paths})
        if dropped:
            LOG.warning(
                '%s: ignoring meshes without a number in their name (use "order" to include them): %s',
                directory.name,
                ", ".join(dropped),
            )
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Multiple mesh formats have the same basename; specify an explicit order.")
    landmark_files = []
    for mesh_path in paths:
        matches = [
            p
            for folder in (directory / "LMKs", directory / "LMKs_json")
            if folder.is_dir()
            for p in folder.iterdir()
            if p.name in (mesh_path.stem + ".fcsv", mesh_path.stem + ".mrk.json")
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple landmark files match {mesh_path.name}; keep a single authoritative file."
            )
        landmark_files.append(matches[0] if matches else None)
    return paths, landmark_files


def save_result(directory, meshes, matrices, poses, report, joints):
    """Write posed meshes, landmarks, transforms, patches and reports to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    colors = [m.colors.copy() for m in meshes]
    patch_dir = directory / "patches"
    patch_dir.mkdir()
    for i, joint in enumerate(joints):
        for pair in joint.pairs:
            for side, patch, mesh_idx in [("a", pair.a, i), ("b", pair.b, i + 1)]:
                # Export triangle IDs in each loaded, triangulated input mesh.
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", pair.name)
                np.save(patch_dir / f"joint_{i + 1:03d}_{name}_{side}_faces.npy", patch.ids)
                # Face ids index the cleaned mesh; centers are in original input coordinates.
                np.save(
                    patch_dir / f"joint_{i + 1:03d}_{name}_{side}_centers.npy",
                    patch.mesh.centers[patch.ids] + patch.mesh.origin,
                )
                ids = np.unique(patch.mesh.faces[patch.ids])
                colors[mesh_idx][ids] = [230, 85, 40] if side == "a" else [0, 130, 180]
    append = vtk.vtkAppendPolyData()
    (directory / "meshes_vtp").mkdir()
    for mesh, M, T, color in zip(meshes, matrices, poses, colors):
        write_mesh(directory / f"neutral_{mesh.name}.ply", mesh, T, color)
        write_mesh(directory / "meshes_vtp" / f"neutral_{mesh.name}.vtp", mesh, T, color)
        append.AddInputData(polydata(transform(T, mesh.vertices), mesh.faces, color))
        if mesh.landmarks is not None:
            lm = mesh.landmarks
            P = transform(M, lm.points)
            output_coord = report["mesh_coordinate_system"]
            write_landmarks(
                directory / "LMKs_json" / f"neutral_{mesh.name}.mrk.json", P, lm.labels, output_coord
            )
    append.Update()
    scene = oriented_poly(append.GetOutput())
    write_precise_ply(
        directory / "column_neutral.ply",
        vtk_to_numpy(scene.GetPoints().GetData()),
        faces_of(scene),
        vtk_to_numpy(scene.GetPointData().GetArray("RGB")),
    )
    write_vtp(directory / "column_neutral.vtp", scene)
    np.save(directory / "neutral_transforms.npy", matrices)
    transforms_doc = {
        "coordinate_system": report["mesh_coordinate_system"],
        "length_unit": "mm",
        "convention": (
            "Column vectors: [x_neutral,1] = matrix @ [x_input,1]. Rotations are proper; no scaling."
        ),
        "meshes": [
            {
                "name": m.name,
                "input": m.source,
                "matrix_input_to_neutral": M.tolist(),
                "matrix_neutral_to_input": inverse(M).tolist(),
            }
            for m, M in zip(meshes, matrices)
        ],
    }
    (directory / "transforms.json").write_text(
        json.dumps(transforms_doc, indent=2, allow_nan=False), encoding="utf-8"
    )
    (directory / "neutral_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (directory / "joint_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "joint",
                "status",
                "local_scale_mm",
                "max_sampled_penetration_mm",
                "intersection_detected",
                "review_reasons",
            ]
        )
        for joint in report["joints"]:
            writer.writerow(
                [
                    joint["name"],
                    joint["status"],
                    joint["local_scale_mm"],
                    joint["metrics"]["max_sampled_penetration"],
                    joint["metrics"]["intersection_detected"],
                    ";".join(joint["review_reasons"]),
                ]
            )
    note = (
        "This is an estimated osteological reference under the settings in neutral_report.json.\n"
        f"Status: {report['status']}\n"
        "Orange and blue mark fitted surface patches, not collision classifications.\n"
        "Check joint_metrics.csv and the per-joint review reasons before downstream analyses.\n"
        "PLY does not encode RAS/LPS or units: use the coordinate system recorded in transforms.json "
        "when loading meshes.\n"
    )
    note += (
        "For Slicer/VTK use column_neutral.vtp or meshes_vtp/ to preserve double precision; "
        "some PLY readers downcast coordinates.\n"
    )
    (directory / "READ_FIRST.txt").write_text(note, encoding="utf-8")


def process_specimen(source, destination, config, options, overwrite=False):
    """Run the configuration-driven pipeline on one specimen folder and save results."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source.is_relative_to(destination):
        raise ValueError("Output must not replace an input directory or one of its parents.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {destination}. Use --overwrite to replace that result.")
    local_config = source / "neutral_config.json"
    if local_config.exists():
        config = deep_merge(config, json.loads(local_config.read_text(encoding="utf-8")))
    config = deep_merge(config, config.pop("_cli_overrides", {}))
    if "mesh_coordinate_system" not in config:
        raise ValueError("Set --mesh-coordinates RAS/LPS or mesh_coordinate_system in the config.")
    paths, landmarks = collect_specimen(source, config)
    if len(paths) < 2:
        raise ValueError("Specimen must contain at least two ordered mesh files.")
    mesh_coord = coordinate_system(config["mesh_coordinate_system"])
    options = copy.deepcopy(options)
    for key, value in config.get("options", {}).items():
        if key not in asdict(options):
            raise ValueError(f"Unknown optimizer option {key!r}.")
        setattr(options, key, value)
    options.validate()
    meshes = [
        Mesh.read(
            p,
            read_landmarks(lm, mesh_coord) if lm else None,
            options.keep_largest_component,
            options.preflight_triangles,
        )
        for p, lm in zip(paths, landmarks)
    ]
    matrices, poses, report, joints = fit_column(meshes, config, options)
    report["inputs"] = [{"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
    report["landmark_inputs"] = [
        {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in landmarks if p
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="neutral-pose-", dir=destination.parent))
    backup = None
    try:
        save_result(temporary, meshes, matrices, poses, report, joints)
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix="neutral-previous-", dir=destination.parent))
            backup.rmdir()
            destination.rename(backup)
        temporary.rename(destination)
        if backup is not None:
            shutil.rmtree(backup)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def process_root(
    root, out_root="neutral_out_batch", config=None, options=None, overwrite=False, specimen=False
):
    """Run :func:`process_specimen` over a specimen folder or every specimen under a root."""
    root, out_root = Path(root).resolve(), Path(out_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if root == out_root or root.is_relative_to(out_root):
        raise ValueError("Output cannot be the input root or one of its parents.")
    config, options = config or {}, options or Options()
    specs = (
        [root]
        if specimen
        else [
            p
            for p in sorted(root.iterdir(), key=numeric_key)
            if p.is_dir() and p != out_root and not p.name.startswith(".")
        ]
    )
    if not specs:
        raise ValueError("No specimen directories found. Use --specimen for a single folder of meshes.")
    results = []
    for sd in specs:
        try:
            output = out_root if specimen else out_root / sd.name
            report = process_specimen(sd, output, config, options, overwrite)
            results.append({"specimen": sd.name, "status": report["status"], "output": str(output)})
            LOG.info("%s: %s", sd.name, report["status"])
        except Exception as error:
            LOG.error("%s: %s", sd.name, error)
            results.append({"specimen": sd.name, "status": "failed", "error": str(error)})
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "batch_summary.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
    )
    return results


def main(argv=None):
    """Command-line entry point for the manual, configuration-driven workflow (``neutral-pose``)."""
    parser = argparse.ArgumentParser(
        prog="neutral-pose", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "root", type=Path, help="Folder containing specimen subfolders (or meshes with --specimen)."
    )
    parser.add_argument("--out", type=Path, default=Path("neutral_out_batch"))
    parser.add_argument("--specimen", action="store_true", help="Process root itself as one specimen.")
    parser.add_argument(
        "--config", type=Path, help="JSON anatomical roles, patch definitions and optimizer options."
    )
    parser.add_argument(
        "--mesh-coordinates", choices=["RAS", "LPS"], help="Required unless provided in configuration."
    )
    parser.add_argument(
        "--require-anatomy", action="store_true", help="Reject geometry-only candidate discovery."
    )
    parser.add_argument("--starts", type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip assumption sensitivity; outputs are flagged for review.",
    )
    parser.add_argument(
        "--noise-floor-mm",
        type=float,
        help="Surface error allowance in mm; overrides automatic estimation, including 0. "
        "Default: no noise-based widening. The optional edge heuristic is configured in JSON.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing result folders after a successful fit."
    )
    parser.add_argument(
        "--strict", action="store_true", help="Also exit nonzero when any specimen requires review."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        explicit = {}
        if args.mesh_coordinates:
            explicit["mesh_coordinate_system"] = args.mesh_coordinates
        if args.require_anatomy:
            explicit["require_anatomy"] = True
        explicit_options = {
            key: getattr(args, key) for key in ("starts", "samples", "seed") if getattr(args, key) is not None
        }
        if args.no_sensitivity:
            explicit_options["sensitivity"] = False
        if args.noise_floor_mm is not None:
            explicit_options["noise_floor_mm"] = args.noise_floor_mm
        if explicit_options:
            explicit["options"] = explicit_options
        config = deep_merge(config, explicit)
        config["_cli_overrides"] = explicit
        options = Options()
        result = process_root(args.root, args.out, config, options, args.overwrite, args.specimen)
        return int(
            any(
                x["status"] == "failed" or (args.strict and x["status"] != "passed_geometric_checks")
                for x in result
            )
        )
    except Exception as error:
        LOG.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
