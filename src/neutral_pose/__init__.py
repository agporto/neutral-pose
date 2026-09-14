"""Estimate an osteological neutral articulation from segmented vertebrae.

The public surface is intentionally small:

* :func:`neutral_pose.auto.run` / the ``neutral-pose-auto`` command take a
  specimen folder or ZIP and infer everything automatically.
* :class:`Mesh`, :class:`Options`, :func:`fit_column` and :func:`save_result`
  are the manual, configuration-driven API in :mod:`neutral_pose.core`.

Results are osteological references fitted with proper rigid transforms;
they are not measurements of physiological resting posture.
"""

from __future__ import annotations

from .core import (
    Landmarks,
    Mesh,
    Options,
    fit_column,
    inverse,
    pose_difference,
    read_landmarks,
    rigid,
    save_result,
    transform,
    validate_rigid,
    write_landmarks,
)
from .version import __version__

__all__ = [
    "Landmarks",
    "Mesh",
    "Options",
    "__version__",
    "fit_column",
    "inverse",
    "pose_difference",
    "read_landmarks",
    "rigid",
    "save_result",
    "transform",
    "validate_rigid",
    "write_landmarks",
]
