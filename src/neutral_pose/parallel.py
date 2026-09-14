"""Run independent per-joint (or per-mesh) tasks in worker processes.

Each task is a pure function of shared, read-only inputs and an index, so the
work can be distributed without changing any arithmetic: every worker executes
exactly the code the serial loop would, in the same order for its own item,
and results are returned in item order. The serial path (``workers <= 1``)
calls the task function directly and is the reference behaviour.

The shared inputs are handed to workers once (via the pool initializer) rather
than once per task; on platforms that fork, they are inherited without copying.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any

LOG = logging.getLogger(__name__)

_SHARED: Any = None


def resolve_workers(workers: int | str | None) -> int:
    """Turn a CLI/API workers value (int, ``'auto'`` or ``None``) into a positive count."""
    if workers is None or workers == "auto":
        return max(1, os.cpu_count() or 1)
    count = int(workers)
    if count < 1:
        raise ValueError("workers must be a positive integer or 'auto'.")
    return count


def _initialize(shared) -> None:
    global _SHARED
    _SHARED = shared


def _call(task, index):
    return task(_SHARED, index)


def _context():
    # Forking avoids re-importing VTK and copying the meshes for every worker.
    # Elsewhere the default (spawn) start method is used and the shared inputs
    # are pickled once per worker.
    if sys.platform.startswith("linux"):
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def parallel_map(task: Callable[[Any, int], Any], count: int, shared, workers: int) -> Iterator:
    """Yield ``task(shared, i)`` for ``i`` in ``range(count)``, in order.

    With ``workers <= 1`` (or a single item) this is a plain loop in the
    calling process. Otherwise items are evaluated by a process pool.
    """
    workers = min(resolve_workers(workers), count)
    if workers <= 1:
        for index in range(count):
            yield task(shared, index)
        return
    LOG.info("Using %d worker processes for %d independent tasks", workers, count)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=_context(), initializer=_initialize, initargs=(shared,)
    ) as pool:
        yield from pool.map(_call, repeat(task), range(count))
