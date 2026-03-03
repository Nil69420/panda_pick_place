"""Artificial Potential Field (APF) force functions.

These pure functions compute the attractive and repulsive forces
that guide the end-effector toward a goal while steering around
obstacles.  They are called by
:meth:`~panda_control.robot.RobotController.move_to_field`.

When Numba is installed the heavy arithmetic is JIT-compiled; otherwise
the functions fall back to equivalent pure-NumPy implementations
automatically.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from panda_control.common_utils.jit_kernels import (
    attractive_jit,
    repulsive_jit,
    total_apf_force as _total_apf_jit,
)

# Pre-allocated empty obstacle array avoids repeated allocation when
# no obstacles are present.
_EMPTY_OBS = np.empty((0, 3), dtype=np.float64)


def _pack_obstacles(obstacles: Sequence[np.ndarray]) -> np.ndarray:
    """Convert a sequence of 1-D obstacle positions to a contiguous (N, 3) array."""
    if len(obstacles) == 0:
        return _EMPTY_OBS
    return np.ascontiguousarray(obstacles, dtype=np.float64).reshape(-1, 3)


def attractive_force(
    position: np.ndarray,
    goal: np.ndarray,
    xi: float = 1.0,
) -> np.ndarray:
    r"""Linear attractive force pulling *position* toward *goal*.

    .. math::

        F_{att} = \xi \cdot (q_{goal} - q)
    """
    return attractive_jit(
        np.ascontiguousarray(position, dtype=np.float64),
        np.ascontiguousarray(goal, dtype=np.float64),
        xi,
    )


def repulsive_force(
    position: np.ndarray,
    obstacles: Sequence[np.ndarray],
    eta: float = 0.005,
    d_thresh: float = 0.12,
) -> np.ndarray:
    r"""Sum of inverse-square repulsive forces from nearby obstacles.

    Each obstacle within *d_thresh* metres contributes:

    .. math::

        F_{rep,i} = \eta \cdot (1/d - 1/d_0) / d^2 \cdot \hat{n}

    where *d* is the Euclidean distance and *d_0* = *d_thresh*.
    """
    obs_2d = _pack_obstacles(obstacles)
    return repulsive_jit(
        np.ascontiguousarray(position, dtype=np.float64),
        obs_2d,
        obs_2d.shape[0],
        eta,
        d_thresh,
    )


def total_field_force(
    position: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[np.ndarray],
    xi: float = 1.0,
    eta: float = 0.005,
    d_thresh: float = 0.12,
) -> np.ndarray:
    """Combined attractive + repulsive force at *position*.

    Uses a fused JIT kernel so the entire computation runs in a single
    compiled call with no Python-level loop overhead.
    """
    obs_2d = _pack_obstacles(obstacles)
    return _total_apf_jit(
        np.ascontiguousarray(position, dtype=np.float64),
        np.ascontiguousarray(goal, dtype=np.float64),
        obs_2d,
        obs_2d.shape[0],
        xi,
        eta,
        d_thresh,
    )
