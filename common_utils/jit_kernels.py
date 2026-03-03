"""Numba JIT-accelerated kernels for performance-critical inner loops.

Provides compiled implementations of Artificial Potential Field force
computation and depth-buffer linearisation.  Falls back to equivalent
pure-NumPy versions when Numba is not installed.

Usage::

    from panda_control.common_utils.jit_kernels import (
        attractive_jit, repulsive_jit, total_apf_force, linearise_depth_buf,
    )
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Identity decorator when Numba is unavailable."""
        if args and callable(args[0]):
            return args[0]

        def _wrap(fn):
            return fn

        return _wrap


# ── APF kernels ────────────────────────────────────────────


@njit(cache=True)
def attractive_jit(position, goal, xi):
    r"""Attractive force: :math:`\xi \, (q_{\text{goal}} - q)`.

    Parameters
    ----------
    position, goal : ndarray, shape (3,)
    xi : float
        Attractive gain.

    Returns
    -------
    ndarray, shape (3,)
    """
    out = np.empty(3)
    for i in range(3):
        out[i] = xi * (goal[i] - position[i])
    return out


@njit(cache=True)
def repulsive_jit(position, obstacles, n_obs, eta, d_thresh):
    r"""Sum of inverse-square repulsive forces from nearby obstacles.

    Each obstacle row within *d_thresh* metres contributes a repulsive
    force proportional to :math:`\eta \, (1/d - 1/d_0) / d^2`.

    Parameters
    ----------
    position : ndarray, shape (3,)
    obstacles : ndarray, shape (N, 3)
        Obstacle positions packed into a contiguous 2-D array.
    n_obs : int
        Number of valid rows in *obstacles*.
    eta : float
        Repulsive gain.
    d_thresh : float
        Influence radius (metres).

    Returns
    -------
    ndarray, shape (3,)
    """
    out = np.zeros(3)
    for k in range(n_obs):
        d_sq = 0.0
        for i in range(3):
            diff = position[i] - obstacles[k, i]
            d_sq += diff * diff
        d = np.sqrt(d_sq)
        if d < 1e-6:
            d = 1e-6
        if d < d_thresh:
            mag = eta * (1.0 / d - 1.0 / d_thresh) / (d * d)
            for i in range(3):
                out[i] += mag * (position[i] - obstacles[k, i]) / d
    return out


@njit(cache=True)
def total_apf_force(position, goal, obstacles, n_obs, xi, eta, d_thresh):
    """Combined attractive + repulsive APF force in a single compiled call.

    Fusing both computations into one kernel avoids the Python-level
    overhead of calling two separate functions per simulation step.

    Parameters
    ----------
    position, goal : ndarray, shape (3,)
    obstacles : ndarray, shape (N, 3)
    n_obs : int
    xi, eta, d_thresh : float

    Returns
    -------
    ndarray, shape (3,)
    """
    out = np.empty(3)
    # Attractive component
    for i in range(3):
        out[i] = xi * (goal[i] - position[i])
    # Repulsive component
    for k in range(n_obs):
        d_sq = 0.0
        for i in range(3):
            diff = position[i] - obstacles[k, i]
            d_sq += diff * diff
        d = np.sqrt(d_sq)
        if d < 1e-6:
            d = 1e-6
        if d < d_thresh:
            mag = eta * (1.0 / d - 1.0 / d_thresh) / (d * d)
            for i in range(3):
                out[i] += mag * (position[i] - obstacles[k, i]) / d
    return out


# ── Depth linearisation ────────────────────────────────────


@njit(cache=True)
def linearise_depth_buf(depth_buffer, near, far, height, width):
    """Convert a normalised depth buffer to metric depth (metres).

    Performs the conversion element-wise without allocating intermediate
    arrays, reducing memory pressure for large frames.

    Parameters
    ----------
    depth_buffer : ndarray, shape (H, W), float32
        Raw depth buffer from PyBullet (values in [0, 1]).
    near, far : float
        Camera clipping planes (metres).
    height, width : int
        Frame dimensions.

    Returns
    -------
    ndarray, shape (H, W), float32
    """
    out = np.empty((height, width), dtype=np.float32)
    diff = far - near
    fn = far * near
    for r in range(height):
        for c in range(width):
            out[r, c] = fn / (far - diff * depth_buffer[r, c])
    return out
