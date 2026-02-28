from __future__ import annotations

from typing import Sequence

import numpy as np


def attractive_force(
    position: np.ndarray,
    goal: np.ndarray,
    xi: float = 1.0,
) -> np.ndarray:
    return xi * (goal - position)


def repulsive_force(
    position: np.ndarray,
    obstacles: Sequence[np.ndarray],
    eta: float = 0.005,
    d_thresh: float = 0.12,
) -> np.ndarray:
    force = np.zeros(len(position))
    for obs in obstacles:
        vec = position - obs
        d = np.linalg.norm(vec)
        if d < 1e-6:
            d = 1e-6
        if d < d_thresh:
            mag = eta * (1.0 / d - 1.0 / d_thresh) / (d * d)
            force += mag * (vec / d)
    return force


def total_field_force(
    position: np.ndarray,
    goal: np.ndarray,
    obstacles: Sequence[np.ndarray],
    xi: float = 1.0,
    eta: float = 0.005,
    d_thresh: float = 0.12,
) -> np.ndarray:
    return attractive_force(position, goal, xi) + repulsive_force(position, obstacles, eta, d_thresh)
