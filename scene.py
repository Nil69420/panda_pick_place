"""Scene manager for spawning and tracking cubes on the table.

:class:`SceneManager` creates randomly coloured cubes at random
positions on the workspace surface.  It guarantees a minimum
separation between cubes so they do not overlap.

All tuneable constants -- cube size, mass, friction, colour range,
spawn bounds, and maximum placement attempts -- are read from
``config/default.yaml`` under the ``scene`` section.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from panda_control.config import get as cfg


def _random_rgba(rng: np.random.Generator) -> np.ndarray:
    """Generate a random opaque RGBA colour within the configured range."""
    lo = cfg("scene", "cube", "color_min")
    hi = cfg("scene", "cube", "color_max")
    rgb = rng.uniform(lo, hi, size=3)
    return np.append(rgb, 1.0)


class SceneManager:
    """Spawn, query, and remove cubes in the PyBullet simulation.

    Parameters
    ----------
    sim : panda_gym.pybullet.PyBullet
        The simulation wrapper.
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.
    """

    TABLE_X_OFFSET = cfg("scene", "table", "x_offset")
    TABLE_LENGTH = cfg("scene", "table", "length")
    TABLE_WIDTH = cfg("scene", "table", "width")
    TABLE_HEIGHT = cfg("scene", "table", "height")

    def __init__(self, sim, rng: Optional[np.random.Generator] = None) -> None:
        self.sim = sim
        self.rng = rng or np.random.default_rng()
        self._spawned: Dict[str, dict] = {}

    def spawn_random_cubes(
        self,
        n: int = None,
        half_extent: float = None,
        mass: float = None,
        z_offset: float = None,
        xy_range: Tuple[float, float] = None,
    ) -> List[str]:
        """Create *n* randomly positioned cubes on the table.

        Returns a list of body names (``cube_0``, ``cube_1``, ...).
        """
        if n is None:
            n = cfg("task", "default_n_cubes")
        if half_extent is None:
            half_extent = cfg("scene", "cube", "half_extent")
        if mass is None:
            mass = cfg("scene", "cube", "mass")
        if z_offset is None:
            z_offset = cfg("scene", "cube", "z_offset")
        if xy_range is None:
            xy_range = (cfg("scene", "spawn", "xy_range_x"),
                        cfg("scene", "spawn", "xy_range_y"))

        half = np.array([half_extent] * 3)
        placed: List[np.ndarray] = []
        names: List[str] = []
        sep_factor = cfg("scene", "spawn", "separation_factor")
        min_separation = half_extent * sep_factor
        max_attempts = cfg("scene", "spawn", "max_attempts")
        friction = cfg("scene", "cube", "lateral_friction")

        for i in range(n):
            name = f"cube_{i}"
            color = _random_rgba(self.rng)

            for _attempt in range(max_attempts):
                x = self.rng.uniform(-xy_range[0], xy_range[0])
                y = self.rng.uniform(-xy_range[1], xy_range[1])
                pos = np.array([x, y, half_extent + z_offset])
                if all(np.linalg.norm(pos[:2] - other[:2]) > min_separation for other in placed):
                    break
            else:
                pos = np.array([
                    self.rng.uniform(-xy_range[0], xy_range[0]),
                    self.rng.uniform(-xy_range[1], xy_range[1]),
                    half_extent + z_offset,
                ])

            self.sim.create_box(
                body_name=name,
                half_extents=half,
                mass=mass,
                position=pos,
                rgba_color=color,
                lateral_friction=friction,
            )
            placed.append(pos)
            self._spawned[name] = {
                "position": pos.copy(),
                "color": color.copy(),
                "half_extent": half_extent,
            }
            names.append(name)

        return names

    def get_cube_info(self) -> Dict[str, dict]:
        """Return metadata for every spawned cube, with live positions."""
        for name in self._spawned:
            self._spawned[name]["position"] = self.sim.get_base_position(name)
        return dict(self._spawned)

    def remove_all_cubes(self) -> None:
        """Delete every spawned cube from the simulation."""
        for name in list(self._spawned.keys()):
            if name in self.sim._bodies_idx:
                self.sim.physics_client.removeBody(self.sim._bodies_idx[name])
                del self.sim._bodies_idx[name]
        self._spawned.clear()
