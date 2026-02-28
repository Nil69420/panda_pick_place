from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _random_rgba(rng: np.random.Generator) -> np.ndarray:
    rgb = rng.uniform(0.15, 1.0, size=3)
    return np.append(rgb, 1.0)


class SceneManager:

    TABLE_X_OFFSET = -0.3
    TABLE_LENGTH = 1.1
    TABLE_WIDTH = 0.7
    TABLE_HEIGHT = 0.4

    def __init__(self, sim, rng: Optional[np.random.Generator] = None) -> None:
        self.sim = sim
        self.rng = rng or np.random.default_rng()
        self._spawned: Dict[str, dict] = {}

    def spawn_random_cubes(
        self,
        n: int = 4,
        half_extent: float = 0.02,
        mass: float = 0.1,
        z_offset: float = 0.0,
        xy_range: Tuple[float, float] = (0.25, 0.15),
    ) -> List[str]:
        half = np.array([half_extent] * 3)
        placed: List[np.ndarray] = []
        names: List[str] = []
        min_separation = half_extent * 3

        for i in range(n):
            name = f"cube_{i}"
            color = _random_rgba(self.rng)

            for _attempt in range(200):
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
                lateral_friction=1.0,
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
        for name in self._spawned:
            self._spawned[name]["position"] = self.sim.get_base_position(name)
        return dict(self._spawned)

    def remove_all_cubes(self) -> None:
        for name in list(self._spawned.keys()):
            if name in self.sim._bodies_idx:
                self.sim.physics_client.removeBody(self.sim._bodies_idx[name])
                del self.sim._bodies_idx[name]
        self._spawned.clear()
