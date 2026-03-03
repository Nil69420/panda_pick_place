import numpy as np
import pytest

from panda_control.task_runner import TaskRunner


class TestReset:

    def test_reset_respawns_cubes(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=3, seed=10)
        old_positions = p.get_cube_positions()
        p.reset(seed=99)
        new_positions = p.get_cube_positions()
        assert len(new_positions) == 3
        changed = any(
            not np.allclose(old_positions[n], new_positions[n], atol=1e-3)
            for n in new_positions
        )
        assert changed
        p.close()

    def test_reset_perception_still_works(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=4, seed=20)
        p.reset(seed=77)
        detections = p.perception.perceive_overhead()
        assert len(detections) >= 1
        p.close()
