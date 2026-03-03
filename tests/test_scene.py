import numpy as np
import pytest


class TestSceneManager:

    def test_cube_count(self, pipeline):
        assert len(pipeline.cube_names) == 5

    def test_cube_names_format(self, pipeline):
        for i, name in enumerate(pipeline.cube_names):
            assert name == f"cube_{i}"

    def test_cube_info_has_all_cubes(self, pipeline):
        info = pipeline.scene.get_cube_info()
        assert len(info) == 5
        for name in pipeline.cube_names:
            assert name in info

    def test_cube_positions_on_table(self, pipeline):
        info = pipeline.scene.get_cube_info()
        for name, data in info.items():
            pos = data["position"]
            assert pos.shape == (3,)
            assert -0.5 < pos[0] < 0.5
            assert -0.5 < pos[1] < 0.5
            assert pos[2] > 0.0

    def test_cube_colors_are_rgba(self, pipeline):
        info = pipeline.scene.get_cube_info()
        for name, data in info.items():
            color = data["color"]
            assert color.shape == (4,)
            assert np.all(color[:3] >= 0.15)
            assert np.all(color[:3] <= 1.0)
            assert color[3] == 1.0

    def test_cube_positions_no_overlap(self, pipeline):
        info = pipeline.scene.get_cube_info()
        positions = [data["position"][:2] for data in info.values()]
        half_ext = list(info.values())[0]["half_extent"]
        min_sep = half_ext * 3
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                assert dist > min_sep * 0.9

    def test_get_cube_positions_helper(self, pipeline):
        positions = pipeline.get_cube_positions()
        assert len(positions) == 5
        for name, pos in positions.items():
            assert pos.shape == (3,)
