import numpy as np
import pytest


class TestPerceptionOverhead:

    def test_detects_all_cubes(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        assert len(detections) == 5

    def test_no_static_bodies_detected(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        body_id_map = pipeline.get_body_id_map()
        cube_ids = set(body_id_map.values())
        for det in detections:
            assert det.body_id in cube_ids

    def test_detection_fields(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        for det in detections:
            assert isinstance(det.body_id, int)
            assert det.pixel_centroid.shape == (2,)
            assert det.world_position.shape == (3,)
            assert det.mean_color.shape == (3,)
            assert det.pixel_count > 0
            assert det.bbox.shape == (4,)

    def test_world_positions_on_table(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        for det in detections:
            assert -0.5 < det.world_position[0] < 0.5
            assert -0.5 < det.world_position[1] < 0.5
            assert det.world_position[2] >= 0.0
            assert det.world_position[2] < 0.2

    def test_bounding_boxes_valid(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        for det in detections:
            u_min, v_min, u_max, v_max = det.bbox
            assert u_min >= 0
            assert v_min >= 0
            assert u_max < 640
            assert v_max < 480
            assert u_max >= u_min
            assert v_max >= v_min

    def test_mean_colors_in_range(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        for det in detections:
            assert np.all(det.mean_color >= 0.0)
            assert np.all(det.mean_color <= 1.0)


class TestPerceptionAccuracy:

    def test_back_projection_error_below_threshold(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        gt = {n: pipeline.sim.get_base_position(n) for n in pipeline.cube_names}
        bmap = pipeline.get_body_id_map()
        errors = pipeline.perception.validate_against_ground_truth(detections, gt, bmap)
        for name, err in errors.items():
            assert err < 0.05, f"{name} error {err:.4f}m exceeds 50mm"

    def test_mean_error_below_threshold(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        gt = {n: pipeline.sim.get_base_position(n) for n in pipeline.cube_names}
        bmap = pipeline.get_body_id_map()
        errors = pipeline.perception.validate_against_ground_truth(detections, gt, bmap)
        mean_err = np.mean(list(errors.values()))
        assert mean_err < 0.03, f"Mean error {mean_err:.4f}m exceeds 30mm"

    def test_xy_accuracy(self, pipeline):
        detections = pipeline.perception.perceive_overhead()
        gt = {n: pipeline.sim.get_base_position(n) for n in pipeline.cube_names}
        bmap = pipeline.get_body_id_map()
        det_by_id = {d.body_id: d for d in detections}
        for name, gt_pos in gt.items():
            bid = bmap[name]
            det_pos = det_by_id[bid].world_position
            xy_err = np.linalg.norm(det_pos[:2] - gt_pos[:2])
            assert xy_err < 0.01, f"{name} XY error {xy_err:.4f}m exceeds 10mm"


class TestLocateAll:

    def test_locate_all_returns_all_cubes(self, pipeline):
        world_map = pipeline.perception.locate_all()
        assert len(world_map) == 5

    def test_locate_all_values_are_3d(self, pipeline):
        world_map = pipeline.perception.locate_all()
        for body_id, pos in world_map.items():
            assert isinstance(body_id, int)
            assert pos.shape == (3,)


class TestPerceptionWrist:

    def test_wrist_returns_list(self, pipeline):
        detections = pipeline.perception.perceive_wrist()
        assert isinstance(detections, list)

    def test_wrist_detections_have_valid_fields(self, pipeline):
        detections = pipeline.perception.perceive_wrist()
        for det in detections:
            assert det.world_position.shape == (3,)
            assert np.all(np.isfinite(det.world_position))
