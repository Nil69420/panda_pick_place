import math

import numpy as np
import pytest

from panda_control.camera import CameraHandler


class TestCameraIntrinsics:

    def test_intrinsics_shape(self, pipeline):
        intrinsics = pipeline.camera.compute_intrinsics()
        assert intrinsics.K.shape == (3, 3)
        assert intrinsics.K_inv.shape == (3, 3)

    def test_intrinsics_values(self, pipeline):
        intrinsics = pipeline.camera.compute_intrinsics()
        fov_rad = math.radians(intrinsics.fov)
        expected_fy = (intrinsics.height / 2.0) / math.tan(fov_rad / 2.0)
        assert np.isclose(intrinsics.K[1, 1], expected_fy)
        assert np.isclose(intrinsics.K[0, 0], expected_fy)
        assert np.isclose(intrinsics.K[0, 2], intrinsics.width / 2.0)
        assert np.isclose(intrinsics.K[1, 2], intrinsics.height / 2.0)

    def test_K_inv_is_inverse(self, pipeline):
        intrinsics = pipeline.camera.compute_intrinsics()
        identity = intrinsics.K @ intrinsics.K_inv
        assert np.allclose(identity, np.eye(3), atol=1e-10)


class TestOverheadCamera:

    def test_overhead_rgb_shape(self, pipeline):
        frame = pipeline.snapshot_overhead()
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.rgb.dtype == np.uint8

    def test_overhead_depth_shape(self, pipeline):
        frame = pipeline.snapshot_overhead()
        assert frame.depth.shape == (480, 640)
        assert frame.depth.dtype == np.float32

    def test_overhead_depth_positive(self, pipeline):
        frame = pipeline.snapshot_overhead()
        assert np.all(frame.depth > 0)

    def test_overhead_segmentation_shape(self, pipeline):
        frame = pipeline.snapshot_overhead()
        assert frame.segmentation.shape == (480, 640)
        assert frame.segmentation.dtype == np.int32

    def test_overhead_extrinsics_shapes(self, pipeline):
        frame = pipeline.snapshot_overhead()
        assert frame.extrinsics.T_cam_to_world.shape == (4, 4)
        assert frame.extrinsics.T_world_to_cam.shape == (4, 4)

    def test_overhead_extrinsics_invertible(self, pipeline):
        frame = pipeline.snapshot_overhead()
        product = frame.extrinsics.T_cam_to_world @ frame.extrinsics.T_world_to_cam
        assert np.allclose(product, np.eye(4), atol=1e-6)


class TestWristCamera:

    def test_wrist_rgb_shape(self, pipeline):
        frame = pipeline.snapshot_wrist()
        assert frame.rgb.shape == (480, 640, 3)
        assert frame.rgb.dtype == np.uint8

    def test_wrist_depth_shape(self, pipeline):
        frame = pipeline.snapshot_wrist()
        assert frame.depth.shape == (480, 640)
        assert frame.depth.dtype == np.float32

    def test_wrist_depth_positive(self, pipeline):
        frame = pipeline.snapshot_wrist()
        assert np.all(frame.depth > 0)

    def test_wrist_extrinsics_invertible(self, pipeline):
        frame = pipeline.snapshot_wrist()
        product = frame.extrinsics.T_cam_to_world @ frame.extrinsics.T_world_to_cam
        assert np.allclose(product, np.eye(4), atol=1e-6)


class TestBackProjection:

    def test_centre_pixel_returns_3d(self, pipeline):
        frame = pipeline.snapshot_overhead()
        cu = frame.intrinsics.width // 2
        cv = frame.intrinsics.height // 2
        world_pt = pipeline.pixel_to_world(cu, cv, frame)
        assert world_pt.shape == (3,)
        assert np.all(np.isfinite(world_pt))

    def test_depth_at_pixel(self, pipeline):
        frame = pipeline.snapshot_overhead()
        d = CameraHandler.depth_at_pixel(frame, 320, 240)
        assert isinstance(d, float)
        assert d > 0

    def test_back_projection_near_table(self, pipeline):
        frame = pipeline.snapshot_overhead()
        cu = frame.intrinsics.width // 2
        cv = frame.intrinsics.height // 2
        world_pt = pipeline.pixel_to_world(cu, cv, frame)
        assert abs(world_pt[2]) < 0.5
