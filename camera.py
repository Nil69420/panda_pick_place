"""RGB-D camera abstraction for overhead and wrist-mounted views.

This module wraps PyBullet's ``getCameraImage`` call into a clean
dataclass-based API.  Two convenience methods are provided:

* :meth:`CameraHandler.capture_overhead` -- bird's-eye view of the
  table, used by the perception system.
* :meth:`CameraHandler.capture_wrist` -- camera rigidly attached to
  the robot's wrist link, useful for close-range inspection.

All tuneable defaults (resolution, field of view, near/far planes,
overhead yaw/pitch/roll, wrist offset) are read from the project
configuration file so they can be changed in one place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pybullet as p

from panda_control.config import get as cfg
from panda_control.common_utils.jit_kernels import linearise_depth_buf


@dataclass
class CameraIntrinsics:
    """Pin-hole intrinsic parameters and their inverse."""

    width: int
    height: int
    fov: float
    near: float
    far: float
    K: np.ndarray
    K_inv: np.ndarray


@dataclass
class CameraExtrinsics:
    """View-matrix decomposition into camera-to-world and inverse."""

    view_matrix_flat: List[float]
    T_cam_to_world: np.ndarray
    T_world_to_cam: np.ndarray


@dataclass
class CameraFrame:
    """Single snapshot: RGB image, metric depth map, and segmentation."""

    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


class CameraHandler:
    """Create projection matrices and capture RGB-D frames.

    Parameters
    ----------
    physics_client
        PyBullet physics client returned by ``pybullet.connect()``.
    width, height : int
        Image resolution in pixels.
    fov : float
        Vertical field of view in degrees.
    near, far : float
        Near and far clipping planes (metres).
    """

    def __init__(
        self,
        physics_client,
        *,
        width: int = None,
        height: int = None,
        fov: float = None,
        near: float = None,
        far: float = None,
    ) -> None:
        self._p = physics_client
        self.width = width if width is not None else cfg("camera", "width")
        self.height = height if height is not None else cfg("camera", "height")
        self.fov = fov if fov is not None else cfg("camera", "fov")
        self.near = near if near is not None else cfg("camera", "near_plane")
        self.far = far if far is not None else cfg("camera", "far_plane")

        self.proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
        )

    def compute_intrinsics(self) -> CameraIntrinsics:
        """Build a :class:`CameraIntrinsics` from the current FOV and resolution."""
        fov_rad = math.radians(self.fov)
        fy = (self.height / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy
        cx = self.width / 2.0
        cy = self.height / 2.0

        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        return CameraIntrinsics(
            width=self.width,
            height=self.height,
            fov=self.fov,
            near=self.near,
            far=self.far,
            K=K,
            K_inv=np.linalg.inv(K),
        )

    @staticmethod
    def _view_matrix_to_extrinsics(view_matrix_flat) -> CameraExtrinsics:
        """Decompose a flat 16-element view matrix into world/camera transforms."""
        V = np.array(view_matrix_flat, dtype=np.float64).reshape(4, 4, order="F")
        T_world_to_cam = V
        T_cam_to_world = np.linalg.inv(V)
        return CameraExtrinsics(
            view_matrix_flat=list(view_matrix_flat),
            T_cam_to_world=T_cam_to_world,
            T_world_to_cam=T_world_to_cam,
        )

    def linearise_depth(self, depth_buffer: np.ndarray) -> np.ndarray:
        """Convert a normalised depth buffer to metric depth (metres).

        Delegates to a JIT-compiled kernel when Numba is available,
        avoiding intermediate-array allocations for large frames.
        """
        return linearise_depth_buf(
            np.ascontiguousarray(depth_buffer, dtype=np.float32),
            self.near,
            self.far,
            self.height,
            self.width,
        )

    def _capture(self, view_matrix_flat) -> CameraFrame:
        """Render one frame and package it as a :class:`CameraFrame`."""
        _, _, rgba, depth_buf, seg = self._p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix_flat,
            projectionMatrix=self.proj_matrix,
            shadow=True,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
            if self._p.getConnectionInfo()["connectionMethod"] == p.GUI
            else p.ER_TINY_RENDERER,
        )
        rgba = np.array(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        rgb = rgba[:, :, :3]
        depth_buf = np.array(depth_buf, dtype=np.float32).reshape(self.height, self.width)
        seg = np.array(seg, dtype=np.int32).reshape(self.height, self.width)

        intrinsics = self.compute_intrinsics()
        extrinsics = self._view_matrix_to_extrinsics(view_matrix_flat)
        depth_metric = self.linearise_depth(depth_buf)

        return CameraFrame(
            rgb=rgb,
            depth=depth_metric,
            segmentation=seg,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    def capture_overhead(
        self,
        target: np.ndarray = np.array([0.0, 0.0, 0.0]),
        distance: float = None,
        yaw: float = None,
        pitch: float = None,
        roll: float = None,
    ) -> CameraFrame:
        """Capture a bird's-eye view of the workspace.

        All camera-pose parameters default to the values stored in
        ``config/default.yaml`` under ``camera.overhead``.
        """
        if distance is None:
            distance = cfg("camera", "overhead", "distance")
        if yaw is None:
            yaw = cfg("camera", "overhead", "yaw")
        if pitch is None:
            pitch = cfg("camera", "overhead", "pitch")
        if roll is None:
            roll = cfg("camera", "overhead", "roll")
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=target.tolist(),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )
        return self._capture(view_matrix)

    def capture_wrist(
        self,
        robot_body_id: int,
        wrist_link_index: int = None,
        offset_distance: float = None,
    ) -> CameraFrame:
        """Capture a frame from the wrist-mounted camera.

        The camera is positioned a short distance behind the wrist
        link, looking along its local Z-axis.
        """
        if wrist_link_index is None:
            wrist_link_index = cfg("camera", "wrist", "link_index")
        if offset_distance is None:
            offset_distance = cfg("camera", "wrist", "offset_distance")
        link_state = self._p.getLinkState(robot_body_id, wrist_link_index)
        link_pos = np.array(link_state[0])
        link_orn = np.array(link_state[1])

        rot_mat = np.array(
            self._p.getMatrixFromQuaternion(link_orn), dtype=np.float64
        ).reshape(3, 3)

        cam_forward = -rot_mat[:, 2]
        cam_up = rot_mat[:, 1]

        cam_pos = link_pos - cam_forward * offset_distance
        target_pos = cam_pos + cam_forward

        view_matrix = self._p.computeViewMatrix(
            cameraEyePosition=cam_pos.tolist(),
            cameraTargetPosition=target_pos.tolist(),
            cameraUpVector=cam_up.tolist(),
        )
        return self._capture(view_matrix)

    @staticmethod
    def pixel_to_world(
        u: float,
        v: float,
        depth: float,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ) -> np.ndarray:
        """Back-project a pixel at a known depth to a 3-D world point."""
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        p_cv = intrinsics.K_inv @ pixel_h * depth
        p_gl = np.array([p_cv[0], -p_cv[1], -p_cv[2]])
        p_world_h = extrinsics.T_cam_to_world @ np.append(p_gl, 1.0)
        return p_world_h[:3]

    @staticmethod
    def depth_at_pixel(frame: CameraFrame, u: int, v: int) -> float:
        """Read the metric depth at pixel ``(u, v)`` from *frame*."""
        return float(frame.depth[int(v), int(u)])
