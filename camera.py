from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pybullet as p


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fov: float
    near: float
    far: float
    K: np.ndarray
    K_inv: np.ndarray


@dataclass
class CameraExtrinsics:
    view_matrix_flat: List[float]
    T_cam_to_world: np.ndarray
    T_world_to_cam: np.ndarray


@dataclass
class CameraFrame:
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


class CameraHandler:

    def __init__(
        self,
        physics_client,
        *,
        width: int = 640,
        height: int = 480,
        fov: float = 60.0,
        near: float = 0.02,
        far: float = 5.0,
    ) -> None:
        self._p = physics_client
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far

        self.proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far,
        )

    def compute_intrinsics(self) -> CameraIntrinsics:
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
        V = np.array(view_matrix_flat, dtype=np.float64).reshape(4, 4, order="F")
        T_world_to_cam = V
        T_cam_to_world = np.linalg.inv(V)
        return CameraExtrinsics(
            view_matrix_flat=list(view_matrix_flat),
            T_cam_to_world=T_cam_to_world,
            T_world_to_cam=T_world_to_cam,
        )

    def linearise_depth(self, depth_buffer: np.ndarray) -> np.ndarray:
        d = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer)
        return d.astype(np.float32)

    def _capture(self, view_matrix_flat) -> CameraFrame:
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
        distance: float = 1.0,
        yaw: float = 0.0,
        pitch: float = -89.9,
        roll: float = 0.0,
    ) -> CameraFrame:
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
        wrist_link_index: int = 11,
        offset_distance: float = 0.05,
    ) -> CameraFrame:
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
        pixel_h = np.array([u, v, 1.0], dtype=np.float64)
        p_cv = intrinsics.K_inv @ pixel_h * depth
        p_gl = np.array([p_cv[0], -p_cv[1], -p_cv[2]])
        p_world_h = extrinsics.T_cam_to_world @ np.append(p_gl, 1.0)
        return p_world_h[:3]

    @staticmethod
    def depth_at_pixel(frame: CameraFrame, u: int, v: int) -> float:
        return float(frame.depth[int(v), int(u)])
