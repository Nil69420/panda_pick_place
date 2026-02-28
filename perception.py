from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from panda_control.camera import CameraFrame, CameraHandler


@dataclass
class Detection:
    body_id: int
    pixel_centroid: np.ndarray
    world_position: np.ndarray
    mean_color: np.ndarray
    pixel_count: int
    bbox: np.ndarray


class PerceptionSystem:

    BACKGROUND_IDS: Set[int] = {-1, 0}
    STATIC_BODY_NAMES: Set[str] = {"panda", "plane", "table", "target", "object"}

    def __init__(
        self,
        camera_handler: CameraHandler,
        physics_client,
        robot_body_id: int,
        wrist_link_index: int,
        body_index: Dict[str, int],
        *,
        ignore_ids: Optional[Set[int]] = None,
        min_pixel_count: int = 10,
    ) -> None:
        self._cam = camera_handler
        self._p = physics_client
        self._robot_id = robot_body_id
        self._wrist_link = wrist_link_index
        self._body_index = body_index
        self._min_px = min_pixel_count

        self._ignore: Set[int] = self._resolve_static_ids()
        if ignore_ids:
            self._ignore |= ignore_ids

    def _resolve_static_ids(self) -> Set[int]:
        ids: Set[int] = set()
        for name, bid in self._body_index.items():
            if name in self.STATIC_BODY_NAMES:
                ids.add(bid)
        return ids

    def _detect(self, frame: CameraFrame) -> List[Detection]:
        seg = frame.segmentation
        unique_ids = set(np.unique(seg)) - self.BACKGROUND_IDS - self._ignore
        detections: List[Detection] = []

        for body_id in sorted(unique_ids):
            mask = seg == body_id
            px_count = int(mask.sum())
            if px_count < self._min_px:
                continue

            vs, us = np.where(mask)

            u_center = float(us.mean())
            v_center = float(vs.mean())

            bbox = np.array([
                int(us.min()), int(vs.min()),
                int(us.max()), int(vs.max()),
            ])

            mean_rgb = frame.rgb[mask].mean(axis=0).astype(np.float32) / 255.0

            depth_vals = frame.depth[mask]
            depth_val = float(np.median(depth_vals))
            world_pos = CameraHandler.pixel_to_world(
                u_center, v_center, depth_val,
                frame.intrinsics, frame.extrinsics,
            )

            detections.append(Detection(
                body_id=int(body_id),
                pixel_centroid=np.array([u_center, v_center]),
                world_position=world_pos,
                mean_color=mean_rgb,
                pixel_count=px_count,
                bbox=bbox,
            ))

        return detections

    def perceive_overhead(self, **cam_kwargs) -> List[Detection]:
        frame = self._cam.capture_overhead(**cam_kwargs)
        return self._detect(frame)

    def perceive_wrist(self, **cam_kwargs) -> List[Detection]:
        frame = self._cam.capture_wrist(
            self._robot_id, self._wrist_link, **cam_kwargs,
        )
        return self._detect(frame)

    def locate_all(self, **cam_kwargs) -> Dict[int, np.ndarray]:
        detections = self.perceive_overhead(**cam_kwargs)
        return {d.body_id: d.world_position for d in detections}

    def validate_against_ground_truth(
        self,
        detections: List[Detection],
        ground_truth: Dict[str, np.ndarray],
        body_id_map: Dict[str, int],
    ) -> Dict[str, float]:
        errors: Dict[str, float] = {}
        det_by_id = {d.body_id: d for d in detections}
        for name, gt_pos in ground_truth.items():
            bid = body_id_map.get(name)
            if bid is None or bid not in det_by_id:
                errors[name] = float("inf")
                continue
            errors[name] = float(np.linalg.norm(det_by_id[bid].world_position - gt_pos))
        return errors
