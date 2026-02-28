from __future__ import annotations

from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np

import panda_gym

from panda_control.camera import CameraFrame, CameraHandler
from panda_control.perception import Detection, PerceptionSystem
from panda_control.robot import (
    GraspResult,
    GraspState,
    PickAndPlaceResult,
    PlaceResult,
    RobotController,
)
from panda_control.scene import SceneManager


class TaskRunner:

    def __init__(
        self,
        render_mode: str = "human",
        n_cubes: int = 4,
        seed: Optional[int] = None,
        cam_width: int = 640,
        cam_height: int = 480,
        cam_fov: float = 60.0,
        sim_delay: float = 0.0,
    ) -> None:
        self.env = gym.make("PandaPickAndPlace-v3", render_mode=render_mode)
        self.obs, self.info = self.env.reset(seed=seed)

        self.sim = self.env.unwrapped.sim
        self.physics_client = self.sim.physics_client
        self.robot = self.env.unwrapped.robot
        self.task = self.env.unwrapped.task
        self.robot_body_id: int = self.sim._bodies_idx[self.robot.body_name]

        self.rng = np.random.default_rng(seed)

        self.scene = SceneManager(self.sim, rng=self.rng)
        self.cube_names = self.scene.spawn_random_cubes(n=n_cubes)

        self.camera = CameraHandler(
            self.physics_client,
            width=cam_width,
            height=cam_height,
            fov=cam_fov,
        )

        self.perception = PerceptionSystem(
            camera_handler=self.camera,
            physics_client=self.physics_client,
            robot_body_id=self.robot_body_id,
            wrist_link_index=self.robot.ee_link,
            body_index=dict(self.sim._bodies_idx),
        )

        self.controller = RobotController(
            sim=self.sim,
            robot=self.robot,
            robot_body_id=self.robot_body_id,
            sim_delay=sim_delay,
        )
        self._sim_delay = sim_delay

        # Hide the default 'object' and 'target' bodies that panda-gym
        # creates (green cube + translucent goal ghost).  We cannot remove
        # them because the env's get_obs() queries their state every step,
        # so we just move them far below the table where they are invisible.
        self._hide_default_bodies()

        for _ in range(10):
            self.sim.step()

    @property
    def ee_position(self) -> np.ndarray:
        return self.robot.get_ee_position()

    _HIDDEN_POS = np.array([0.0, 0.0, -2.0])
    _HIDDEN_ORN = np.array([0.0, 0.0, 0.0, 1.0])

    def _hide_default_bodies(self) -> None:
        """Stash the env's built-in 'object' and 'target' under the table."""
        for name in ("object", "target"):
            if name in self.sim._bodies_idx:
                self.sim.set_base_pose(name, self._HIDDEN_POS, self._HIDDEN_ORN)

    def get_cube_positions(self) -> Dict[str, np.ndarray]:
        return {n: info["position"] for n, info in self.scene.get_cube_info().items()}

    def snapshot_overhead(self, **kwargs) -> CameraFrame:
        return self.camera.capture_overhead(**kwargs)

    def snapshot_wrist(self) -> CameraFrame:
        return self.camera.capture_wrist(
            robot_body_id=self.robot_body_id,
            wrist_link_index=self.robot.ee_link,
        )

    def pixel_to_world(self, u, v, frame: CameraFrame) -> np.ndarray:
        d = CameraHandler.depth_at_pixel(frame, u, v)
        return CameraHandler.pixel_to_world(u, v, d, frame.intrinsics, frame.extrinsics)

    def get_body_id_map(self) -> Dict[str, int]:
        return {name: self.sim._bodies_idx[name] for name in self.cube_names
                if name in self.sim._bodies_idx}

    def _obstacle_positions(self, exclude_name: Optional[str] = None) -> List[np.ndarray]:
        """Return live 3-D positions of all cubes except *exclude_name*."""
        positions: List[np.ndarray] = []
        for name in self.cube_names:
            if name == exclude_name:
                continue
            try:
                positions.append(np.array(self.sim.get_base_position(name)))
            except Exception:
                pass
        return positions

    def detect_and_grasp(self, cube_name: Optional[str] = None) -> GraspResult:
        self.controller.retract_for_sensing()
        detections = self.perception.perceive_overhead()
        if not detections:
            return GraspResult(
                success=False,
                final_ee_position=self.ee_position,
                final_object_position=None,
                states_visited=[GraspState.FAILED],
            )
        body_id_map = self.get_body_id_map()
        inv_map = {v: k for k, v in body_id_map.items()}

        if cube_name and cube_name in body_id_map:
            target_id = body_id_map[cube_name]
            det = next((d for d in detections if d.body_id == target_id), None)
            if det is None:
                det = detections[0]
                cube_name = inv_map.get(det.body_id, cube_name)
        else:
            det = detections[0]
            cube_name = inv_map.get(det.body_id)

        return self.controller.grasp_point_world(
            det.world_position,
            object_name=cube_name,
            obstacle_positions=self._obstacle_positions(exclude_name=cube_name),
        )

    def reset(self, seed: Optional[int] = None) -> None:
        self.scene.remove_all_cubes()
        self.obs, self.info = self.env.reset(seed=seed)
        self.cube_names = self.scene.spawn_random_cubes(n=len(self.cube_names))
        self.perception = PerceptionSystem(
            camera_handler=self.camera,
            physics_client=self.physics_client,
            robot_body_id=self.robot_body_id,
            wrist_link_index=self.robot.ee_link,
            body_index=dict(self.sim._bodies_idx),
        )
        self.controller = RobotController(
            sim=self.sim,
            robot=self.robot,
            robot_body_id=self.robot_body_id,
            sim_delay=self._sim_delay,
        )
        self._hide_default_bodies()
        for _ in range(10):
            self.sim.step()

    def pick_and_place_one(
        self,
        cube_name: Optional[str] = None,
        place_position: Optional[np.ndarray] = None,
        cube_half_extent: float = 0.02,
    ) -> PickAndPlaceResult:
        """Pick one cube and place it at *place_position*."""
        self.controller.retract_for_sensing()
        detections = self.perception.perceive_overhead()
        if not detections:
            return PickAndPlaceResult(
                pick_success=False,
                place_success=False,
                pick_result=None,
                place_result=None,
                object_name=cube_name,
            )

        body_id_map = self.get_body_id_map()
        inv_map = {v: k for k, v in body_id_map.items()}

        if cube_name and cube_name in body_id_map:
            target_id = body_id_map[cube_name]
            det = next((d for d in detections if d.body_id == target_id), None)
            if det is None:
                det = detections[0]
                cube_name = inv_map.get(det.body_id, cube_name)
        else:
            det = min(detections, key=lambda d: np.linalg.norm(d.world_position[:2]))
            cube_name = inv_map.get(det.body_id)

        if place_position is None:
            place_position = np.array([0.0, 0.15, 0.0])

        place_target = place_position.copy()
        place_target[2] = cube_half_extent

        return self.controller.pick_and_place(
            pick_xyz=det.world_position,
            place_xyz=place_target,
            object_name=cube_name,
            place_height=cube_half_extent,
            obstacle_positions=self._obstacle_positions(exclude_name=cube_name),
        )

    def pick_and_place_all(
        self,
        place_position: Optional[np.ndarray] = None,
        cube_half_extent: float = 0.02,
        spacing: float = 0.12,
    ) -> List[PickAndPlaceResult]:
        """Pick every visible cube and place it at its optimally-assigned slot.

        Placement slots form a centred row along the Y-axis (slot 0 …
        n−1).  Before any picking starts the method detects all cubes,
        builds an XY-distance cost matrix, and solves the optimal
        one-to-one cube→slot assignment via the Hungarian algorithm
        (``scipy.optimize.linear_sum_assignment``).  Cubes are then
        executed in slot order (lowest slot index first) so earlier
        placements never block later ones.
        """
        from scipy.optimize import linear_sum_assignment

        results: List[PickAndPlaceResult] = []
        placed_ids: set = set()

        if place_position is None:
            place_position = np.array([0.0, 0.0, 0.0])

        # Pre-compute centred placement slots along the Y-axis
        n_slots = len(self.cube_names)
        y_start = place_position[1] - (n_slots - 1) / 2.0 * spacing
        slots = [
            np.array([place_position[0], y_start + i * spacing, cube_half_extent])
            for i in range(n_slots)
        ]

        # --- Mark slots already occupied by *any* cube --------------------
        # First, retract the arm so the overhead camera has a clear view.
        self.controller.retract_for_sensing()
        # Query PyBullet for the true positions of all cubes so that
        # undetected cubes sitting on a slot are not overwritten.
        occupy_thresh = spacing / 2.0  # a cube within half-a-slot is "there"
        occupied_slots: set = set()     # slot indices that already hold a cube
        already_placed_ids: set = set() # body ids of cubes already at a slot
        for name in self.cube_names:
            try:
                pos = np.array(self.sim.get_base_position(name))
            except Exception:
                continue
            for j, slot in enumerate(slots):
                if np.linalg.norm(pos[:2] - slot[:2]) < occupy_thresh:
                    occupied_slots.add(j)
                    bid = self.sim._bodies_idx.get(name)
                    if bid is not None:
                        already_placed_ids.add(bid)
                    break

        # --- Initial detection + optimal assignment -----------------------
        detections = self.perception.perceive_overhead()
        # Exclude cubes that are already sitting at a slot
        detections = [d for d in detections if d.body_id not in already_placed_ids]
        # Exclude cubes that are out of the robot's reliable reach
        max_reach = 0.20  # metres from base in XY
        detections = [
            d for d in detections
            if np.linalg.norm(d.world_position[:2]) <= max_reach
        ]
        if not detections:
            return results

        body_id_map = self.get_body_id_map()
        inv_map = {v: k for k, v in body_id_map.items()}

        # Available (unoccupied) slot indices
        free_slots = [j for j in range(n_slots) if j not in occupied_slots]
        if not free_slots:
            return results

        # Cost matrix: rows = detections, cols = free slots
        cost = np.zeros((len(detections), len(free_slots)))
        for i, det in enumerate(detections):
            for jj, slot_idx in enumerate(free_slots):
                cost[i, jj] = np.linalg.norm(det.world_position[:2] - slots[slot_idx][:2])

        row_ind, col_ind = linear_sum_assignment(cost)

        # Build ordered plan sorted by real slot index
        plan = sorted(
            [(row_ind[k], free_slots[col_ind[k]]) for k in range(len(row_ind))],
            key=lambda rc: rc[1],
        )

        # --- Execute plan in slot order -----------------------------------
        for det_idx, slot_idx in plan:
            # Re-perceive to get fresh position (cube may have shifted)
            self.controller.retract_for_sensing()
            fresh = self.perception.perceive_overhead()
            fresh = [d for d in fresh if d.body_id not in placed_ids]

            # Match original detection by body_id
            orig = detections[det_idx]
            det = next((d for d in fresh if d.body_id == orig.body_id), None)
            if det is None:
                continue  # cube lost – skip

            # Live re-check: abort if this slot is now occupied (a cube
            # may have been nudged onto it during an earlier cycle).
            slot_pos = slots[slot_idx]
            slot_blocked = False
            for cname in self.cube_names:
                bid = self.sim._bodies_idx.get(cname)
                if bid is not None and bid != det.body_id and bid not in placed_ids:
                    try:
                        cpos = np.array(self.sim.get_base_position(cname))
                        if np.linalg.norm(cpos[:2] - slot_pos[:2]) < occupy_thresh:
                            slot_blocked = True
                            break
                    except Exception:
                        pass
            if slot_blocked:
                continue  # skip this slot – something is in the way

            obj_name = inv_map.get(det.body_id)

            res = self.controller.pick_and_place(
                pick_xyz=det.world_position,
                place_xyz=slot_pos,
                object_name=obj_name,
                place_height=cube_half_extent,
                obstacle_positions=self._obstacle_positions(exclude_name=obj_name),
            )
            res.slot_index = slot_idx  # 0-based; display as slot_idx+1
            results.append(res)
            placed_ids.add(det.body_id)

            # Brief settle
            for _ in range(20):
                self.sim.step()

        return results

    def stack_all(
        self,
        stack_position: Optional[np.ndarray] = None,
        cube_half_extent: float = 0.02,
    ) -> List[PickAndPlaceResult]:
        results: List[PickAndPlaceResult] = []
        stack_height = 0

        if stack_position is None:
            stack_position = np.array([0.0, 0.0, 0.0])

        while True:
            detections = self.perception.perceive_overhead()
            if len(detections) <= 1 and stack_height > 0:
                break
            if not detections:
                break

            body_id_map = self.get_body_id_map()
            inv_map = {v: k for k, v in body_id_map.items()}

            if stack_height == 0:
                base_det = min(
                    detections,
                    key=lambda d: np.linalg.norm(d.world_position[:2] - stack_position[:2]),
                )
                stack_position = base_det.world_position.copy()
                stack_position[2] = cube_half_extent
                stack_height = 1

                detections = [d for d in detections if d.body_id != base_det.body_id]
                if not detections:
                    break

            det = min(
                detections,
                key=lambda d: np.linalg.norm(d.world_position[:2]),
            )
            obj_name = inv_map.get(det.body_id)

            place_z = cube_half_extent * 2 * stack_height
            place_target = stack_position.copy()
            place_target[2] = place_z + cube_half_extent

            res = self.controller.pick_and_place(
                pick_xyz=det.world_position,
                place_xyz=place_target,
                object_name=obj_name,
                place_height=cube_half_extent,
                obstacle_positions=self._obstacle_positions(exclude_name=obj_name),
            )
            results.append(res)

            if res.place_success:
                stack_height += 1

            for _ in range(60):
                self.sim.step()

        return results

    def close(self) -> None:
        self.env.close()
