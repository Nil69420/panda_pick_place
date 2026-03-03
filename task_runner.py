"""High-level task orchestrator for vision-guided pick-and-place.

:class:`TaskRunner` ties together every subsystem -- simulation,
perception, robot control, and scene management -- into a single
facade that higher-level scripts (``main.py``) can drive with
one-line calls such as :meth:`pick_and_place_all` or
:meth:`stack_all`.

Typical usage::

    runner = TaskRunner(render_mode="human", n_cubes=4, seed=42)
    results = runner.pick_and_place_all()
    runner.close()
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

import panda_gym

from panda_control.camera import CameraFrame, CameraHandler
from panda_control.config import get as cfg
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
    """Facade that owns the simulation and exposes task-level primitives.

    Parameters
    ----------
    render_mode : str
        ``"human"`` for a GUI window, ``"rgb_array"`` for headless.
    n_cubes : int
        Number of cubes to spawn on the table.
    seed : int, optional
        Seed for reproducibility (``None`` for random).
    cam_width, cam_height, cam_fov : int / float
        Override the camera resolution or field of view.
    sim_delay : float
        Per-step sleep for slowing down the visualisation.
    """

    def __init__(
        self,
        render_mode: str = "human",
        n_cubes: int = None,
        seed: Optional[int] = None,
        cam_width: int = None,
        cam_height: int = None,
        cam_fov: float = None,
        sim_delay: float = 0.0,
    ) -> None:
        if n_cubes is None:
            n_cubes = cfg("task", "default_n_cubes")
        if cam_width is None:
            cam_width = cfg("camera", "width")
        if cam_height is None:
            cam_height = cfg("camera", "height")
        if cam_fov is None:
            cam_fov = cfg("camera", "fov")

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
        # creates (green cube and translucent goal ghost).  They cannot
        # be removed because the environment's get_obs() queries their
        # state every step, so they are moved below the table instead.
        self._hide_default_bodies()

        settle = cfg("task", "sim_settle_steps")
        for _ in range(settle):
            self.sim.step()

    # ── properties ──────────────────────────────────────────

    @property
    def ee_position(self) -> np.ndarray:
        """Current 3-D end-effector position in world coordinates."""
        return self.robot.get_ee_position()

    # ── internal helpers ────────────────────────────────────

    def _hide_default_bodies(self) -> None:
        """Stash the environment's built-in bodies under the table."""
        pos = np.array(cfg("task", "hidden_position"), dtype=np.float64)
        orn = np.array(cfg("task", "hidden_orientation"), dtype=np.float64)
        for name in ("object", "target"):
            if name in self.sim._bodies_idx:
                self.sim.set_base_pose(name, pos, orn)

    def _obstacle_positions(
        self, exclude_name: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Return live 3-D positions of all cubes except *exclude_name*.

        Used to populate the APF obstacle list so the gripper steers
        around cubes it is not currently targeting.
        """
        positions: List[np.ndarray] = []
        for name in self.cube_names:
            if name == exclude_name:
                continue
            try:
                positions.append(np.array(self.sim.get_base_position(name)))
            except Exception:
                pass
        return positions

    def _resolve_detection(
        self,
        detections: List[Detection],
        cube_name: Optional[str],
        body_id_map: Dict[str, int],
        inv_map: Dict[int, str],
        pick_nearest: bool = False,
    ) -> Tuple[Optional[Detection], Optional[str]]:
        """Match a cube name to a detection, or pick the best one.

        Parameters
        ----------
        detections : list[Detection]
            Current overhead detections.
        cube_name : str or None
            Requested cube.  ``None`` means pick automatically.
        body_id_map : dict
            ``{cube_name: body_id}``.
        inv_map : dict
            ``{body_id: cube_name}``.
        pick_nearest : bool
            If ``True`` and no *cube_name* is given, choose the cube
            closest to the table centre rather than the first detection.

        Returns
        -------
        (detection, resolved_name)
            The matched detection and its cube name.
        """
        if not detections:
            return None, cube_name

        if cube_name and cube_name in body_id_map:
            target_id = body_id_map[cube_name]
            det = next((d for d in detections if d.body_id == target_id), None)
            if det is None:
                det = detections[0]
                cube_name = inv_map.get(det.body_id, cube_name)
        else:
            if pick_nearest:
                det = min(detections, key=lambda d: np.linalg.norm(d.world_position[:2]))
            else:
                det = detections[0]
            cube_name = inv_map.get(det.body_id)

        return det, cube_name

    # ── sensor helpers ──────────────────────────────────────

    def get_cube_positions(self) -> Dict[str, np.ndarray]:
        """Return ``{name: position}`` for every spawned cube."""
        return {n: info["position"] for n, info in self.scene.get_cube_info().items()}

    def snapshot_overhead(self, **kwargs) -> CameraFrame:
        """Capture an overhead camera frame."""
        return self.camera.capture_overhead(**kwargs)

    def snapshot_wrist(self) -> CameraFrame:
        """Capture a wrist-mounted camera frame."""
        return self.camera.capture_wrist(
            robot_body_id=self.robot_body_id,
            wrist_link_index=self.robot.ee_link,
        )

    def pixel_to_world(self, u, v, frame: CameraFrame) -> np.ndarray:
        """Back-project a pixel to a 3-D world coordinate."""
        d = CameraHandler.depth_at_pixel(frame, u, v)
        return CameraHandler.pixel_to_world(u, v, d, frame.intrinsics, frame.extrinsics)

    def get_body_id_map(self) -> Dict[str, int]:
        """Return ``{cube_name: body_id}`` for all spawned cubes."""
        return {name: self.sim._bodies_idx[name] for name in self.cube_names
                if name in self.sim._bodies_idx}

    # ── single-cube tasks ───────────────────────────────────

    def detect_and_grasp(self, cube_name: Optional[str] = None) -> GraspResult:
        """Perceive the scene and grasp one cube.

        The arm retracts for sensing, an overhead snapshot is taken,
        and the closest matching detection is used as the grasp target.
        """
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

        det, cube_name = self._resolve_detection(
            detections, cube_name, body_id_map, inv_map,
        )

        return self.controller.grasp_point_world(
            det.world_position,
            object_name=cube_name,
            obstacle_positions=self._obstacle_positions(exclude_name=cube_name),
        )

    def pick_and_place_one(
        self,
        cube_name: Optional[str] = None,
        place_position: Optional[np.ndarray] = None,
        cube_half_extent: float = None,
    ) -> PickAndPlaceResult:
        """Pick one cube and place it at *place_position*.

        If no position is specified the default from the configuration
        file is used.
        """
        if cube_half_extent is None:
            cube_half_extent = cfg("scene", "cube", "half_extent")

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

        det, cube_name = self._resolve_detection(
            detections, cube_name, body_id_map, inv_map,
            pick_nearest=True,
        )

        if place_position is None:
            place_position = np.array(
                cfg("task", "pick_and_place_one", "default_place_position"),
                dtype=np.float64,
            )

        place_target = place_position.copy()
        place_target[2] = cube_half_extent

        return self.controller.pick_and_place(
            pick_xyz=det.world_position,
            place_xyz=place_target,
            object_name=cube_name,
            place_height=cube_half_extent,
            obstacle_positions=self._obstacle_positions(exclude_name=cube_name),
        )

    # ── pick-and-place all (multi-cube) ─────────────────────
    #
    # The logic is broken into four focused helpers so that each
    # stage can be understood, tested, and maintained independently.

    def _compute_slots(
        self,
        place_origin: np.ndarray,
        n_slots: int,
        spacing: float,
        cube_half_extent: float,
    ) -> List[np.ndarray]:
        """Return a centred row of placement slots along the Y-axis.

        Parameters
        ----------
        place_origin : ndarray
            Centre point of the slot row in XY.
        n_slots : int
            How many slots to generate.
        spacing : float
            Distance between adjacent slot centres.
        cube_half_extent : float
            Half-width of a cube -- sets the Z of each slot.

        Returns
        -------
        list[ndarray]
            One 3-D position per slot.
        """
        y_start = place_origin[1] - (n_slots - 1) / 2.0 * spacing
        return [
            np.array([place_origin[0], y_start + i * spacing, cube_half_extent])
            for i in range(n_slots)
        ]

    def _detect_occupied_slots(
        self,
        slots: List[np.ndarray],
        occupy_thresh: float,
    ) -> Tuple[set, set]:
        """Find which slots already hold a cube.

        Returns
        -------
        (occupied_slot_indices, already_placed_body_ids)
        """
        occupied: set = set()
        placed_ids: set = set()
        for name in self.cube_names:
            try:
                pos = np.array(self.sim.get_base_position(name))
            except Exception:
                continue
            for j, slot in enumerate(slots):
                if np.linalg.norm(pos[:2] - slot[:2]) < occupy_thresh:
                    occupied.add(j)
                    bid = self.sim._bodies_idx.get(name)
                    if bid is not None:
                        placed_ids.add(bid)
                    break
        return occupied, placed_ids

    def _assign_cubes_to_slots(
        self,
        detections: List[Detection],
        slots: List[np.ndarray],
        free_slot_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """Solve the optimal cube-to-slot assignment.

        Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
        to minimise total XY travel distance.

        Returns
        -------
        list[tuple[int, int]]
            ``(detection_index, slot_index)`` pairs sorted by slot
            index so that earlier slots are filled first.
        """
        from scipy.optimize import linear_sum_assignment

        cost = np.zeros((len(detections), len(free_slot_indices)))
        for i, det in enumerate(detections):
            for jj, slot_idx in enumerate(free_slot_indices):
                cost[i, jj] = np.linalg.norm(
                    det.world_position[:2] - slots[slot_idx][:2],
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        plan = sorted(
            [(row_ind[k], free_slot_indices[col_ind[k]]) for k in range(len(row_ind))],
            key=lambda rc: rc[1],
        )
        return plan

    def _execute_placement_plan(
        self,
        plan: List[Tuple[int, int]],
        detections: List[Detection],
        slots: List[np.ndarray],
        inv_map: Dict[int, str],
        cube_half_extent: float,
        occupy_thresh: float,
    ) -> List[PickAndPlaceResult]:
        """Execute the ordered placement plan one cube at a time.

        For each ``(detection_index, slot_index)`` pair the method:

        1. Retracts the arm and re-detects to get a fresh position.
        2. Checks that the target slot has not been blocked by an
           earlier placement nudging a cube into it.
        3. Picks and places the cube.

        Returns
        -------
        list[PickAndPlaceResult]
            One result per attempted placement.
        """
        results: List[PickAndPlaceResult] = []
        placed_ids: set = set()
        settle = cfg("task", "pick_and_place_all", "settle_steps")

        for det_idx, slot_idx in plan:
            # Fresh perception each cycle
            self.controller.retract_for_sensing()
            fresh = self.perception.perceive_overhead()
            fresh = [d for d in fresh if d.body_id not in placed_ids]

            orig = detections[det_idx]
            det = next((d for d in fresh if d.body_id == orig.body_id), None)
            if det is None:
                continue

            # Guard: check the slot has not been blocked
            slot_pos = slots[slot_idx]
            blocked = False
            for cname in self.cube_names:
                bid = self.sim._bodies_idx.get(cname)
                if bid is not None and bid != det.body_id and bid not in placed_ids:
                    try:
                        cpos = np.array(self.sim.get_base_position(cname))
                        if np.linalg.norm(cpos[:2] - slot_pos[:2]) < occupy_thresh:
                            blocked = True
                            break
                    except Exception:
                        pass
            if blocked:
                continue

            obj_name = inv_map.get(det.body_id)
            res = self.controller.pick_and_place(
                pick_xyz=det.world_position,
                place_xyz=slot_pos,
                object_name=obj_name,
                place_height=cube_half_extent,
                obstacle_positions=self._obstacle_positions(exclude_name=obj_name),
            )
            res.slot_index = slot_idx
            results.append(res)
            placed_ids.add(det.body_id)

            for _ in range(settle):
                self.sim.step()

        return results

    def pick_and_place_all(
        self,
        place_position: Optional[np.ndarray] = None,
        cube_half_extent: float = None,
        spacing: float = None,
    ) -> List[PickAndPlaceResult]:
        """Pick every visible cube and place it at its assigned slot.

        Placement slots form a centred row along the Y-axis.  Before
        picking starts the method detects all cubes, builds an XY cost
        matrix, and solves the optimal one-to-one assignment with the
        Hungarian algorithm.  Cubes are then placed in slot order so
        that earlier placements never block later ones.
        """
        if cube_half_extent is None:
            cube_half_extent = cfg("scene", "cube", "half_extent")
        if spacing is None:
            spacing = cfg("task", "pick_and_place_all", "spacing")
        if place_position is None:
            place_position = np.array(
                cfg("task", "pick_and_place_all", "default_place_position"),
                dtype=np.float64,
            )

        max_reach = cfg("task", "pick_and_place_all", "max_reach")
        occupy_thresh = spacing / 2.0

        # 1. Compute placement slot positions
        n_slots = len(self.cube_names)
        slots = self._compute_slots(place_position, n_slots, spacing, cube_half_extent)

        # 2. Identify slots already occupied
        self.controller.retract_for_sensing()
        occupied, already_placed_ids = self._detect_occupied_slots(slots, occupy_thresh)

        # 3. Detect and filter cubes
        detections = self.perception.perceive_overhead()
        detections = [d for d in detections if d.body_id not in already_placed_ids]
        detections = [
            d for d in detections
            if np.linalg.norm(d.world_position[:2]) <= max_reach
        ]
        if not detections:
            return []

        body_id_map = self.get_body_id_map()
        inv_map = {v: k for k, v in body_id_map.items()}

        free_slots = [j for j in range(n_slots) if j not in occupied]
        if not free_slots:
            return []

        # 4. Optimal assignment
        plan = self._assign_cubes_to_slots(detections, slots, free_slots)

        # 5. Execute
        return self._execute_placement_plan(
            plan, detections, slots, inv_map,
            cube_half_extent, occupy_thresh,
        )

    # ── stacking ────────────────────────────────────────────

    def stack_all(
        self,
        stack_position: Optional[np.ndarray] = None,
        cube_half_extent: float = None,
    ) -> List[PickAndPlaceResult]:
        """Pick every cube and stack them on top of one another.

        The first cube detected nearest to *stack_position* becomes the
        base of the stack.  Subsequent cubes are placed at incremental
        heights above it.
        """
        if cube_half_extent is None:
            cube_half_extent = cfg("scene", "cube", "half_extent")
        settle = cfg("task", "stack", "settle_steps")

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
                    key=lambda d: np.linalg.norm(
                        d.world_position[:2] - stack_position[:2],
                    ),
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

            for _ in range(settle):
                self.sim.step()

        return results

    # ── lifecycle ───────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> None:
        """Tear down the current scene and rebuild it from scratch."""
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
        settle = cfg("task", "sim_settle_steps")
        for _ in range(settle):
            self.sim.step()

    def close(self) -> None:
        """Shut down the simulation."""
        self.env.close()
