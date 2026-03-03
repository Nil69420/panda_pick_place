"""Low-level robot controller: IK solving, motion, grasping, and placement.

:class:`RobotController` owns every physical interaction with the
Franka Panda arm.  Higher-level orchestration (which cube to pick,
where to place it) lives in :mod:`panda_control.task_runner`; this
module is concerned only with executing motions and state machines.

Key capabilities:

* **Inverse kinematics** -- :meth:`solve_ik` wraps PyBullet's IK.
* **Point-to-point motion** -- :meth:`move_to` (straight-line) and
  :meth:`move_to_field` (Artificial Potential Field with obstacle
  avoidance).
* **Grasp state machine** -- :meth:`grasp_point_world` drives the
  arm through APPROACH -> DESCEND -> GRASP -> LIFT.
* **Place state machine** -- :meth:`place_at_world` lowers the held
  object and releases it.
* **Combined pick-and-place** -- :meth:`pick_and_place`.

All tuneable constants (gripper widths, heights, APF gains, step
limits) are read from ``config/default.yaml`` at construction time.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from panda_control.config import get as cfg
from panda_control.common_utils.robot_constants import (
    PANDA_LOWER_LIMITS,
    PANDA_UPPER_LIMITS,
    PANDA_JOINT_RANGES,
    PANDA_REST_POSES,
)
from panda_control.common_utils.apf import attractive_force, repulsive_force, total_field_force


# ── state enumerations ──────────────────────────────────────

class GraspState(enum.Enum):
    """Finite-state labels for the grasp state machine."""

    IDLE = "idle"
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    LIFT = "lift"
    DONE = "done"
    FAILED = "failed"


class PlaceState(enum.Enum):
    """Finite-state labels for the place state machine."""

    TRANSIT = "transit"
    LOWER = "lower"
    RELEASE = "release"
    RETRACT = "retract"
    DONE = "done"
    FAILED = "failed"


# ── result dataclasses ──────────────────────────────────────

@dataclass
class GraspResult:
    """Outcome of a single grasp attempt."""

    success: bool
    final_ee_position: np.ndarray
    final_object_position: Optional[np.ndarray]
    states_visited: List[GraspState]


@dataclass
class PlaceResult:
    """Outcome of a single placement attempt."""

    success: bool
    final_ee_position: np.ndarray
    final_object_position: Optional[np.ndarray]
    states_visited: List[PlaceState]


@dataclass
class PickAndPlaceResult:
    """Combined outcome of a pick followed by a place."""

    pick_success: bool
    place_success: bool
    pick_result: Optional[GraspResult]
    place_result: Optional[PlaceResult]
    object_name: Optional[str]
    slot_index: Optional[int] = None


# ── controller ──────────────────────────────────────────────

class RobotController:
    """Joint-level and task-space controller for the Franka Panda.

    Parameters
    ----------
    sim : panda_gym.pybullet.PyBullet
        Simulation wrapper.
    robot : panda_gym.envs.robots.panda.Panda
        Robot instance.
    robot_body_id : int
        PyBullet body index of the robot.
    approach_height, lift_height, descend_offset : float
        Vertical offsets used by the grasp state machine.
    move_tolerance : float
        Cartesian tolerance (metres) for ``move_to`` convergence.
    move_max_steps : int
        Maximum simulation steps per motion primitive.
    grasp_steps : int
        Number of steps to hold the gripper closed.
    sim_delay : float
        Optional per-step sleep for visual slow-motion.
    """

    # Expose joint-limit arrays as class-level attributes for tests
    PANDA_LOWER_LIMITS = PANDA_LOWER_LIMITS
    PANDA_UPPER_LIMITS = PANDA_UPPER_LIMITS
    PANDA_JOINT_RANGES = PANDA_JOINT_RANGES
    PANDA_REST_POSES = PANDA_REST_POSES

    # Gripper width constants (metres)
    GRIPPER_OPEN = cfg("robot", "gripper", "open_width")
    GRIPPER_PREGRASP = cfg("robot", "gripper", "pregrasp_width")
    GRIPPER_CLOSED = cfg("robot", "gripper", "closed_width")

    # Altitude at which lateral motion is collision-free
    SAFE_HEIGHT = cfg("robot", "motion", "safe_height")

    def __init__(
        self,
        sim,
        robot,
        robot_body_id: int,
        *,
        approach_height: float = None,
        lift_height: float = None,
        descend_offset: float = None,
        move_tolerance: float = None,
        move_max_steps: int = None,
        grasp_steps: int = None,
        sim_delay: float = 0.0,
    ) -> None:
        self._sim = sim
        self._robot = robot
        self._body_id = robot_body_id
        self._p = sim.physics_client

        self.approach_height = approach_height if approach_height is not None else cfg("robot", "motion", "approach_height")
        self.lift_height = lift_height if lift_height is not None else cfg("robot", "motion", "lift_height")
        self.descend_offset = descend_offset if descend_offset is not None else cfg("robot", "motion", "descend_offset")
        self.move_tolerance = move_tolerance if move_tolerance is not None else cfg("robot", "motion", "move_tolerance")
        self.move_max_steps = move_max_steps if move_max_steps is not None else cfg("robot", "motion", "move_max_steps")
        self.grasp_steps = grasp_steps if grasp_steps is not None else cfg("robot", "gripper", "grasp_steps")
        self.sim_delay = sim_delay

    def _step(self) -> None:
        self._sim.step()
        if self.sim_delay > 0:
            time.sleep(self.sim_delay)

    # ── Artificial Potential Field tuning constants ────────────────
    #
    # All values are read from config/default.yaml -> robot.apf.
    #   eta       -- repulsive gain
    #   xi        -- attractive gain
    #   d_thresh  -- influence radius of each obstacle (m)
    #   alpha     -- velocity scaling gain
    #   dt        -- pseudo-timestep for position update
    #   perturb   -- random nudge magnitude for local-minimum escape
    #   vel_min   -- velocity floor that triggers the perturbation
    APF_ETA = cfg("robot", "apf", "eta")
    APF_XI = cfg("robot", "apf", "xi")
    APF_D_THRESH = cfg("robot", "apf", "d_thresh")
    APF_ALPHA = cfg("robot", "apf", "alpha")
    APF_DT = cfg("robot", "apf", "dt")
    APF_PERTURB = cfg("robot", "apf", "perturb")
    APF_VEL_MIN = cfg("robot", "apf", "vel_min")

    @staticmethod
    def _attractive_force(
        gripper_pos: np.ndarray,
        goal_pos: np.ndarray,
        xi: float,
    ) -> np.ndarray:
        """Compute the attractive potential-field force toward *goal_pos*."""
        return attractive_force(gripper_pos, goal_pos, xi)

    @staticmethod
    def _repulsive_force(
        gripper_pos: np.ndarray,
        obstacle_pos: np.ndarray,
        eta: float,
        d_thresh: float,
    ) -> np.ndarray:
        """Compute the repulsive potential-field force away from obstacles."""
        return repulsive_force(gripper_pos, obstacle_pos, eta, d_thresh)

    def _total_field_force(
        self,
        gripper_pos: np.ndarray,
        goal_pos: np.ndarray,
        obstacle_positions: Sequence[np.ndarray],
    ) -> np.ndarray:
        return total_field_force(
            gripper_pos, goal_pos, list(obstacle_positions),
            xi=self.APF_XI, eta=self.APF_ETA, d_thresh=self.APF_D_THRESH,
        )

    def move_to_field(
        self,
        target_position: np.ndarray,
        gripper_width: float,
        obstacle_positions: Sequence[np.ndarray],
        tolerance: float = 0.005,
        max_steps: int = 480,
        orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
    ) -> bool:
        """Move toward *target_position* while steering around obstacles
        via an Artificial Potential Field.

        Each step:
            1. Compute F_total = F_att + F_rep
            2. V_desired = α · F_total
            3. P_new = P_current + V_desired · Δt  (clamped to reasonable step)
            4. IK → joint angles → control
        If F_total is nearly zero (local minimum) add a random perturbation.
        """
        for _ in range(max_steps):
            ee_pos = self._robot.get_ee_position()
            err = np.linalg.norm(ee_pos - target_position)
            if err < tolerance:
                return True

            f_total = self._total_field_force(ee_pos, target_position, obstacle_positions)
            v_desired = self.APF_ALPHA * f_total

            # Local-minimum escape: random perturbation
            if np.linalg.norm(v_desired) < self.APF_VEL_MIN:
                v_desired += np.random.uniform(-self.APF_PERTURB, self.APF_PERTURB, size=3)

            # Clamp step size to avoid overshoot
            step = v_desired * self.APF_DT
            step_norm = np.linalg.norm(step)
            max_step = cfg("robot", "motion", "max_step_size")
            if step_norm > max_step:
                step = step * (max_step / step_norm)

            p_new = ee_pos + step
            # Don't push below table surface
            p_new[2] = max(p_new[2], 0.0)

            arm_angles = self.solve_ik(p_new, orientation)
            finger_val = gripper_width / 2.0
            target_angles = np.concatenate([arm_angles, [finger_val, finger_val]])
            self._sim.control_joints(
                body=self._robot.body_name,
                joints=self._robot.joint_indices,
                target_angles=target_angles,
                forces=self._robot.joint_forces,
            )
            self._step()

        return False

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
    ) -> np.ndarray:
        """Solve inverse kinematics for the given end-effector pose."""
        joint_angles = self._p.calculateInverseKinematics(
            bodyIndex=self._body_id,
            endEffectorLinkIndex=self._robot.ee_link,
            targetPosition=target_position.tolist(),
            targetOrientation=target_orientation.tolist(),
        )
        return np.array(joint_angles[:7])

    def move_to(
        self,
        target_position: np.ndarray,
        gripper_width: float,
        tolerance: float = 0.005,
        max_steps: int = 480,
        orientation: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
    ) -> bool:
        """Drive the end-effector to *target_position* via straight-line IK."""
        for _ in range(max_steps):
            arm_angles = self.solve_ik(target_position, orientation)
            finger_val = gripper_width / 2.0
            target_angles = np.concatenate([arm_angles, [finger_val, finger_val]])
            self._sim.control_joints(
                body=self._robot.body_name,
                joints=self._robot.joint_indices,
                target_angles=target_angles,
                forces=self._robot.joint_forces,
            )
            self._step()
            ee_pos = self._robot.get_ee_position()
            if np.linalg.norm(ee_pos - target_position) < tolerance:
                return True
        return False

    def set_gripper(self, width: float, steps: int = None) -> None:
        """Command the parallel-jaw gripper to *width* over *steps* ticks."""
        if steps is None:
            steps = cfg("robot", "gripper", "set_gripper_steps")
        finger_val = width / 2.0
        current_arm = np.array([self._robot.get_joint_angle(j) for j in range(7)])
        target_angles = np.concatenate([current_arm, [finger_val, finger_val]])
        for _ in range(steps):
            self._sim.control_joints(
                body=self._robot.body_name,
                joints=self._robot.joint_indices,
                target_angles=target_angles,
                forces=self._robot.joint_forces,
            )
            self._step()

    def get_ee_position(self) -> np.ndarray:
        """Return the current 3-D world position of the end-effector."""
        return self._robot.get_ee_position()

    # Retract pose: arm tucked to the side so the overhead camera has
    # a clear view of the entire workspace.
    _SENSING_POSITION = np.array(cfg("robot", "sensing_position"), dtype=np.float64)

    def retract_for_sensing(self) -> bool:
        """Move the arm out of the camera's field of view."""
        return self.move_to(
            self._SENSING_POSITION,
            gripper_width=self.GRIPPER_CLOSED,
            tolerance=0.01,
            max_steps=self.move_max_steps,
        )

    def _move(
        self,
        target: np.ndarray,
        gripper_width: float,
        obstacles: Sequence[np.ndarray],
    ) -> bool:
        """Convenience: use APF when obstacles are supplied, else straight-line."""
        if len(obstacles) > 0:
            return self.move_to_field(
                target,
                gripper_width=gripper_width,
                obstacle_positions=obstacles,
                tolerance=self.move_tolerance,
                max_steps=self.move_max_steps,
            )
        return self.move_to(
            target,
            gripper_width=gripper_width,
            tolerance=self.move_tolerance,
            max_steps=self.move_max_steps,
        )

    def grasp_point_world(
        self,
        target_xyz: np.ndarray,
        object_name: Optional[str] = None,
        obstacle_positions: Optional[Sequence[np.ndarray]] = None,
    ) -> GraspResult:
        """Execute the full APPROACH-DESCEND-GRASP-LIFT state machine."""
        visited: List[GraspState] = []
        state = GraspState.APPROACH
        obs = list(obstacle_positions) if obstacle_positions else []

        approach_pos = target_xyz.copy()
        approach_pos[2] = max(target_xyz[2] + self.approach_height, self.SAFE_HEIGHT)

        descend_pos = target_xyz.copy()
        descend_pos[2] += self.descend_offset

        lift_pos = descend_pos.copy()
        lift_pos[2] = max(descend_pos[2] + self.lift_height, self.SAFE_HEIGHT)

        while state not in (GraspState.DONE, GraspState.FAILED):
            visited.append(state)

            if state == GraspState.APPROACH:
                self.set_gripper(self.GRIPPER_OPEN, steps=self.grasp_steps // 2)
                # Rise to safe height first
                safe_pos = self._robot.get_ee_position().copy()
                safe_pos[2] = self.SAFE_HEIGHT
                self._move(safe_pos, self.GRIPPER_OPEN, obs)
                # Lateral move to above target – APF steers around cubes
                ok = self._move(approach_pos, self.GRIPPER_OPEN, obs)
                if ok:
                    # Narrow to pre-grasp width before descending so the
                    # fingers don't clip neighbouring cubes.
                    self.set_gripper(self.GRIPPER_PREGRASP, steps=self.grasp_steps // 4)
                state = GraspState.DESCEND if ok else GraspState.FAILED

            elif state == GraspState.DESCEND:
                # Committed vertical drop with narrow pre-grasp width –
                # bypass APF so repulsion cannot deflect sideways.
                ok = self.move_to(
                    descend_pos,
                    gripper_width=self.GRIPPER_PREGRASP,
                    tolerance=self.move_tolerance,
                    max_steps=self.move_max_steps,
                )
                state = GraspState.GRASP if ok else GraspState.FAILED

            elif state == GraspState.GRASP:
                self.set_gripper(self.GRIPPER_CLOSED, steps=self.grasp_steps)
                state = GraspState.LIFT

            elif state == GraspState.LIFT:
                # Lift up – APF ensures we don't sideswipe neighbours
                ok = self._move(lift_pos, self.GRIPPER_CLOSED, obs)
                state = GraspState.DONE if ok else GraspState.FAILED

        visited.append(state)

        final_ee = self._robot.get_ee_position()
        final_obj = None
        if object_name is not None:
            try:
                final_obj = self._sim.get_base_position(object_name)
            except Exception:
                pass

        success = (state == GraspState.DONE)
        if success and final_obj is not None:
            success = final_obj[2] > (target_xyz[2] + self.lift_height * 0.3)

        return GraspResult(
            success=success,
            final_ee_position=final_ee,
            final_object_position=final_obj,
            states_visited=visited,
        )

    def place_at_world(
        self,
        target_xyz: np.ndarray,
        object_name: Optional[str] = None,
        *,
        place_height: float = 0.0,
        retract_height: float = None,
        obstacle_positions: Optional[Sequence[np.ndarray]] = None,
    ) -> PlaceResult:
        """Execute the TRANSIT-LOWER-RELEASE-RETRACT placement state machine."""
        if retract_height is None:
            retract_height = cfg("robot", "motion", "retract_height")
        visited: List[PlaceState] = []
        state = PlaceState.TRANSIT
        obs = list(obstacle_positions) if obstacle_positions else []

        transit_pos = target_xyz.copy()
        transit_pos[2] = max(target_xyz[2] + self.approach_height, self.SAFE_HEIGHT)

        lower_pos = target_xyz.copy()
        lower_pos[2] += place_height

        retract_pos = lower_pos.copy()
        retract_pos[2] = max(lower_pos[2] + retract_height, self.SAFE_HEIGHT)

        while state not in (PlaceState.DONE, PlaceState.FAILED):
            visited.append(state)

            if state == PlaceState.TRANSIT:
                ok = self._move(transit_pos, self.GRIPPER_CLOSED, obs)
                state = PlaceState.LOWER if ok else PlaceState.FAILED

            elif state == PlaceState.LOWER:
                # Committed vertical drop – bypass APF so the held cube
                # lands precisely on its target slot without lateral drift.
                ok = self.move_to(
                    lower_pos,
                    gripper_width=self.GRIPPER_CLOSED,
                    tolerance=self.move_tolerance,
                    max_steps=self.move_max_steps,
                )
                state = PlaceState.RELEASE if ok else PlaceState.FAILED

            elif state == PlaceState.RELEASE:
                self.set_gripper(self.GRIPPER_OPEN, steps=self.grasp_steps)
                # Let the object settle on the surface before retracting
                for _ in range(20):
                    self._step()
                state = PlaceState.RETRACT

            elif state == PlaceState.RETRACT:
                ok = self._move(retract_pos, self.GRIPPER_OPEN, obs)
                state = PlaceState.DONE if ok else PlaceState.FAILED

        visited.append(state)

        final_ee = self._robot.get_ee_position()
        final_obj = None
        if object_name is not None:
            try:
                final_obj = self._sim.get_base_position(object_name)
            except Exception:
                pass

        return PlaceResult(
            success=(state == PlaceState.DONE),
            final_ee_position=final_ee,
            final_object_position=final_obj,
            states_visited=visited,
        )

    def pick_and_place(
        self,
        pick_xyz: np.ndarray,
        place_xyz: np.ndarray,
        object_name: Optional[str] = None,
        *,
        place_height: float = 0.0,
        obstacle_positions: Optional[Sequence[np.ndarray]] = None,
    ) -> PickAndPlaceResult:
        """Grasp at *pick_xyz* then place at *place_xyz* in one call."""
        obs = list(obstacle_positions) if obstacle_positions else []
        grasp_res = self.grasp_point_world(
            pick_xyz, object_name=object_name, obstacle_positions=obs,
        )
        if not grasp_res.success:
            return PickAndPlaceResult(
                pick_success=False,
                place_success=False,
                pick_result=grasp_res,
                place_result=None,
                object_name=object_name,
            )

        # Move to safe height before lateral transit
        safe_pos = self._robot.get_ee_position().copy()
        safe_pos[2] = self.SAFE_HEIGHT
        self._move(safe_pos, self.GRIPPER_CLOSED, obs)

        place_res = self.place_at_world(
            place_xyz,
            object_name=object_name,
            place_height=place_height,
            obstacle_positions=obs,
        )

        return PickAndPlaceResult(
            pick_success=True,
            place_success=place_res.success,
            pick_result=grasp_res,
            place_result=place_res,
            object_name=object_name,
        )
