import numpy as np
import pytest

from panda_control.robot import (
    GraspResult,
    GraspState,
    PickAndPlaceResult,
    PlaceResult,
    PlaceState,
    RobotController,
)
from panda_control.task_runner import TaskRunner


class TestGraspImports:

    def test_import_grasp_state(self):
        assert hasattr(GraspState, "APPROACH")
        assert hasattr(GraspState, "DESCEND")
        assert hasattr(GraspState, "GRASP")
        assert hasattr(GraspState, "LIFT")
        assert hasattr(GraspState, "DONE")
        assert hasattr(GraspState, "FAILED")

    def test_import_grasp_result(self):
        r = GraspResult(
            success=True,
            final_ee_position=np.zeros(3),
            final_object_position=np.zeros(3),
            states_visited=[GraspState.DONE],
        )
        assert r.success is True

    def test_import_from_package(self):
        from panda_control import (
            GraspResult, GraspState, PlaceState, PlaceResult,
            PickAndPlaceResult, RobotController,
        )
        assert GraspState.IDLE.value == "idle"
        assert PlaceState.TRANSIT.value == "transit"


class TestRobotController:

    def test_joint_limits_shape(self):
        assert RobotController.PANDA_LOWER_LIMITS.shape == (7,)
        assert RobotController.PANDA_UPPER_LIMITS.shape == (7,)
        assert RobotController.PANDA_JOINT_RANGES.shape == (7,)
        assert RobotController.PANDA_REST_POSES.shape == (7,)

    def test_joint_ranges_positive(self):
        assert np.all(RobotController.PANDA_JOINT_RANGES > 0)

    def test_solve_ik(self, pipeline):
        target = np.array([0.3, 0.0, 0.15])
        angles = pipeline.controller.solve_ik(target)
        assert angles.shape == (7,)

    def test_ik_within_limits(self, pipeline):
        target = np.array([0.3, 0.0, 0.15])
        angles = pipeline.controller.solve_ik(target)
        assert np.all(angles >= RobotController.PANDA_LOWER_LIMITS - 0.1)
        assert np.all(angles <= RobotController.PANDA_UPPER_LIMITS + 0.1)

    def test_move_to_reachable(self, pipeline):
        target = np.array([0.3, 0.0, 0.25])
        ee_before = pipeline.robot.get_ee_position().copy()
        pipeline.controller.move_to(
            target,
            gripper_width=RobotController.GRIPPER_OPEN,
            tolerance=0.01,
            max_steps=480,
        )
        ee_after = pipeline.robot.get_ee_position()
        dist_before = np.linalg.norm(ee_before - target)
        dist_after = np.linalg.norm(ee_after - target)
        assert dist_after < dist_before


class TestGraspExecution:

    def test_controller_exists(self, pipeline):
        assert isinstance(pipeline.controller, RobotController)

    def test_default_parameters(self, pipeline):
        c = pipeline.controller
        assert c.approach_height == 0.15
        assert c.lift_height == 0.20

    def test_grasp_visits_all_states(self, pipeline):
        cube_name = pipeline.cube_names[0]
        gt_pos = np.array(pipeline.sim.get_base_position(cube_name))
        result = pipeline.controller.grasp_point_world(gt_pos, object_name=cube_name)
        assert isinstance(result, GraspResult)
        assert GraspState.APPROACH in result.states_visited
        assert GraspState.DESCEND in result.states_visited
        assert GraspState.GRASP in result.states_visited
        assert GraspState.LIFT in result.states_visited
        assert result.states_visited[-1] in (GraspState.DONE, GraspState.FAILED)

    def test_grasp_result_has_positions(self, pipeline):
        cube_name = pipeline.cube_names[0]
        gt_pos = np.array(pipeline.sim.get_base_position(cube_name))
        result = pipeline.controller.grasp_point_world(gt_pos, object_name=cube_name)
        assert result.final_ee_position.shape == (3,)
        assert result.final_object_position is not None
        assert result.final_object_position.shape == (3,)


class TestDetectAndGrasp:

    def test_detect_and_grasp_returns_result(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=1, seed=7)
        result = p.detect_and_grasp()
        assert isinstance(result, GraspResult)
        assert len(result.states_visited) >= 2
        p.close()

    def test_detect_and_grasp_with_name(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=2, seed=8)
        cube = p.cube_names[0]
        result = p.detect_and_grasp(cube_name=cube)
        assert isinstance(result, GraspResult)
        p.close()


class TestGraspAfterReset:

    def test_controller_works_after_reset(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=1, seed=15)
        p.reset(seed=25)
        cube = p.cube_names[0]
        gt_pos = np.array(p.sim.get_base_position(cube))
        result = p.controller.grasp_point_world(gt_pos, object_name=cube)
        assert isinstance(result, GraspResult)
        assert GraspState.APPROACH in result.states_visited
        p.close()


class TestPickAndPlace:

    def test_pick_and_place_returns_result(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=2, seed=30)
        cube = p.cube_names[0]
        pick_pos = np.array(p.sim.get_base_position(cube))
        place_pos = np.array([0.0, 0.0, 0.02])
        result = p.controller.pick_and_place(
            pick_pos, place_pos, object_name=cube,
        )
        assert isinstance(result, PickAndPlaceResult)
        assert result.pick_result is not None
        p.close()

    def test_place_visits_all_states(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=1, seed=31)
        cube = p.cube_names[0]
        pick_pos = np.array(p.sim.get_base_position(cube))
        place_pos = np.array([0.0, 0.0, 0.02])
        result = p.controller.pick_and_place(
            pick_pos, place_pos, object_name=cube,
        )
        if result.pick_success and result.place_result is not None:
            assert PlaceState.TRANSIT in result.place_result.states_visited
            assert PlaceState.LOWER in result.place_result.states_visited
            assert PlaceState.RELEASE in result.place_result.states_visited
            assert PlaceState.RETRACT in result.place_result.states_visited
        p.close()


class TestStackAll:

    def test_stack_all_returns_list(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=2, seed=50)
        results = p.stack_all()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, PickAndPlaceResult)
        p.close()


class TestPickAndPlaceAll:

    def test_pick_and_place_all_returns_list(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=2, seed=55)
        results = p.pick_and_place_all()
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, PickAndPlaceResult)
        p.close()


class TestPickAndPlaceOne:

    def test_pick_and_place_one_returns_result(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=2, seed=60)
        result = p.pick_and_place_one()
        assert isinstance(result, PickAndPlaceResult)
        p.close()

    def test_pick_and_place_one_with_target(self):
        p = TaskRunner(render_mode="rgb_array", n_cubes=2, seed=61)
        name = p.cube_names[0]
        result = p.pick_and_place_one(cube_name=name)
        assert isinstance(result, PickAndPlaceResult)
        assert result.object_name is not None
        p.close()


class TestPotentialField:

    def test_attractive_force_direction(self):
        """F_att should point from gripper toward goal."""
        gripper = np.array([0.0, 0.0, 0.1])
        goal = np.array([0.3, 0.0, 0.1])
        f = RobotController._attractive_force(gripper, goal, xi=1.0)
        assert f[0] > 0  # should pull toward positive X
        assert np.allclose(f[1:], [0.0, 0.0])

    def test_repulsive_force_pushes_away(self):
        """F_rep should push gripper away from obstacle."""
        gripper = np.array([0.1, 0.0, 0.05])
        obstacle = [np.array([0.1, 0.05, 0.05])]  # 5cm away in Y
        f = RobotController._repulsive_force(gripper, obstacle, eta=0.005, d_thresh=0.12)
        assert f[1] < 0  # pushed away in -Y

    def test_repulsive_force_zero_outside_threshold(self):
        """No repulsion when obstacle is beyond d_thresh."""
        gripper = np.array([0.0, 0.0, 0.1])
        obstacle = [np.array([1.0, 0.0, 0.1])]  # 1m away
        f = RobotController._repulsive_force(gripper, obstacle, eta=0.005, d_thresh=0.12)
        assert np.allclose(f, 0.0)

    def test_move_to_field_no_obstacles(self, pipeline):
        """move_to_field with no obstacles converges toward target."""
        target = np.array([0.3, 0.0, 0.25])
        ee_before = pipeline.robot.get_ee_position().copy()
        pipeline.controller.move_to_field(
            target,
            gripper_width=RobotController.GRIPPER_OPEN,
            obstacle_positions=[],
            tolerance=0.02,
            max_steps=480,
        )
        ee_after = pipeline.robot.get_ee_position()
        dist_before = np.linalg.norm(ee_before - target)
        dist_after = np.linalg.norm(ee_after - target)
        assert dist_after < dist_before
