import numpy as np
import pytest


class TestImports:

    def test_import_camera(self):
        from panda_control.camera import (
            CameraHandler, CameraIntrinsics, CameraExtrinsics, CameraFrame,
        )

    def test_import_scene(self):
        from panda_control.scene import SceneManager

    def test_import_perception(self):
        from panda_control.perception import Detection, PerceptionSystem

    def test_import_robot(self):
        from panda_control.robot import (
            GraspState, GraspResult, PlaceState, PlaceResult,
            PickAndPlaceResult, RobotController,
        )

    def test_import_task_runner(self):
        from panda_control.task_runner import TaskRunner

    def test_import_package(self):
        from panda_control import (
            CameraHandler, CameraIntrinsics, CameraExtrinsics, CameraFrame,
            Detection, PerceptionSystem,
            GraspState, GraspResult, PlaceState, PlaceResult,
            PickAndPlaceResult, RobotController,
            SceneManager, TaskRunner,
        )
