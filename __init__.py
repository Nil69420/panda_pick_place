"""panda_control -- vision-guided pick-and-place for the Franka Panda.

Public API re-exports for convenient top-level imports::

    from panda_control import TaskRunner, RobotController, CameraHandler
"""
from panda_control.camera import CameraExtrinsics, CameraFrame, CameraHandler, CameraIntrinsics
from panda_control.perception import Detection, PerceptionSystem
from panda_control.robot import GraspResult, GraspState, PickAndPlaceResult, PlaceResult, PlaceState, RobotController
from panda_control.scene import SceneManager
from panda_control.task_runner import TaskRunner

__all__ = [
    "CameraExtrinsics",
    "CameraFrame",
    "CameraHandler",
    "CameraIntrinsics",
    "Detection",
    "GraspResult",
    "GraspState",
    "PerceptionSystem",
    "PickAndPlaceResult",
    "PlaceResult",
    "PlaceState",
    "RobotController",
    "SceneManager",
    "TaskRunner",
]
