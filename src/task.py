"""Pick-and-place task orchestration as a simple phase machine."""

import logging
import time

import numpy as np

from config import Config
from src.grasp_planner import GraspPlanner
from src.motion_planner import MotionPlanner
from src.perception import Camera, ObjectDetector, PoseEstimator
from src.robot_controller import RobotController
from src.simulator import PyBulletSimulator

logger = logging.getLogger(__name__)


class TaskManager:
    """Runs perceive -> plan -> execute cycles for pick-and-place."""

    def __init__(
        self,
        simulator: PyBulletSimulator,
        camera: Camera,
        detector: ObjectDetector,
        pose_estimator: PoseEstimator,
        grasp_planner: GraspPlanner,
        motion_planner: MotionPlanner,
        robot_controller: RobotController,
        config: Config,
    ) -> None:
        self.simulator = simulator
        self.camera = camera
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.grasp_planner = grasp_planner
        self.motion_planner = motion_planner
        self.robot_controller = robot_controller
        self.config = config
        self._remaining: list[int] = list(simulator.objects)

    def execute_pick_and_place(self) -> tuple[bool, dict]:
        """Pick one object from the workspace and place it in the tray.

        Returns:
            (success, data) where data contains perception_time,
            planning_time, execution_time, success, and error.
        """
        data: dict = {
            "perception_time": 0.0,
            "planning_time": 0.0,
            "execution_time": 0.0,
            "success": False,
            "error": None,
        }
        try:
            target_id, obj_pos, obj_orn = self._phase_perceive(data)
            grasp, path = self._phase_plan(data, obj_pos, obj_orn)
            self._phase_execute(data, target_id, grasp, path)
            data["success"] = self._check_placement(target_id)
            if data["success"]:
                self._remaining.remove(target_id)
        except TaskError as exc:
            data["error"] = str(exc)
            logger.warning("Pick-and-place failed: %s", exc)
        return data["success"], data

    def _phase_perceive(self, data: dict) -> tuple[int, tuple, tuple]:
        start = time.monotonic()
        rgb, depth = self.camera.get_image()
        detections = self.detector.detect(rgb)
        pose = None
        if detections:
            pose = self.pose_estimator.estimate_pose(rgb, depth, detections[0])
        if pose is not None:
            target_id, obj_pos, obj_orn = (
                self._nearest_object(pose["pos"]),
                tuple(pose["pos"]),
                tuple(pose["orn"]),
            )
        else:
            logger.info("Perception skeleton returned nothing; using ground truth")
            if not self._remaining:
                raise TaskError("no objects remaining in workspace")
            target_id = self._remaining[0]
            obj_pos, obj_orn = self.simulator.get_object_state(target_id)
        data["perception_time"] = time.monotonic() - start
        return target_id, obj_pos, obj_orn

    def _phase_plan(
        self, data: dict, obj_pos: tuple, obj_orn: tuple
    ) -> tuple[dict, list[list[float]]]:
        start = time.monotonic()
        grasp = self.grasp_planner.plan_grasp(obj_pos, obj_orn)
        if grasp is None:
            raise TaskError("grasp planning failed")
        pregrasp_pos = self._offset_along_approach(grasp)
        current = self.simulator.get_robot_state()["joint_positions"]
        path = self.motion_planner.plan_path(
            current, pregrasp_pos, grasp["gripper_orientation"]
        )
        if path is None:
            raise TaskError("motion planning to pre-grasp failed")
        data["planning_time"] = time.monotonic() - start
        return grasp, path

    def _phase_execute(
        self, data: dict, target_id: int, grasp: dict, path: list[list[float]]
    ) -> None:
        start = time.monotonic()
        cfg = self.config
        if not self.robot_controller.execute_trajectory(
            path, timeout=cfg.trajectory_timeout
        ):
            raise TaskError("pre-grasp trajectory execution failed")
        self._move_to_pose(grasp["gripper_position"], grasp["gripper_orientation"])
        if not self.robot_controller.grasp_object(cfg.grip_force):
            raise TaskError("gripper closed on nothing")
        lift_pos = (
            grasp["gripper_position"][0],
            grasp["gripper_position"][1],
            grasp["gripper_position"][2] + cfg.lift_height,
        )
        self._move_to_pose(lift_pos, grasp["gripper_orientation"])
        drop_pos = (
            cfg.tray_position[0],
            cfg.tray_position[1],
            cfg.tray_position[2] + cfg.place_height_offset,
        )
        tray_path = self.motion_planner.plan_path(
            self.simulator.get_robot_state()["joint_positions"],
            drop_pos,
            grasp["gripper_orientation"],
        )
        if tray_path is None:
            raise TaskError("motion planning to tray failed")
        if not self.robot_controller.execute_trajectory(
            tray_path, timeout=cfg.trajectory_timeout
        ):
            raise TaskError("tray trajectory execution failed")
        self.robot_controller.release_object()
        self.simulator.step_n(cfg.settle_steps)
        data["execution_time"] = time.monotonic() - start

    def _move_to_pose(self, position: tuple, orientation: tuple) -> None:
        """Move to a nearby Cartesian pose via a single IK waypoint."""
        goal = self.motion_planner.inverse_kinematics(position, orientation)
        if goal is None:
            raise TaskError(f"IK failed for pose {np.round(position, 3)}")
        if not self.robot_controller.execute_trajectory(
            [goal], timeout=self.config.trajectory_timeout
        ):
            raise TaskError(f"move to pose {np.round(position, 3)} failed")

    def _offset_along_approach(self, grasp: dict) -> tuple[float, float, float]:
        pos = np.asarray(grasp["gripper_position"])
        approach = np.asarray(grasp["approach_direction"])
        pregrasp = pos - approach * grasp["approach_distance"]
        return tuple(pregrasp)

    def _nearest_object(self, position: tuple) -> int:
        target = np.asarray(position)
        return min(
            self._remaining,
            key=lambda obj_id: float(
                np.linalg.norm(
                    np.asarray(self.simulator.get_object_state(obj_id)[0]) - target
                )
            ),
        )

    def _check_placement(self, target_id: int) -> bool:
        obj_pos, _ = self.simulator.get_object_state(target_id)
        tray_xy = np.asarray(self.config.tray_position[:2])
        distance = float(np.linalg.norm(np.asarray(obj_pos[:2]) - tray_xy))
        return distance < self.config.place_success_radius


class TaskError(RuntimeError):
    """Raised when a pick-and-place phase fails."""
