"""Joint-position control for trajectory execution and gripper actions."""

import logging

import numpy as np
import pybullet as p

from config import Config
from src.simulator import PyBulletSimulator

logger = logging.getLogger(__name__)


class RobotController:
    """Executes joint trajectories and gripper open/close commands."""

    def __init__(self, config: Config, simulator: PyBulletSimulator) -> None:
        self.config = config
        self.simulator = simulator

    def execute_trajectory(
        self, joint_trajectory: list[list[float]], timeout: float = 30.0
    ) -> bool:
        """Track a sequence of joint waypoints with position control.

        Args:
            joint_trajectory: List of 7-value joint configurations.
            timeout: Maximum simulated seconds for the whole trajectory.

        Returns:
            True if every waypoint converged within tolerance.
        """
        max_steps = int(timeout / self.config.time_step)
        steps_used = 0
        for waypoint in joint_trajectory:
            self._command_arm(waypoint)
            converged = False
            while steps_used < max_steps:
                self.simulator.step()
                steps_used += 1
                if self._at_target(waypoint):
                    converged = True
                    break
            if not converged:
                logger.warning(
                    "Trajectory timed out after %.1fs (waypoint %s)",
                    steps_used * self.config.time_step,
                    np.round(waypoint, 2),
                )
                return False
        return True

    def grasp_object(self, grip_force: float = 50.0) -> bool:
        """Close the gripper and report whether something was gripped.

        Returns:
            True if the fingers stalled before fully closing (an object
            is between them), False if they closed on nothing.
        """
        self._command_fingers(self.config.finger_closed_width, grip_force)
        self.simulator.step_n(self.config.gripper_settle_steps)
        width = self._finger_width()
        gripped = width > self.config.grasp_margin
        logger.info("Gripper closed to %.4fm (gripped=%s)", width, gripped)
        return gripped

    def release_object(self) -> bool:
        """Open the gripper fully.

        Returns:
            True once the fingers have reopened.
        """
        self._command_fingers(
            self.config.finger_open_width, self.config.max_joint_force
        )
        self.simulator.step_n(self.config.gripper_settle_steps)
        logger.info("Gripper released")
        return True

    def _command_arm(self, waypoint: list[float]) -> None:
        for joint, angle in zip(self.config.arm_joint_indices, waypoint):
            p.setJointMotorControl2(
                self.simulator.robot_id,
                joint,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=self.config.max_joint_force,
                maxVelocity=self.config.max_joint_velocity,
                physicsClientId=self.simulator.client_id,
            )

    def _command_fingers(self, width: float, force: float) -> None:
        for finger in self.config.finger_joint_indices:
            p.setJointMotorControl2(
                self.simulator.robot_id,
                finger,
                p.POSITION_CONTROL,
                targetPosition=width,
                force=force,
                physicsClientId=self.simulator.client_id,
            )

    def _at_target(self, waypoint: list[float]) -> bool:
        current = self.simulator.get_robot_state()["joint_positions"]
        error = np.abs(np.asarray(current) - np.asarray(waypoint))
        return bool(np.max(error) < self.config.joint_tolerance)

    def _finger_width(self) -> float:
        states = p.getJointStates(
            self.simulator.robot_id,
            self.config.finger_joint_indices,
            physicsClientId=self.simulator.client_id,
        )
        return float(sum(s[0] for s in states) / len(states))
