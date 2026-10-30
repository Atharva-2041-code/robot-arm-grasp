"""Joint-space RRT motion planner using PyBullet inverse kinematics."""

import logging
import random
import time

import numpy as np
import pybullet as p

from config import Config
from src.simulator import PyBulletSimulator

logger = logging.getLogger(__name__)


class MotionPlanner:
    """Plans collision-free joint trajectories with a basic RRT."""

    def __init__(self, config: Config, simulator: PyBulletSimulator) -> None:
        self.config = config
        self.simulator = simulator
        self.joint_limits: list[tuple[float, float]] = [
            p.getJointInfo(
                simulator.robot_id, joint, physicsClientId=simulator.client_id
            )[8:10]
            for joint in config.arm_joint_indices
        ]

    def plan_path(
        self,
        start_config: list[float],
        goal_pos: tuple[float, ...],
        goal_orn: tuple[float, ...],
        planning_time: float = 5.0,
    ) -> list[list[float]] | None:
        """Plan a joint-space path from start_config to a Cartesian goal.

        Args:
            start_config: Current arm joint angles (7 values).
            goal_pos: Target end-effector position in world coordinates.
            goal_orn: Target end-effector orientation quaternion.
            planning_time: Wall-clock budget in seconds.

        Returns:
            List of joint configurations (waypoints), or None on failure.
        """
        goal_config = self._inverse_kinematics(goal_pos, goal_orn)
        if goal_config is None:
            return None
        state_id = self.simulator.save_state()
        try:
            if not self._collision_free(goal_config):
                logger.warning("IK goal configuration is in collision")
                return None
            path = self._rrt(start_config, goal_config, planning_time)
            if path is None:
                logger.warning("RRT failed to reach goal within budget")
                return None
            return self._smooth_path(path)
        finally:
            self.simulator.restore_state(state_id)

    def inverse_kinematics(
        self, position: tuple[float, ...], orientation: tuple[float, ...]
    ) -> list[float] | None:
        """Solve IK for the end effector; returns arm joints or None."""
        return self._inverse_kinematics(position, orientation)

    def _inverse_kinematics(
        self, position: tuple[float, ...], orientation: tuple[float, ...]
    ) -> list[float] | None:
        cfg = self.config
        solution = p.calculateInverseKinematics(
            self.simulator.robot_id,
            cfg.ee_link_index,
            position,
            orientation,
            maxNumIterations=cfg.ik_max_iterations,
            residualThreshold=cfg.ik_residual_threshold,
            physicsClientId=self.simulator.client_id,
        )
        arm_solution = list(solution[: len(cfg.arm_joint_indices)])
        for angle, (low, high) in zip(arm_solution, self.joint_limits):
            if low < high and not low <= angle <= high:
                logger.warning("IK solution violates joint limits")
                return None
        return arm_solution

    def _collision_free(self, config: list[float]) -> bool:
        for joint, angle in zip(self.config.arm_joint_indices, config):
            p.resetJointState(
                self.simulator.robot_id,
                joint,
                angle,
                physicsClientId=self.simulator.client_id,
            )
        obstacles = [self.simulator.plane_id, self.simulator.table_id]
        base_link = -1
        return not any(
            self.simulator.check_collision(
                self.simulator.robot_id,
                obstacle,
                contact_distance=0.0,
                ignore_links_a=(base_link,),
            )
            for obstacle in obstacles
        )

    def _rrt(
        self,
        start: list[float],
        goal: list[float],
        planning_time: float,
    ) -> list[list[float]] | None:
        cfg = self.config
        nodes = [np.asarray(start)]
        parents = [0]
        goal_array = np.asarray(goal)
        deadline = time.monotonic() + planning_time
        for _ in range(cfg.rrt_max_iterations):
            if time.monotonic() > deadline:
                break
            sample = (
                goal_array
                if random.random() < cfg.rrt_goal_bias
                else self._sample_config()
            )
            nearest_index = int(
                np.argmin([np.linalg.norm(node - sample) for node in nodes])
            )
            new_node = self._steer(nodes[nearest_index], sample)
            if not self._collision_free(new_node.tolist()):
                continue
            nodes.append(new_node)
            parents.append(nearest_index)
            if np.linalg.norm(new_node - goal_array) < cfg.rrt_goal_tolerance:
                path = [goal]
                index = len(nodes) - 1
                while index != 0:
                    path.append(nodes[index].tolist())
                    index = parents[index]
                path.append(start)
                path.reverse()
                logger.info("RRT found path with %d waypoints", len(path))
                return path
        return None

    def _sample_config(self) -> np.ndarray:
        return np.array(
            [random.uniform(low, high) for low, high in self.joint_limits]
        )

    def _steer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        direction = target - source
        distance = float(np.linalg.norm(direction))
        if distance <= self.config.rrt_step_size:
            return target
        return source + direction / distance * self.config.rrt_step_size

    def _smooth_path(self, path: list[list[float]]) -> list[list[float]]:
        """Shortcut smoothing: try to connect random waypoint pairs directly."""
        smoothed = [np.asarray(waypoint) for waypoint in path]
        for _ in range(self.config.smoothing_iterations):
            if len(smoothed) < 3:
                break
            i, j = sorted(random.sample(range(len(smoothed)), 2))
            if j - i < 2:
                continue
            if self._segment_free(smoothed[i], smoothed[j]):
                smoothed = smoothed[: i + 1] + smoothed[j:]
        return [waypoint.tolist() for waypoint in smoothed]

    def _segment_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        distance = float(np.linalg.norm(end - start))
        steps = max(int(distance / self.config.rrt_step_size), 1)
        return all(
            self._collision_free((start + (end - start) * t / steps).tolist())
            for t in range(steps + 1)
        )
