"""Top-down grasp candidate generation and scoring."""

import logging
import math

import numpy as np
import pybullet as p

from config import Config
from src.simulator import PyBulletSimulator

logger = logging.getLogger(__name__)


class GraspPlanner:
    """Generates and ranks top-down antipodal grasp candidates."""

    def __init__(self, config: Config, simulator: PyBulletSimulator) -> None:
        self.config = config
        self.simulator = simulator

    def plan_grasp(
        self,
        obj_pos: tuple[float, ...],
        obj_orn: tuple[float, ...],
        obj_shape: str = "box",
    ) -> dict | None:
        """Return the best grasp for an object, or None if no valid grasp.

        Args:
            obj_pos: Object position in world coordinates.
            obj_orn: Object orientation quaternion (x, y, z, w).
            obj_shape: Shape hint ("box", "sphere", "cylinder").

        Returns:
            Grasp dict with gripper_position, gripper_orientation,
            grip_width, quality_score, approach_direction, and
            approach_distance, or None if all candidates are invalid.
        """
        candidates = self._generate_candidates(obj_pos, obj_orn, obj_shape)
        scored = self._score_grasps(candidates)
        if not scored:
            logger.warning("No valid grasp candidates for object at %s", obj_pos)
            return None
        best = scored[0]
        logger.info(
            "Selected grasp at %s (score %.3f)",
            np.round(best["gripper_position"], 3),
            best["quality_score"],
        )
        return best

    def _generate_candidates(
        self,
        obj_pos: tuple[float, ...],
        obj_orn: tuple[float, ...],
        obj_shape: str,
    ) -> list[dict]:
        cfg = self.config
        grasp_z = obj_pos[2] + cfg.grasp_depth_offset
        grip_width = cfg.finger_open_width - cfg.grasp_margin
        candidates = []
        num_yaws = 1 if obj_shape == "sphere" else cfg.grasp_num_samples
        for i in range(num_yaws):
            yaw = i * math.pi / num_yaws
            if obj_shape != "sphere":
                yaw += p.getEulerFromQuaternion(obj_orn)[2]
            orientation = p.getQuaternionFromEuler((math.pi, 0.0, yaw))
            candidates.append(
                {
                    "gripper_position": (obj_pos[0], obj_pos[1], grasp_z),
                    "gripper_orientation": orientation,
                    "grip_width": grip_width,
                    "quality_score": 0.0,
                    "approach_direction": (0.0, 0.0, -1.0),
                    "approach_distance": cfg.grasp_approach_distance,
                }
            )
        return candidates

    def _score_grasps(self, grasps: list[dict]) -> list[dict]:
        cfg = self.config
        valid = []
        for grasp in grasps:
            pos = grasp["gripper_position"]
            if pos[2] < cfg.table_height:
                continue
            clearance = self._min_clearance(pos)
            reach_penalty = np.linalg.norm(
                np.asarray(pos) - np.asarray(cfg.robot_base_position)
            )
            grasp["quality_score"] = float(
                min(clearance, cfg.grasp_min_clearance) - 0.1 * reach_penalty
            )
            valid.append(grasp)
        return sorted(valid, key=lambda g: g["quality_score"], reverse=True)

    def _min_clearance(self, position: tuple[float, ...]) -> float:
        """Distance from a grasp point to the nearest other object."""
        target = np.asarray(position[:2])
        distances = []
        for obj_id in self.simulator.objects:
            obj_pos, _ = self.simulator.get_object_state(obj_id)
            distance = float(np.linalg.norm(np.asarray(obj_pos[:2]) - target))
            if distance > 1e-3:
                distances.append(distance)
        return min(distances, default=self.config.grasp_min_clearance)
