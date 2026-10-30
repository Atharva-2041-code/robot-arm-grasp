"""PyBullet simulation wrapper for the Franka pick-and-place scene."""

import logging
import os
import random
import time

import numpy as np
import pybullet as p
import pybullet_data

from config import Config

logger = logging.getLogger(__name__)


class PyBulletSimulator:
    """Owns the PyBullet client, robot, and scene objects.

    Attributes:
        client_id: PyBullet physics client id.
        robot_id: Body unique id of the Franka robot (set by setup_scene).
        objects: Body unique ids of graspable objects.
        step_count: Number of simulation steps taken.
    """

    def __init__(self, config: Config, gui: bool = True) -> None:
        self.config = config
        self.gui = gui
        self.client_id: int = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet")
        self.robot_id: int = -1
        self.plane_id: int = -1
        self.table_id: int = -1
        self.tray_id: int = -1
        self.objects: list[int] = []
        self.step_count: int = 0
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(*config.gravity, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            fixedTimeStep=config.time_step,
            numSubSteps=config.num_sub_steps,
            physicsClientId=self.client_id,
        )
        logger.info("Connected to PyBullet (client %d, gui=%s)", self.client_id, gui)

    @property
    def simulation_time(self) -> float:
        """Elapsed simulated time in seconds."""
        return self.step_count * self.config.time_step

    def setup_scene(self) -> None:
        """Load plane, table, tray, robot, and random graspable objects."""
        cfg = self.config
        self.plane_id = self._load_urdf(cfg.plane_urdf, (0.0, 0.0, 0.0))
        self.table_id = self._load_urdf(cfg.table_urdf, cfg.table_position)
        self.tray_id = self._load_urdf(
            cfg.tray_urdf, cfg.tray_position, scale=cfg.tray_scale
        )
        self.robot_id = self._load_urdf(
            cfg.robot_urdf, cfg.robot_base_position, fixed=True
        )
        self.set_robot_config(list(cfg.home_config))
        self._spawn_objects()
        self.step_n(cfg.settle_steps)
        logger.info(
            "Scene ready: robot=%d, %d objects", self.robot_id, len(self.objects)
        )

    def step(self) -> None:
        """Advance one physics step, paced to wall-clock time in GUI mode."""
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_count += 1
        if self.gui:
            time.sleep(self.config.time_step * self.config.gui_time_scale)

    def step_n(self, n: int) -> None:
        """Advance the simulation by n physics steps."""
        for _ in range(n):
            self.step()

    def set_robot_config(self, joint_angles: list[float]) -> None:
        """Instantly reset arm joints (bypasses dynamics; use for setup only)."""
        if len(joint_angles) != len(self.config.arm_joint_indices):
            raise ValueError(
                f"Expected {len(self.config.arm_joint_indices)} joint angles, "
                f"got {len(joint_angles)}"
            )
        for joint, angle in zip(self.config.arm_joint_indices, joint_angles):
            p.resetJointState(
                self.robot_id, joint, angle, physicsClientId=self.client_id
            )
        for finger in self.config.finger_joint_indices:
            p.resetJointState(
                self.robot_id,
                finger,
                self.config.finger_open_width,
                physicsClientId=self.client_id,
            )

    def get_robot_state(self) -> dict[str, list[float]]:
        """Return arm joint positions/velocities and end-effector pose."""
        states = p.getJointStates(
            self.robot_id,
            self.config.arm_joint_indices,
            physicsClientId=self.client_id,
        )
        ee = p.getLinkState(
            self.robot_id, self.config.ee_link_index, physicsClientId=self.client_id
        )
        return {
            "joint_positions": [s[0] for s in states],
            "joint_velocities": [s[1] for s in states],
            "ee_position": list(ee[4]),
            "ee_orientation": list(ee[5]),
        }

    def get_object_state(
        self, obj_id: int
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return (position, orientation quaternion) of a body."""
        pos, orn = p.getBasePositionAndOrientation(
            obj_id, physicsClientId=self.client_id
        )
        return pos, orn

    def get_camera_image(
        self, view_matrix: list[float], proj_matrix: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render an RGB image and raw depth buffer from the given matrices."""
        cfg = self.config
        _, _, rgb, depth, _ = p.getCameraImage(
            cfg.camera_width,
            cfg.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client_id,
        )
        rgb_array = np.reshape(rgb, (cfg.camera_height, cfg.camera_width, 4))[:, :, :3]
        depth_array = np.reshape(depth, (cfg.camera_height, cfg.camera_width))
        return rgb_array.astype(np.uint8), depth_array.astype(np.float32)

    def check_collision(
        self,
        body_a: int,
        body_b: int,
        contact_distance: float = 0.005,
        ignore_links_a: tuple[int, ...] = (),
    ) -> bool:
        """Return True if the bodies touch within the contact tolerance.

        Args:
            body_a: First body unique id.
            body_b: Second body unique id.
            contact_distance: Separation below which bodies count as touching.
            ignore_links_a: Link indices of body_a to exclude (e.g. -1 for a
                base link that intentionally rests on a support surface).
        """
        points = p.getClosestPoints(
            body_a, body_b, distance=contact_distance, physicsClientId=self.client_id
        )
        return any(point[3] not in ignore_links_a for point in points)

    def save_state(self) -> int:
        """Snapshot the full simulation state; returns a state id."""
        return p.saveState(physicsClientId=self.client_id)

    def restore_state(self, state_id: int) -> None:
        """Restore a snapshot taken with save_state and free it."""
        p.restoreState(stateId=state_id, physicsClientId=self.client_id)
        p.removeState(state_id, physicsClientId=self.client_id)

    def disconnect(self) -> None:
        """Shut down the physics client."""
        if p.isConnected(self.client_id):
            p.disconnect(physicsClientId=self.client_id)
            logger.info("Disconnected client %d", self.client_id)

    def _load_urdf(
        self,
        relative_path: str,
        position: tuple[float, float, float],
        fixed: bool = False,
        scale: float = 1.0,
    ) -> int:
        path = os.path.join(pybullet_data.getDataPath(), relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"URDF not found in pybullet_data: {relative_path}")
        body_id = p.loadURDF(
            relative_path,
            basePosition=position,
            useFixedBase=fixed,
            globalScaling=scale,
            physicsClientId=self.client_id,
        )
        if body_id < 0:
            raise RuntimeError(f"Failed to load URDF: {relative_path}")
        return body_id

    def _spawn_objects(self) -> None:
        cfg = self.config
        for i in range(cfg.num_objects):
            urdf = cfg.object_urdfs[i % len(cfg.object_urdfs)]
            position = (
                random.uniform(cfg.spawn_region_min[0], cfg.spawn_region_max[0]),
                random.uniform(cfg.spawn_region_min[1], cfg.spawn_region_max[1]),
                cfg.spawn_height,
            )
            obj_id = self._load_urdf(urdf, position)
            self.objects.append(obj_id)
            self.step_n(60)
