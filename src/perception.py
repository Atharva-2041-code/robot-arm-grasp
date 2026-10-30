"""Camera and perception interfaces (detector/pose estimator are skeletons)."""

import logging

import numpy as np
import pybullet as p

from config import Config
from src.simulator import PyBulletSimulator

logger = logging.getLogger(__name__)


class Camera:
    """Fixed eye-to-hand camera above the workspace."""

    def __init__(self, config: Config, simulator: PyBulletSimulator) -> None:
        self.config = config
        self.simulator = simulator
        self.view_matrix: list[float] = p.computeViewMatrix(
            cameraEyePosition=config.camera_eye,
            cameraTargetPosition=config.camera_target,
            cameraUpVector=config.camera_up,
        )
        self.proj_matrix: list[float] = p.computeProjectionMatrixFOV(
            fov=config.camera_fov,
            aspect=config.camera_width / config.camera_height,
            nearVal=config.camera_near,
            farVal=config.camera_far,
        )

    def get_image(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rgb uint8 HxWx3, metric depth float32 HxW)."""
        rgb, depth_buffer = self.simulator.get_camera_image(
            self.view_matrix, self.proj_matrix
        )
        return rgb, self._depth_buffer_to_metric(depth_buffer)

    def pixel_to_world(self, u: int, v: int, depth: float) -> np.ndarray:
        """Back-project pixel (u, v) with metric depth to a world point."""
        cfg = self.config
        x_ndc = (2.0 * u / cfg.camera_width) - 1.0
        y_ndc = 1.0 - (2.0 * v / cfg.camera_height)
        z_ndc = self._metric_to_ndc_depth(depth)
        proj = np.reshape(self.proj_matrix, (4, 4), order="F")
        view = np.reshape(self.view_matrix, (4, 4), order="F")
        clip = np.linalg.inv(proj @ view)
        point = clip @ np.array([x_ndc, y_ndc, z_ndc, 1.0])
        return point[:3] / point[3]

    def _depth_buffer_to_metric(self, depth_buffer: np.ndarray) -> np.ndarray:
        near, far = self.config.camera_near, self.config.camera_far
        return far * near / (far - (far - near) * depth_buffer)

    def _metric_to_ndc_depth(self, depth: float) -> float:
        near, far = self.config.camera_near, self.config.camera_far
        buffer_value = (far - far * near / depth) / (far - near)
        return 2.0 * buffer_value - 1.0


class ObjectDetector:
    """Skeleton object detector — integrate your own model here.

    To implement: load a detection model (e.g. torchvision Mask R-CNN or a
    YOLO variant) in __init__ and run inference in detect(). Each detection
    dict must contain keys: class (str), confidence (float),
    bbox (x_min, y_min, x_max, y_max in pixels), and mask (HxW bool array
    or None).
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Detect objects in an RGB image.

        Args:
            rgb: HxWx3 uint8 image.

        Returns:
            List of detection dicts (empty until a model is integrated;
            the task manager falls back to simulator ground truth).
        """
        logger.debug("ObjectDetector.detect is a skeleton; returning no detections")
        return []


class PoseEstimator:
    """Skeleton 6-DoF pose estimator — integrate your own model here.

    To implement: use the detection mask/bbox to crop the depth image,
    back-project points with Camera.pixel_to_world, and fit a pose (e.g.
    centroid + PCA, or a learned estimator such as DenseFusion). Return a
    dict with keys: pos (x, y, z), orn (quaternion x, y, z, w), and
    confidence (float).
    """

    def __init__(self, config: Config, camera: Camera) -> None:
        self.config = config
        self.camera = camera

    def estimate_pose(
        self, rgb: np.ndarray, depth: np.ndarray, detection: dict
    ) -> dict | None:
        """Estimate the 6-DoF pose of a detected object.

        Args:
            rgb: HxWx3 uint8 image.
            depth: HxW metric depth image.
            detection: One detection dict from ObjectDetector.detect.

        Returns:
            Pose dict {pos, orn, confidence}, or None if estimation fails.
        """
        logger.debug("PoseEstimator.estimate_pose is a skeleton; returning None")
        return None
