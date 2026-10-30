"""Tests for the camera and perception interfaces."""

import numpy as np
import pytest

from config import Config
from src.perception import Camera, ObjectDetector, PoseEstimator
from src.simulator import PyBulletSimulator


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(num_objects=1, settle_steps=60)


@pytest.fixture(scope="module")
def simulator(config: Config) -> PyBulletSimulator:
    sim = PyBulletSimulator(config, gui=False)
    sim.setup_scene()
    yield sim
    sim.disconnect()


@pytest.fixture(scope="module")
def camera(config: Config, simulator: PyBulletSimulator) -> Camera:
    return Camera(config, simulator)


def test_camera_image_shapes(camera: Camera, config: Config) -> None:
    rgb, depth = camera.get_image()
    assert rgb.shape == (config.camera_height, config.camera_width, 3)
    assert depth.shape == (config.camera_height, config.camera_width)
    assert depth.min() >= config.camera_near
    assert depth.max() <= config.camera_far


def test_pixel_to_world_roundtrip(camera: Camera, config: Config) -> None:
    _, depth = camera.get_image()
    center_u = config.camera_width // 2
    center_v = config.camera_height // 2
    point = camera.pixel_to_world(center_u, center_v, float(depth[center_v, center_u]))
    assert point.shape == (3,)
    # The camera looks at the table workspace; the center ray hits near it.
    assert abs(point[2] - config.table_height) < 0.3


def test_detector_interface(config: Config) -> None:
    detector = ObjectDetector(config)
    rgb = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
    detections = detector.detect(rgb)
    assert isinstance(detections, list)


def test_pose_estimator_interface(camera: Camera, config: Config) -> None:
    estimator = PoseEstimator(config, camera)
    rgb = np.zeros((config.camera_height, config.camera_width, 3), dtype=np.uint8)
    depth = np.ones((config.camera_height, config.camera_width), dtype=np.float32)
    detection = {"class": "cube", "confidence": 1.0, "bbox": (0, 0, 1, 1), "mask": None}
    pose = estimator.estimate_pose(rgb, depth, detection)
    assert pose is None or {"pos", "orn", "confidence"} <= set(pose)
