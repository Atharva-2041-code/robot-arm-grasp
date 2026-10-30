"""Tests for the PyBullet simulator wrapper."""

import numpy as np
import pytest

from config import Config
from src.perception import Camera
from src.simulator import PyBulletSimulator


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(num_objects=2, settle_steps=60)


@pytest.fixture(scope="module")
def simulator(config: Config) -> PyBulletSimulator:
    sim = PyBulletSimulator(config, gui=False)
    sim.setup_scene()
    yield sim
    sim.disconnect()


def test_initialization(simulator: PyBulletSimulator) -> None:
    assert simulator.client_id >= 0
    assert simulator.robot_id >= 0


def test_scene_setup(simulator: PyBulletSimulator, config: Config) -> None:
    assert simulator.table_id >= 0
    assert simulator.tray_id >= 0
    assert len(simulator.objects) == config.num_objects


def test_physics_steps(simulator: PyBulletSimulator, config: Config) -> None:
    before = simulator.step_count
    simulator.step_n(10)
    assert simulator.step_count == before + 10
    assert simulator.simulation_time == pytest.approx(
        simulator.step_count * config.time_step
    )


def test_robot_state_query(simulator: PyBulletSimulator, config: Config) -> None:
    state = simulator.get_robot_state()
    assert len(state["joint_positions"]) == len(config.arm_joint_indices)
    assert len(state["ee_position"]) == 3
    assert len(state["ee_orientation"]) == 4


def test_set_robot_config(simulator: PyBulletSimulator, config: Config) -> None:
    simulator.set_robot_config(list(config.home_config))
    state = simulator.get_robot_state()
    assert np.allclose(state["joint_positions"], config.home_config, atol=1e-6)
    with pytest.raises(ValueError):
        simulator.set_robot_config([0.0, 0.0])


def test_object_state_query(simulator: PyBulletSimulator, config: Config) -> None:
    pos, orn = simulator.get_object_state(simulator.objects[0])
    assert len(pos) == 3
    assert len(orn) == 4
    assert pos[2] > 0.0


def test_collision_check(simulator: PyBulletSimulator) -> None:
    assert not simulator.check_collision(simulator.robot_id, simulator.plane_id)
    simulator.step_n(simulator.config.settle_steps)
    resting_on_table = any(
        simulator.check_collision(obj_id, simulator.table_id)
        for obj_id in simulator.objects
    )
    assert resting_on_table


def test_camera_image(simulator: PyBulletSimulator, config: Config) -> None:
    camera = Camera(config, simulator)
    rgb, depth_buffer = simulator.get_camera_image(
        camera.view_matrix, camera.proj_matrix
    )
    assert rgb.shape == (config.camera_height, config.camera_width, 3)
    assert rgb.dtype == np.uint8
    assert depth_buffer.shape == (config.camera_height, config.camera_width)


def test_save_restore_state(simulator: PyBulletSimulator) -> None:
    pos_before, _ = simulator.get_object_state(simulator.objects[0])
    state_id = simulator.save_state()
    simulator.step_n(100)
    simulator.restore_state(state_id)
    pos_after, _ = simulator.get_object_state(simulator.objects[0])
    assert np.allclose(pos_before, pos_after, atol=1e-9)
