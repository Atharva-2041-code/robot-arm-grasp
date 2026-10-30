"""Tests for the grasp planner and motion planner."""

import numpy as np
import pytest

from config import Config
from src.grasp_planner import GraspPlanner
from src.motion_planner import MotionPlanner
from src.simulator import PyBulletSimulator

GRASP_KEYS = {
    "gripper_position",
    "gripper_orientation",
    "grip_width",
    "quality_score",
    "approach_direction",
    "approach_distance",
}


@pytest.fixture(scope="module")
def config() -> Config:
    return Config(num_objects=2, settle_steps=60)


@pytest.fixture(scope="module")
def simulator(config: Config) -> PyBulletSimulator:
    sim = PyBulletSimulator(config, gui=False)
    sim.setup_scene()
    yield sim
    sim.disconnect()


@pytest.fixture(scope="module")
def grasp_planner(config: Config, simulator: PyBulletSimulator) -> GraspPlanner:
    return GraspPlanner(config, simulator)


@pytest.fixture(scope="module")
def motion_planner(config: Config, simulator: PyBulletSimulator) -> MotionPlanner:
    return MotionPlanner(config, simulator)


def test_grasp_plan_structure(
    grasp_planner: GraspPlanner, simulator: PyBulletSimulator
) -> None:
    obj_pos, obj_orn = simulator.get_object_state(simulator.objects[0])
    grasp = grasp_planner.plan_grasp(obj_pos, obj_orn)
    assert grasp is not None
    assert GRASP_KEYS <= set(grasp)
    assert grasp["gripper_position"][2] > obj_pos[2]
    assert grasp["grip_width"] > 0.0


def test_grasp_below_table_rejected(grasp_planner: GraspPlanner) -> None:
    grasp = grasp_planner.plan_grasp((0.5, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    assert grasp is None


def test_inverse_kinematics(
    motion_planner: MotionPlanner, config: Config
) -> None:
    solution = motion_planner.inverse_kinematics(
        (0.4, 0.0, 0.9), (1.0, 0.0, 0.0, 0.0)
    )
    assert solution is not None
    assert len(solution) == len(config.arm_joint_indices)


def test_collision_free_at_home(
    motion_planner: MotionPlanner, simulator: PyBulletSimulator, config: Config
) -> None:
    state_id = simulator.save_state()
    try:
        assert motion_planner._collision_free(list(config.home_config))
    finally:
        simulator.restore_state(state_id)


def test_plan_path_returns_waypoints(
    motion_planner: MotionPlanner, simulator: PyBulletSimulator, config: Config
) -> None:
    start = simulator.get_robot_state()["joint_positions"]
    path = motion_planner.plan_path(
        start, (0.4, 0.0, 0.95), (1.0, 0.0, 0.0, 0.0), planning_time=10.0
    )
    assert path is not None
    assert len(path) >= 2
    assert np.allclose(path[0], start, atol=1e-6)
    assert all(len(waypoint) == len(config.arm_joint_indices) for waypoint in path)
    # Planning must not disturb the live simulation state.
    assert np.allclose(
        simulator.get_robot_state()["joint_positions"], start, atol=1e-6
    )
