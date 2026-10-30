"""Flat configuration for the Franka pick-and-place simulation."""

from dataclasses import dataclass


@dataclass
class Config:
    """All tunable parameters for physics, robot, camera, and planners."""

    # Physics
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    time_step: float = 1.0 / 240.0
    num_sub_steps: int = 1
    settle_steps: int = 240
    # Wall-clock seconds slept per step in GUI mode = time_step * scale.
    # 1.0 = real time, 0.5 = 2x speed, 0.0 = as fast as possible.
    gui_time_scale: float = 1.0

    # Assets (relative to pybullet_data.getDataPath())
    plane_urdf: str = "plane.urdf"
    table_urdf: str = "table/table.urdf"
    tray_urdf: str = "tray/traybox.urdf"
    robot_urdf: str = "franka_panda/panda.urdf"
    object_urdfs: tuple[str, ...] = (
        "cube_small.urdf",
        "sphere_small.urdf",
        "lego/lego.urdf",
    )

    # Scene layout
    table_position: tuple[float, float, float] = (0.5, 0.0, 0.0)
    table_height: float = 0.625
    robot_base_position: tuple[float, float, float] = (0.0, 0.0, 0.625)
    tray_position: tuple[float, float, float] = (0.4, 0.45, 0.625)
    tray_scale: float = 0.6
    num_objects: int = 3
    spawn_region_min: tuple[float, float] = (0.4, -0.15)
    spawn_region_max: tuple[float, float] = (0.6, 0.15)
    spawn_height: float = 0.70

    # Robot
    arm_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    finger_joint_indices: tuple[int, int] = (9, 10)
    ee_link_index: int = 11
    home_config: tuple[float, ...] = (0.0, 0.41, 0.0, -1.84, 0.0, 2.25, 0.785)
    max_joint_velocity: float = 1.0
    max_joint_force: float = 200.0
    finger_open_width: float = 0.04
    finger_closed_width: float = 0.0
    grip_force: float = 50.0
    joint_tolerance: float = 0.02
    trajectory_timeout: float = 30.0
    gripper_settle_steps: int = 120

    # Camera (eye-to-hand, fixed above workspace)
    camera_eye: tuple[float, float, float] = (0.5, -0.6, 1.3)
    camera_target: tuple[float, float, float] = (0.5, 0.0, 0.625)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 60.0
    camera_width: int = 640
    camera_height: int = 480
    camera_near: float = 0.1
    camera_far: float = 3.0

    # Grasp planning
    grasp_num_samples: int = 8
    grasp_approach_distance: float = 0.12
    grasp_margin: float = 0.005
    grasp_depth_offset: float = 0.015
    grasp_min_clearance: float = 0.04

    # Motion planning (RRT)
    rrt_max_iterations: int = 2000
    rrt_step_size: float = 0.15
    rrt_goal_bias: float = 0.2
    rrt_goal_tolerance: float = 0.1
    smoothing_iterations: int = 40
    ik_max_iterations: int = 100
    ik_residual_threshold: float = 1e-4

    # Task
    lift_height: float = 0.15
    place_height_offset: float = 0.25
    place_success_radius: float = 0.2
