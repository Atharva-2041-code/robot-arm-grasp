# Robot Grasping — Franka Pick-and-Place in PyBullet

Minimal autonomous pick-and-place from clutter with a Franka Emika Panda
(7 DOF), a fixed eye-to-hand camera, top-down grasp planning, and a
joint-space RRT motion planner. Uses only PyBullet's built-in assets
(`pybullet_data`) — no downloads needed.

## Quick start

```bash
cd robot_grasping
source .robot_venv/bin/activate      # existing project venv
pip install -r requirements.txt
python main.py --num-picks 1 --gui
```

## How to run

```bash
python main.py --num-picks 5              # 5 picks with GUI
python main.py --num-picks 3 --headless   # no GUI (CI / servers)
python main.py --num-picks 2 --run-name exp1
```

The run logs per-pick timing (perception / planning / execution) and a
final success-rate summary. Exit code 0 means at least one pick succeeded.

## Modules

| File | Purpose |
| --- | --- |
| `config.py` | Single flat `Config` dataclass with every tunable parameter |
| `main.py` | CLI entry point, wiring, and metrics reporting |
| `src/simulator.py` | `PyBulletSimulator` — physics client, scene loading, state queries, camera rendering |
| `src/perception.py` | `Camera` (rendering + back-projection) plus `ObjectDetector` / `PoseEstimator` skeletons |
| `src/grasp_planner.py` | `GraspPlanner` — top-down grasp candidates, clearance-based scoring |
| `src/motion_planner.py` | `MotionPlanner` — IK + joint-space RRT with shortcut smoothing |
| `src/robot_controller.py` | `RobotController` — trajectory tracking and gripper open/close |
| `src/task.py` | `TaskManager` — perceive → plan grasp → plan motion → grasp → place |

## Perception skeletons

`ObjectDetector.detect()` and `PoseEstimator.estimate_pose()` are stubs
for you to plug in your own ML models (see their docstrings for the
expected input/output contract). Until then, the task manager falls back
to ground-truth object poses from the simulator so the full pipeline runs
end to end.

## Tests

```bash
pytest tests/ -v
```

Tests run headless (`DIRECT` mode) and cover the simulator, camera,
perception interfaces, and both planners.
