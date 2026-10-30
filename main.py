"""Entry point for the Franka pick-and-place simulation."""

import argparse
import logging
import sys

from config import Config
from src.grasp_planner import GraspPlanner
from src.motion_planner import MotionPlanner
from src.perception import Camera, ObjectDetector, PoseEstimator
from src.robot_controller import RobotController
from src.simulator import PyBulletSimulator
from src.task import TaskManager

logger = logging.getLogger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Franka pick-and-place from clutter in PyBullet"
    )
    parser.add_argument(
        "--num-picks", type=int, default=5, help="Number of pick attempts"
    )
    parser.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the PyBullet GUI",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI (overrides --gui)"
    )
    parser.add_argument(
        "--run-name", type=str, default="run", help="Label used in log output"
    )
    return parser.parse_args(argv)


def build_task_manager(
    config: Config, gui: bool
) -> tuple[PyBulletSimulator, TaskManager]:
    """Wire up the simulator and all planning/control modules."""
    simulator = PyBulletSimulator(config, gui=gui)
    simulator.setup_scene()
    camera = Camera(config, simulator)
    detector = ObjectDetector(config)
    pose_estimator = PoseEstimator(config, camera)
    grasp_planner = GraspPlanner(config, simulator)
    motion_planner = MotionPlanner(config, simulator)
    robot_controller = RobotController(config, simulator)
    task_manager = TaskManager(
        simulator,
        camera,
        detector,
        pose_estimator,
        grasp_planner,
        motion_planner,
        robot_controller,
        config,
    )
    return simulator, task_manager


def run(args: argparse.Namespace) -> int:
    """Run the requested number of picks and report metrics."""
    config = Config()
    gui = args.gui and not args.headless
    simulator, task_manager = build_task_manager(config, gui)
    results: list[dict] = []
    try:
        for pick in range(args.num_picks):
            logger.info("[%s] Pick %d/%d", args.run_name, pick + 1, args.num_picks)
            success, data = task_manager.execute_pick_and_place()
            results.append(data)
            logger.info(
                "[%s] Pick %d %s (perceive %.2fs, plan %.2fs, execute %.2fs)",
                args.run_name,
                pick + 1,
                "succeeded" if success else f"failed ({data['error']})",
                data["perception_time"],
                data["planning_time"],
                data["execution_time"],
            )
    finally:
        simulator.disconnect()
    report(args.run_name, results)
    return 0 if any(r["success"] for r in results) else 1


def report(run_name: str, results: list[dict]) -> None:
    """Log aggregate success rate and timing metrics."""
    if not results:
        return
    successes = sum(r["success"] for r in results)
    total_time = sum(
        r["perception_time"] + r["planning_time"] + r["execution_time"]
        for r in results
    )
    logger.info(
        "[%s] Summary: %d/%d picks succeeded (%.0f%%), total %.1fs",
        run_name,
        successes,
        len(results),
        100.0 * successes / len(results),
        total_time,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
