"""Command-line entry point for the Panda pick-and-place demo.

Usage::

    python -m panda_control.main                # pick and place one cube
    python -m panda_control.main --all          # pick and place all cubes
    python -m panda_control.main --stack        # stack all cubes
    python -m panda_control.main --n-cubes 6    # spawn 6 cubes
    python -m panda_control.main --seed 42      # reproducible scene
"""
from __future__ import annotations

import argparse

import numpy as np

from panda_control.config import get as cfg
from panda_control.task_runner import TaskRunner


def parse_args() -> argparse.Namespace:
    """Build and parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Panda Pick-and-Place demo")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--one", action="store_true", help="Pick and place one cube")
    group.add_argument("--all", action="store_true", help="Pick and place all cubes")
    group.add_argument("--stack", action="store_true", help="Stack all cubes")
    parser.add_argument("--n-cubes", type=int,
                        default=cfg("task", "default_n_cubes"),
                        help="Number of cubes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (omit for random)")
    parser.add_argument("--delay", type=float,
                        default=cfg("cli", "default_delay"),
                        help="Sim delay per step (s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Phase 1 – Environment Setup & Scene Generation")
    print("=" * 60)

    runner = TaskRunner(render_mode="human", n_cubes=args.n_cubes, seed=args.seed, sim_delay=args.delay)

    cubes = runner.scene.get_cube_info()
    print(f"\nSpawned {len(cubes)} cubes:")
    for name, info in cubes.items():
        pos = info["position"]
        col = info["color"][:3]
        print(f"  {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
              f"colour=({col[0]:.2f}, {col[1]:.2f}, {col[2]:.2f})")

    overhead = runner.snapshot_overhead()
    print(f"\nOverhead camera:")
    print(f"  RGB shape  : {overhead.rgb.shape}")
    print(f"  Depth shape: {overhead.depth.shape}")
    print(f"  K (intrinsic):\n{overhead.intrinsics.K}")
    print(f"  T_cam->world (4x4):\n{overhead.extrinsics.T_cam_to_world}")

    wrist = runner.snapshot_wrist()
    print(f"\nWrist camera:")
    print(f"  RGB shape  : {wrist.rgb.shape}")
    print(f"  Depth shape: {wrist.depth.shape}")

    cu, cv = overhead.intrinsics.width // 2, overhead.intrinsics.height // 2
    world_pt = runner.pixel_to_world(cu, cv, overhead)
    print(f"\nOverhead centre pixel ({cu}, {cv}) -> world: "
          f"({world_pt[0]:.4f}, {world_pt[1]:.4f}, {world_pt[2]:.4f})")

    print("\n" + "=" * 60)
    print("Phase 2 – 3D Perception")
    print("=" * 60)

    detections = runner.perception.perceive_overhead()
    print(f"\nDetected {len(detections)} objects from overhead camera:")
    for det in detections:
        wp = det.world_position
        mc = det.mean_color
        print(f"  body_id={det.body_id:>3d}  |  "
              f"world=({wp[0]:+.4f}, {wp[1]:+.4f}, {wp[2]:+.4f})  |  "
              f"colour=({mc[0]:.2f}, {mc[1]:.2f}, {mc[2]:.2f})  |  "
              f"pixels={det.pixel_count}  |  "
              f"bbox={det.bbox.tolist()}")

    gt_positions = {name: runner.sim.get_base_position(name)
                    for name in runner.cube_names}
    body_id_map = runner.get_body_id_map()
    errors = runner.perception.validate_against_ground_truth(
        detections, gt_positions, body_id_map,
    )
    print(f"\nBack-projection error vs ground truth:")
    for name, err in errors.items():
        print(f"  {name}: {err:.6f} m")
    mean_err = np.mean(list(errors.values()))
    print(f"  mean : {mean_err:.6f} m")

    wrist_detections = runner.perception.perceive_wrist()
    print(f"\nDetected {len(wrist_detections)} objects from wrist camera:")
    for det in wrist_detections:
        wp = det.world_position
        print(f"  body_id={det.body_id:>3d}  |  "
              f"world=({wp[0]:+.4f}, {wp[1]:+.4f}, {wp[2]:+.4f})  |  "
              f"pixels={det.pixel_count}")

    print("\n" + "=" * 60)
    print("Phase 3 – Pick-and-Place")
    print("=" * 60)

    print("\nPress Enter to start the animation ...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass

    if args.stack:
        print("\nStacking all cubes...")
        results = runner.stack_all()
        for i, res in enumerate(results):
            name = res.object_name or "unknown"
            print(f"  [{i + 1}] {name}: pick={'OK' if res.pick_success else 'FAIL'}, "
                  f"place={'OK' if res.place_success else 'FAIL'}")
            if res.place_result and res.place_result.final_object_position is not None:
                op = res.place_result.final_object_position
                print(f"       final pos: ({op[0]:.4f}, {op[1]:.4f}, {op[2]:.4f})")
        placed = sum(1 for r in results if r.place_success)
        print(f"\nStacking complete: {placed}/{len(results)} cubes placed.")

    elif args.all:
        print("\nPicking and placing all cubes...")
        results = runner.pick_and_place_all()
        for i, res in enumerate(results):
            name = res.object_name or "unknown"
            slot_label = f"slot {res.slot_index + 1}" if res.slot_index is not None else f"#{i + 1}"
            print(f"  [{slot_label}] {name}: pick={'OK' if res.pick_success else 'FAIL'}, "
                  f"place={'OK' if res.place_success else 'FAIL'}")
            if res.place_result and res.place_result.final_object_position is not None:
                op = res.place_result.final_object_position
                print(f"       final pos: ({op[0]:.4f}, {op[1]:.4f}, {op[2]:.4f})")
        placed = sum(1 for r in results if r.place_success)
        print(f"\nPick-and-place complete: {placed}/{len(results)} cubes placed.")

    else:  # --one or default
        print("\nPicking and placing one cube...")
        res = runner.pick_and_place_one()
        name = res.object_name or "unknown"
        print(f"  {name}: pick={'OK' if res.pick_success else 'FAIL'}, "
              f"place={'OK' if res.place_success else 'FAIL'}")
        if res.place_result and res.place_result.final_object_position is not None:
            op = res.place_result.final_object_position
            print(f"  final pos: ({op[0]:.4f}, {op[1]:.4f}, {op[2]:.4f})")

    print("\nClose the PyBullet window or press Enter to exit.")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    runner.close()
    print("Done.")


if __name__ == "__main__":
    main()
