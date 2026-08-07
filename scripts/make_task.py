#!/usr/bin/env python3
"""Generate a Booster BeyondMimic task for a prepared robot motion.

The command renders ``overlay/train/task_template`` into the upstream
Booster Train task tree, registers the generated package, and prints the
corresponding training command. It reads motion duration from the existing
GMR robot-motion pickle and never starts training.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expressive_motion import booster_paths, task_generation  # noqa: E402
from expressive_motion.pipeline_config import PipelineConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _motion_metadata(args: argparse.Namespace) -> tuple[dict, Path]:
    """Build task metadata from the two verified robot-pickle fields."""
    if not args.clip:
        raise task_generation.TaskGenerationError("--clip is required")

    clip = str(args.clip)
    if Path(clip).name != clip or clip in (".", ".."):
        raise task_generation.TaskGenerationError(f"clip must be a name, not a path: {clip!r}")
    path = (
        Path(args.output_root)
        / clip
        / "robot_data"
        / args.robot
        / f"{clip}_{args.robot}.pkl"
    )
    if not path.is_file():
        raise task_generation.TaskGenerationError(
            f"robot-motion pickle not found: {path}\n"
            "Run the retarget stage first, or check --clip, --robot and --output-root."
        )

    try:
        with path.open("rb") as handle:
            motion = pickle.load(handle)
    except (OSError, EOFError, ImportError, pickle.UnpicklingError) as exc:
        raise task_generation.TaskGenerationError(f"cannot read robot-motion pickle {path}: {exc}") from exc

    if not isinstance(motion, dict):
        raise task_generation.TaskGenerationError(f"robot-motion pickle must contain a dictionary: {path}")

    fps_raw = motion.get("fps")
    try:
        fps = float(fps_raw)
    except (TypeError, ValueError) as exc:
        raise task_generation.TaskGenerationError(f"robot-motion pickle has invalid fps: {fps_raw!r}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise task_generation.TaskGenerationError(f"robot-motion pickle has invalid fps: {fps_raw!r}")

    root_pos = motion.get("root_pos")
    shape = getattr(root_pos, "shape", None)
    if shape is None or len(shape) != 2 or shape[1] != 3 or shape[0] <= 0:
        raise task_generation.TaskGenerationError(
            f"robot-motion pickle root_pos must have non-empty shape (frames, 3), got {shape!r}"
        )

    frames = int(shape[0])
    return {
        "clip": clip,
        "robot": args.robot,
        "duration_s": frames / fps,
    }, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip",
        required=True,
        help="clip name; reads outputs/<clip>/robot_data/<robot>/<clip>_<robot>.pkl",
    )
    parser.add_argument("--robot", default="booster_k1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a config leaf, e.g. --set train.save_interval=200")
    parser.add_argument("--booster-train", default=None,
                        help="override the booster_train checkout location")
    parser.add_argument("--motion-file", default=None,
                        help="motion path relative to BOOSTER_ASSETS_DIR; "
                             "defaults to <subpaths.motions_dir>/<clip>.npz")
    parser.add_argument("--episode-length-s", type=float, default=None,
                        help="override the episode length; default is the motion duration")
    parser.add_argument("--num-envs", type=int, default=4096, help="for the printed launch command only")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="for the printed launch command only; default train.max_iterations")
    parser.add_argument("--force", action="store_true", help="regenerate a task this tool generated")
    parser.add_argument("--no-register", action="store_true",
                        help="do not append the import line to the task registry")
    parser.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    args = parser.parse_args()

    try:
        config = PipelineConfig.load(args.pipeline_config, args.overrides)
        train_cfg = config.section("train")

        meta, source_path = _motion_metadata(args)
        clip = str(meta["clip"])
        meta_robot = str(meta.get("robot", args.robot))
        if meta_robot != args.robot:
            raise task_generation.TaskGenerationError(
                f"metadata robot is {meta_robot!r}, but --robot is {args.robot!r}"
            )

        path_overrides = list(args.overrides)
        if args.booster_train:
            path_overrides.append(f"paths.booster_train={args.booster_train}")
        booster = booster_paths.resolve_paths(
            config=config,
            overrides=path_overrides,
            paths_file=args.paths_config,
        )
        booster_train = booster.booster_train
        task_root = task_generation.subpath("task_root", args.paths_config)
        registry_rel = task_generation.subpath("task_registry", args.paths_config)
        train_script = task_generation.subpath("train_script", args.paths_config)
        motions_dir = task_generation.subpath("motions_dir", args.paths_config)

        motion_file = args.motion_file or f"{motions_dir}/{clip}.npz"
        motion_rel = Path(motion_file)
        if (
            not motion_file
            or motion_rel.is_absolute()
            or ".." in motion_rel.parts
            or motion_rel.suffix.lower() != ".npz"
        ):
            raise task_generation.TaskGenerationError(
                "motion file must be an .npz path inside the Booster Assets checkout"
            )
        installed_motion = booster.booster_assets / motion_rel
        if not installed_motion.is_file():
            raise task_generation.TaskGenerationError(
                f"installed training motion not found: {installed_motion}\n"
                "Run the conversion/install stage before generating the task."
            )
        robot_config = ROOT / "configs" / "robots" / f"{args.robot}.json"

        max_iterations = (
            args.max_iterations
            if args.max_iterations is not None
            else int(train_cfg.get("max_iterations", 30000))
        )
        names = task_generation.task_names(clip, meta_robot, train_cfg)
        command = task_generation.launch_command(
            booster_train, train_script, names["gym_id"], args.num_envs, max_iterations
        )
        if not args.no_register:
            task_generation.register_task(
                booster_train / registry_rel, names["task_name"], dry_run=True
            )

        result = task_generation.generate_task(
            meta,
            booster_train=booster_train,
            task_root=task_root,
            motion_file=motion_file,
            robot_config=robot_config,
            train_cfg=train_cfg,
            source_motion=source_path,
            episode_length=args.episode_length_s,
            force=args.force,
            dry_run=args.dry_run,
        )

        registered = None
        if not args.no_register:
            registered = task_generation.register_task(
                booster_train / registry_rel, result["task_name"], dry_run=args.dry_run
            )

    except (
        booster_paths.BoosterPathError,
        task_generation.TaskGenerationError,
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    prefix = "would write" if args.dry_run else "wrote"
    print(f"task        {result['task_name']}")
    print(f"gym id      {result['gym_id']}")
    print(f"experiment  {result['experiment_name']}")
    print(f"motion      {motion_file}")
    print(f"episode     {result['episode_length_s']} s")
    print(f"{prefix:11} {result['task_dir']}")
    if registered is not None:
        state = "added" if registered else "already present"
        print(f"registry    {state}: from . import {result['task_name']}")
    print()
    print("Training is not launched. Run it yourself:")
    print(f"  {command}")

    if args.dry_run:
        print()
        print(json.dumps(result["marker"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
