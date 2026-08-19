#!/usr/bin/env python3
"""Prepare Booster reference motions from one or more input videos.

Each video is staged at 30 fps, processed by GVHMR, retargeted with GMR, and
converted with the upstream GMR and Booster tools. Existing stage outputs are
reused unless ``--force`` is passed, and failures are reported per input
video.

This command stops after writing the Booster Assets CSV and NPZ. It never
starts training or deployment.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import filecmp
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expressive_motion import booster_paths, video_processing  # noqa: E402
from expressive_motion.pipeline_config import PipelineConfig  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
GVHMR = ROOT / "external" / "GVHMR"
GMR = ROOT / "external" / "GMR"
GMR_PKL_TO_CSV = GMR / "scripts" / "batch_gmr_pkl_to_csv.py"
CONDA_FILE = ROOT / ".install" / "conda_path"


def _env_name(variable: str, state_file: str, default: str) -> str:
    """Resolve a conda environment name: EM_* override, installer record, default."""
    explicit = os.environ.get(variable)
    if explicit:
        return explicit
    recorded = ROOT / ".install" / state_file
    if recorded.is_file():
        value = recorded.read_text(encoding="utf-8").strip()
        if value:
            return value
    return default


GVHMR_ENV = _env_name("EM_GVHMR_ENV", "gvhmr_env", "expressive-motion-gvhmr")
GMR_ENV = _env_name("EM_GMR_ENV", "gmr_env", "expressive-motion-gmr")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}


@dataclass
class ClipResult:
    name: str
    source: str
    status: str = "pending"
    stage: str = ""
    error: str = ""
    seconds: float = 0.0


def safe_name(value: str) -> str:
    import hashlib

    cleaned = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in value
    ).strip("._") or "motion"
    if len(cleaned) > 48:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:39]}_{digest}"
    return cleaned


def conda_path() -> Path:
    if CONDA_FILE.is_file():
        value = CONDA_FILE.read_text(encoding="utf-8").strip()
        if value:
            return Path(value)
    found = shutil.which("conda")
    if found:
        return Path(found)
    raise SystemExit("Conda not found. Run bash install.sh first.")


def isaac_command(conda: Path) -> list[str]:
    """Return the explicitly configured Isaac-capable Python command."""
    explicit = os.environ.get("EM_ISAAC_PYTHON")
    if explicit:
        python = Path(explicit).expanduser().resolve()
        if not python.is_file():
            raise SystemExit(f"EM_ISAAC_PYTHON is not a file: {python}")
        return [str(python)]

    environment = os.environ.get("EM_ISAAC_ENV")
    if environment:
        return [str(conda), "run", "--no-capture-output", "-n", environment, "python"]

    raise SystemExit(
        "Booster CSV-to-NPZ conversion requires an Isaac-capable Python. "
        "Set EM_ISAAC_PYTHON=/path/to/python or EM_ISAAC_ENV=<conda-env>."
    )


def control_rate_hz(robot: str) -> int:
    """Derive the integer Booster output rate from the robot control config."""
    path = ROOT / "configs" / "robots" / f"{robot}.json"
    if not path.is_file():
        raise SystemExit(f"Robot config not found: {path}")
    control = json.loads(path.read_text(encoding="utf-8")).get("control", {})
    decimation = control.get("decimation")
    sim_dt = control.get("sim_dt_s")
    if not decimation or not sim_dt:
        raise SystemExit(f"{path} has no control.decimation/control.sim_dt_s")

    derived = 1.0 / (float(decimation) * float(sim_dt))
    stated = control.get("control_rate_hz")
    if stated is not None and abs(float(stated) - derived) > 1e-6:
        raise SystemExit(
            f"{path} is inconsistent: control_rate_hz={stated}, derived={derived:.6f}"
        )
    rounded = round(derived)
    if abs(derived - rounded) > 1e-6:
        raise SystemExit(
            f"Derived control rate {derived:.6f} is not an integer; upstream "
            "csv_to_npz.py accepts only integer --output_fps."
        )
    return int(rounded)


def run_logged(
    command: list[str],
    cwd: Path,
    log_file: Path,
    timeout: float | None = None,
    stream: bool = True,
) -> None:
    """Run a stage, recording its exact command and combined output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("wb") as log:
        log.write(f"$ {shlex.join(command)}\n\n".encode())
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None

        fd = process.stdout.fileno()
        try:
            while True:
                chunk = os.read(fd, 8192)
                if not chunk:
                    break
                log.write(chunk)
                if stream:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            returncode = process.wait(timeout=timeout)
        finally:
            process.stdout.close()

    if returncode != 0:
        tail = ""
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = " | ".join(line for line in lines[-4:] if line.strip())
        except OSError:
            pass
        raise RuntimeError(f"exit {returncode}: {tail}")


def discover(inputs: list[Path], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        item = item.expanduser()
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES:
            found.append(item.resolve())
        elif item.is_dir():
            walker = item.rglob("*") if recursive else item.glob("*")
            found.extend(
                path.resolve()
                for path in sorted(walker)
                if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
            )

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique



def process_clip(
    video: Path,
    robot: str,
    output_root: Path,
    config: PipelineConfig,
    conda: Path,
    isaac_prefix: list[str],
    booster: booster_paths.BoosterPaths,
    output_fps: int,
    force: bool,
    config_path: Path | None,
    overrides: list[str],
    stream: bool = True,
) -> ClipResult:
    started = time.time()
    name = safe_name(video.stem)
    result = ClipResult(name=name, source=str(video))

    clip_dir = output_root / name
    gvhmr_dir = clip_dir / "gvhmr"
    logs_dir = clip_dir / "logs"
    data_dir = clip_dir / "robot_data" / robot
    for directory in (gvhmr_dir, logs_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        result.stage = "probe"
        info = video_processing.probe(video)
        min_duration = float(config.get("video.min_duration_s", 0.7))
        max_duration = float(config.get("video.max_duration_s", 600.0))
        if info.duration and not (min_duration <= info.duration <= max_duration):
            result.status = "skipped"
            result.error = (
                f"duration {info.duration:.2f}s outside "
                f"[{min_duration}, {max_duration}]"
            )
            return result

        # The manual upstream workflow is 30 fps at both the GVHMR and GMR CSV
        # boundaries. Always stage that input; native-rate mode is not accepted.
        result.stage = "resample"
        staged = clip_dir / "input" / f"{name}.mp4"
        if (
            force
            or not staged.is_file()
            or video.stat().st_mtime_ns > staged.stat().st_mtime_ns
        ):
            video_processing.resample_to_30(
                video, staged, bool(config.get("video.resample_interpolate", False))
            )
        staged_info = video_processing.probe(staged)
        if abs(staged_info.fps - 30.0) > 0.01:
            raise RuntimeError(
                f"30 fps staging failed: {staged} probes as {staged_info.fps:.6f} fps"
            )

        result.stage = "gvhmr"
        gvhmr_result = gvhmr_dir / name / "hmr4d_results.pt"

        def gvhmr_command() -> list[str]:
            command = [
                str(conda), "run", "--no-capture-output", "-n", GVHMR_ENV,
                "python", "tools/demo/demo.py",
                f"--video={staged}",
                f"--output_root={gvhmr_dir}",
                "-s",
            ]
            return command

        if force or not gvhmr_result.is_file():
            run_logged(
                gvhmr_command(),
                GVHMR,
                logs_dir / "gvhmr.log",
                stream=stream,
            )
        if not gvhmr_result.is_file():
            raise RuntimeError("GVHMR produced no hmr4d_results.pt")

        result.stage = "retarget"
        motion_file = data_dir / f"{name}_{robot}.pkl"
        if force or not motion_file.is_file():
            command = [
                str(conda), "run", "--no-capture-output", "-n", GMR_ENV,
                "python", str(ROOT / "scripts" / "retarget_gvhmr.py"),
                "--gvhmr-result", str(gvhmr_result),
                "--robot", robot,
                "--output", str(motion_file),
                "--body-models", str(GMR / "assets" / "body_models"),
                "--fps", "30",
                "--clip-name", name,
            ]
            if config_path:
                command += ["--pipeline-config", str(config_path)]
            for item in overrides:
                command += ["--set", item]
            run_logged(command, ROOT, logs_dir / f"retarget_{robot}.log", stream=stream)
        if not motion_file.is_file():
            raise RuntimeError("retargeting produced no motion pickle")

        result.stage = "pkl_to_csv"
        gmr_csv = data_dir / "csv" / f"{name}_{robot}.csv"
        if (
            force
            or not gmr_csv.is_file()
            or motion_file.stat().st_mtime_ns > gmr_csv.stat().st_mtime_ns
        ):
            run_logged(
                [
                    str(conda), "run", "--no-capture-output", "-n", GMR_ENV,
                    "python", str(GMR_PKL_TO_CSV),
                    "--folder", str(data_dir),
                ],
                ROOT,
                logs_dir / f"pkl_to_csv_{robot}.log",
                stream=stream,
            )
        if not gmr_csv.is_file():
            raise RuntimeError(f"GMR pickle-to-CSV produced no file: {gmr_csv}")

        result.stage = "install_csv"
        booster.motions_dir.mkdir(parents=True, exist_ok=True)
        asset_csv = booster.motions_dir / f"{name}.csv"
        if force or not asset_csv.is_file() or not filecmp.cmp(gmr_csv, asset_csv, shallow=False):
            shutil.copy2(gmr_csv, asset_csv)

        result.stage = "csv_to_npz"
        asset_npz = booster.motions_dir / f"{name}.npz"
        if (
            force
            or not asset_npz.is_file()
            or asset_csv.stat().st_mtime_ns > asset_npz.stat().st_mtime_ns
        ):
            run_logged(
                isaac_prefix
                + [
                    str(booster.csv_to_npz_script),
                    "--headless",
                    f"--input_file={asset_csv}",
                    "--input_fps=30",
                    f"--output_name={asset_npz}",
                    f"--output_fps={output_fps}",
                ],
                booster.booster_train,
                logs_dir / f"csv_to_npz_{robot}.log",
                stream=stream,
            )
        if not asset_npz.is_file():
            raise RuntimeError(f"Booster CSV-to-NPZ produced no file: {asset_npz}")

        result.stage = "complete"
        result.status = "ok"

    except subprocess.TimeoutExpired:
        result.status = "failed"
        result.error = f"timeout during {result.stage}"
    except Exception as exc:  # one broken clip must not stop the batch
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"[:400]

    result.seconds = round(time.time() - started, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Video files or directories.")
    parser.add_argument("--robot", default="booster_k1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute existing stages.")
    parser.add_argument("--workers", type=int, default=None, help="Parallel clip workers.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe inputs only; do not require Isaac or run any pipeline stage.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not mirror stage output to the terminal.",
    )
    args = parser.parse_args()

    config = PipelineConfig.load(args.pipeline_config, args.overrides)
    conda = conda_path()
    if not GVHMR.is_dir():
        raise SystemExit(f"GVHMR repository is missing: {GVHMR}")
    if not GMR.is_dir() or not GMR_PKL_TO_CSV.is_file():
        raise SystemExit(f"GMR pickle-to-CSV script is missing: {GMR_PKL_TO_CSV}")

    videos = discover(args.inputs, args.recursive)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit("No videos found.")

    print(f"Found {len(videos)} video(s); robot={args.robot}; pipeline input=30 fps")
    if args.dry_run:
        for video in videos:
            try:
                info = video_processing.probe(video)
                print(
                    f"  {video}  {info.fps:.3f} fps  {info.frame_count} frames  "
                    f"{info.duration:.2f}s -> staged at 30 fps"
                )
            except Exception as exc:
                print(f"  {video}  PROBE FAILED: {exc}")
        return

    try:
        booster = booster_paths.resolve_paths(
            config=config,
            overrides=args.overrides,
            verify=False,
        )
    except booster_paths.BoosterPathError as exc:
        raise SystemExit(str(exc)) from exc
    if not booster.booster_train.is_dir() or not booster.csv_to_npz_script.is_file():
        raise SystemExit(
            "Booster Train is missing or incomplete: "
            f"{booster.csv_to_npz_script}. Initialize the submodules or set "
            "EM_BOOSTER_ROOT/--set paths.booster_train."
        )
    if not booster.booster_assets.is_dir():
        raise SystemExit(
            "Booster Assets is missing: "
            f"{booster.booster_assets}. Initialize the submodules or set "
            "EM_BOOSTER_ROOT/--set paths.booster_assets."
        )

    isaac_prefix = isaac_command(conda)
    output_fps = control_rate_hz(args.robot)
    print(
        f"Booster conversion: input=30 fps, output={output_fps} fps, "
        f"motions={booster.motions_dir}"
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    workers = max(1, args.workers or int(config.get("batch.gvhmr_workers", 1)))
    results: list[ClipResult] = []
    started = time.time()

    def work(video: Path) -> ClipResult:
        return process_clip(
            video,
            args.robot,
            args.output_root,
            config,
            conda,
            isaac_prefix,
            booster,
            output_fps,
            args.force,
            args.pipeline_config,
            args.overrides,
            stream=not args.quiet and workers == 1,
        )

    if workers == 1:
        for index, video in enumerate(videos, 1):
            print(f"\n[{index}/{len(videos)}] {safe_name(video.stem)} <- {video.name}", flush=True)
            outcome = work(video)
            results.append(outcome)
            print(
                f"[{index}/{len(videos)}] {outcome.name}: {outcome.status} "
                f"stage={outcome.stage} {outcome.seconds:.1f}s {outcome.error}".rstrip()
            )
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(work, video): video for video in videos}
            for index, future in enumerate(futures.as_completed(pending), 1):
                outcome = future.result()
                results.append(outcome)
                print(
                    f"[{index}/{len(videos)}] {outcome.name}: {outcome.status} "
                    f"stage={outcome.stage} {outcome.seconds:.1f}s {outcome.error}".rstrip()
                )

    results.sort(key=lambda item: item.name)
    ok = sum(result.status == "ok" for result in results)
    failed = sum(result.status == "failed" for result in results)
    skipped = sum(result.status == "skipped" for result in results)

    print("\n" + "=" * 62)
    print(f"processed {len(results)} clip(s) in {time.time() - started:.1f}s")
    print(f"  ok {ok}   failed {failed}   skipped {skipped}")
    print(f"  Booster motions: {booster.motions_dir}")
    if failed:
        print("\nfailures:")
        for result in results:
            if result.status == "failed":
                print(f"  {result.name} [{result.stage}] {result.error}")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
