"""Generate a Booster BeyondMimic task package from a prepared motion.

The generator renders the local task template into the upstream Booster Train
task tree and registers it through the K1 task registry. Generated task
configuration references the prepared NPZ motion and inherits actuator
parameters from the pinned upstream Booster configuration.

This module returns or prints training commands but never launches training.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import os
import re
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent

GENERATED_MARKER = ".expressive_motion_generated.json"
GENERATOR_VERSION = "1.0.0"
GENERATOR_ID = "expressive_motion/task_generation.py"

TEMPLATE_FILES = {
    "__init__.py.j2": "__init__.py",
    "env_cfg.py.j2": "env_cfg.py",
    "tracking_env_cfg.py.j2": "tracking_env_cfg.py",
    "ppo_cfg.py.j2": "ppo_cfg.py",
}

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class TaskGenerationError(RuntimeError):
    """Anything that should stop generation with one readable message."""


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def render_text(text: str, variables: dict[str, Any]) -> str:
    """Substitute ``{{ name }}`` placeholders. Unknown names are an error.

    Deliberately a strict subset of Jinja: the templates are upstream Python
    with single braces everywhere (f-strings, dict literals), and a plain
    substitution cannot misread them.
    """
    missing: list[str] = []

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name not in variables:
            missing.append(name)
            return match.group(0)
        return str(variables[name])

    out = _PLACEHOLDER.sub(repl, text)
    if missing:
        raise TaskGenerationError(f"template referenced unknown variables: {sorted(set(missing))}")
    return out


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_booster_train(
    paths_config: Path | None = None,
    override: str | None = None,
    *,
    config: Any = None,
    overrides: list[str] | None = None,
) -> Path:
    """Locate the booster_train checkout.

    Uses ``expressive_motion.booster_paths`` when available so ``--set paths.*``,
    ``EM_BOOSTER_ROOT`` and pinned-commit validation have the same semantics as
    the conversion and task-generation stages.
    """
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise TaskGenerationError(f"--booster-train does not exist: {path}")
        return path

    try:  # pragma: no cover - fallback only matters before that branch is merged
        from . import booster_paths  # type: ignore
    except ImportError:
        return _resolve_from_paths_json(paths_config)

    try:
        resolved = booster_paths.resolve_paths(
            config=config,
            overrides=overrides,
            paths_file=paths_config,
        )
    except booster_paths.BoosterPathError as exc:
        raise TaskGenerationError(str(exc)) from exc
    return resolved.booster_train


def _resolve_from_paths_json(paths_config: Path | None = None) -> Path:
    source = Path(paths_config) if paths_config else ROOT / "configs" / "paths.json"
    if not source.is_file():
        raise TaskGenerationError(f"paths config not found: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))

    env_root = os.environ.get("EM_BOOSTER_ROOT")
    if env_root:
        path = Path(env_root).expanduser() / "booster_train"
    else:
        raw = Path(str(data.get("booster_train", "")))
        path = raw if raw.is_absolute() else ROOT / raw

    path = path.resolve()
    if not path.is_dir():
        raise TaskGenerationError(
            f"booster_train checkout not found at {path}\n"
            "Set it in configs/paths.json, export EM_BOOSTER_ROOT, or pass --booster-train."
        )
    return path


def subpath(key: str, paths_config: Path | None = None) -> str:
    source = Path(paths_config) if paths_config else ROOT / "configs" / "paths.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    value = data.get("subpaths", {}).get(key)
    if not value:
        raise TaskGenerationError(f"configs/paths.json has no subpaths.{key}")
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise TaskGenerationError(
            f"configs/paths.json subpaths.{key} must stay inside its Booster checkout"
        )
    return str(path)


# --------------------------------------------------------------------------- #
# naming
# --------------------------------------------------------------------------- #

def _name_fields(clip: str, task_name: str, robot: str) -> dict[str, str]:
    return {
        "clip": clip,
        "Clip": clip[:1].upper() + clip[1:],
        "task": task_name,
        "Task": task_name[:1].upper() + task_name[1:],
        "robot": robot,
    }


def task_names(clip: str, robot: str, train_cfg: dict) -> dict[str, str]:
    """Task directory name, gym id and experiment name from the config patterns."""
    fields = _name_fields(clip, clip, robot)
    try:
        task_name = str(train_cfg.get("task_name_pattern", "{clip}")).format(**fields)
    except KeyError as exc:
        raise TaskGenerationError(f"train.task_name_pattern uses unknown field {exc}") from exc

    if not task_name.isidentifier() or keyword.iskeyword(task_name):
        raise TaskGenerationError(
            f"task name {task_name!r} is not a valid Python module name; "
            "it becomes a package imported by the task registry"
        )

    fields = _name_fields(clip, task_name, robot)
    try:
        gym_id = str(train_cfg.get("gym_id_pattern", "Booster-K1-{Clip}-v0")).format(**fields)
        experiment = str(train_cfg.get("experiment_name_pattern", "k1_{clip}")).format(**fields)
    except KeyError as exc:
        raise TaskGenerationError(f"train.*_pattern uses unknown field {exc}") from exc

    return {"task_name": task_name, "gym_id": gym_id, "experiment_name": experiment}


def episode_length_s(meta: dict, train_cfg: dict, override: float | None = None) -> float:
    """Return the configured episode length or the motion duration rounded to 0.1 s."""
    if override is not None:
        value = float(override)
        if not math.isfinite(value) or value <= 0:
            raise TaskGenerationError(f"episode length must be positive and finite, got {override!r}")
        return round(value, 3)

    source = train_cfg.get("episode_length_source", "motion_duration")
    if source != "motion_duration":
        try:
            value = float(source)
        except (TypeError, ValueError):
            raise TaskGenerationError(
                f"train.episode_length_source must be 'motion_duration' or a number, got {source!r}"
            )
        if not math.isfinite(value) or value <= 0:
            raise TaskGenerationError(
                f"train.episode_length_source must be positive and finite, got {source!r}"
            )
        return round(value, 3)

    duration = meta.get("duration_s")
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = math.nan
    if not math.isfinite(duration_value) or duration_value <= 0:
        raise TaskGenerationError(f"motion metadata has no usable duration_s: {duration!r}")
    return round(duration_value, 1)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def _read_marker(task_dir: Path) -> dict | None:
    marker = task_dir / GENERATED_MARKER
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskGenerationError(
            f"refusing to overwrite {task_dir}: {GENERATED_MARKER} is unreadable or invalid"
        ) from exc
    generator = data.get("generator") if isinstance(data, dict) else None
    if generator != GENERATOR_ID:
        raise TaskGenerationError(
            f"refusing to overwrite {task_dir}: {GENERATED_MARKER} is not an "
            "Expressive Motion task marker"
        )
    return data


def generate_task(
    meta: dict,
    *,
    booster_train: Path,
    task_root: str,
    motion_file: str,
    robot_config: Path,
    train_cfg: dict,
    source_motion: Path | None = None,
    template_dir: Path | None = None,
    episode_length: float | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Render the template into ``<booster_train>/<task_root>/<task_name>/``."""
    clip = str(meta["clip"])
    robot = str(meta.get("robot", "booster_k1"))
    names = task_names(clip, robot, train_cfg)
    length = episode_length_s(meta, train_cfg, episode_length)

    template_dir = Path(template_dir or ROOT / str(train_cfg.get("template", "overlay/train/task_template")))
    if not template_dir.is_dir():
        raise TaskGenerationError(f"task template directory not found: {template_dir}")

    try:
        save_interval = int(train_cfg.get("save_interval", 100))
    except (TypeError, ValueError) as exc:
        raise TaskGenerationError("train.save_interval must be a positive integer") from exc
    if save_interval <= 0:
        raise TaskGenerationError("train.save_interval must be a positive integer")

    robot_config = Path(robot_config)
    if not robot_config.is_file():
        raise TaskGenerationError(f"robot config not found: {robot_config}")

    motion_path = Path(motion_file)
    if (
        not motion_file
        or motion_path.is_absolute()
        or ".." in motion_path.parts
        or motion_path.suffix.lower() != ".npz"
    ):
        raise TaskGenerationError(
            "motion_file must be an .npz path relative to BOOSTER_ASSETS_DIR without '..'"
        )

    variables = {
        "gym_id": names["gym_id"],
        "experiment_name": names["experiment_name"],
        "motion_file": motion_file,
        "episode_length_s": length,
        "save_interval": save_interval,
    }

    task_root_path = Path(task_root)
    if task_root_path.is_absolute() or ".." in task_root_path.parts:
        raise TaskGenerationError("task_root must stay inside the booster_train checkout")
    task_dir = Path(booster_train) / task_root_path / names["task_name"]
    existing = _read_marker(task_dir) if task_dir.exists() else None

    if task_dir.exists():
        if existing is None:
            raise TaskGenerationError(
                f"refusing to write into {task_dir}\n"
                f"It exists but has no {GENERATED_MARKER}, so it was not produced by this "
                "generator and may be hand-written. Remove it or choose another task name."
            )
        if not force:
            raise TaskGenerationError(
                f"{task_dir} already exists and was generated by this tool.\n"
                "Pass --force to regenerate it."
            )

    rendered = {}
    for src_name, dst_name in TEMPLATE_FILES.items():
        src = template_dir / src_name
        if not src.is_file():
            raise TaskGenerationError(f"template file missing: {src}")
        rendered[dst_name] = render_text(src.read_text(encoding="utf-8"), variables)

    marker = {
        "generator": GENERATOR_ID,
        "version": GENERATOR_VERSION,
        "task_name": names["task_name"],
        "gym_id": names["gym_id"],
        "experiment_name": names["experiment_name"],
        "clip": clip,
        "robot": robot,
        "motion_file": motion_file,
        "episode_length_s": length,
        "save_interval": save_interval,
        "source_motion": str(source_motion) if source_motion else None,
        "robot_config_sha256": sha256_file(robot_config),
        "template": str(template_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    result = {
        "task_dir": task_dir,
        "variables": variables,
        **names,
        "episode_length_s": length,
        "marker": marker,
        "written": [],
    }
    if dry_run:
        return result

    if task_dir.exists():
        shutil.rmtree(task_dir)  # generated output only; guarded by the marker check above
    task_dir.mkdir(parents=True)

    for dst_name, text in rendered.items():
        (task_dir / dst_name).write_text(text, encoding="utf-8")
        result["written"].append(dst_name)
    (task_dir / GENERATED_MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    result["written"].append(GENERATED_MARKER)
    return result


def register_task(registry: Path, task_name: str, dry_run: bool = False) -> bool:
    """Append ``from . import <task>`` to upstream's registry, idempotently.

    Returns True when the line was added, False when it was already present.
    This file is upstream's own extension point; nothing else under the Booster
    checkouts is written.
    """
    registry = Path(registry)
    if not registry.is_file():
        raise TaskGenerationError(
            f"task registry not found: {registry}\n"
            "Check subpaths.task_registry in configs/paths.json."
        )
    line = f"from . import {task_name}"
    text = registry.read_text(encoding="utf-8")
    if any(stripped == line for stripped in (l.strip() for l in text.splitlines())):
        return False
    if dry_run:
        return True
    if text and not text.endswith("\n"):
        text += "\n"
    registry.write_text(text + line + "\n", encoding="utf-8")
    return True


def launch_command(
    booster_train: Path,
    train_script: str,
    gym_id: str,
    num_envs: int,
    max_iterations: int,
    headless: bool = True,
) -> str:
    """The exact command to start training. This module never runs it."""
    script_rel = Path(train_script)
    if script_rel.is_absolute() or ".." in script_rel.parts:
        raise TaskGenerationError("train_script must stay inside the booster_train checkout")
    script = Path(booster_train) / script_rel
    if not script.is_file():
        raise TaskGenerationError(
            f"training script not found: {script}\n"
            "Check subpaths.train_script in configs/paths.json."
        )
    if num_envs <= 0:
        raise TaskGenerationError(f"num_envs must be positive, got {num_envs}")
    if max_iterations <= 0:
        raise TaskGenerationError(f"max_iterations must be positive, got {max_iterations}")
    parts = [
        "python",
        str(script),
        "--task",
        gym_id,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(max_iterations),
    ]
    if headless:
        parts.append("--headless")
    return shlex.join(parts)
