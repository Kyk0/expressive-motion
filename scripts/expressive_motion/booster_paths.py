"""Locate and validate the three Booster checkouts.

Resolution order for each of ``booster_train`` / ``booster_assets`` /
``booster_deploy`` (highest priority first):

1. ``--set paths.<key>=<path>``
2. ``EM_BOOSTER_ROOT`` -- parent of all three, used only for keys not set by (1)
3. the values in ``configs/paths.json``, relative to the repository root

Every checkout is checked for existence and for being at its pinned commit.
Failures are collected and raised as one actionable message.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATHS_FILE = ROOT / "configs" / "paths.json"

REPOS = ("booster_train", "booster_assets", "booster_deploy")

# Which checkout each subpath in configs/paths.json belongs to.
SUBPATH_REPO = {
    "motions_dir": "booster_assets",
    "task_root": "booster_train",
    "task_registry": "booster_train",
    "train_script": "booster_train",
    "csv_to_npz_script": "booster_train",
}


class BoosterPathError(RuntimeError):
    """Raised with one message listing every unresolved or mispinned checkout."""


def _strip_comments(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [_strip_comments(v) for v in node]
    return node


def load_paths_file(path: Path | str | None = None) -> dict:
    """Read configs/paths.json (comment keys stripped)."""
    source = Path(path) if path else DEFAULT_PATHS_FILE
    if not source.is_file():
        raise BoosterPathError(
            f"Paths config not found: {source}\n"
            "Expected configs/paths.json in the repository root."
        )
    return _strip_comments(json.loads(source.read_text(encoding="utf-8")))


def parse_overrides(items: Iterable[str] | None) -> dict:
    """Turn ``--set`` strings into {repo: path}.

    Accepts ``paths.booster_train=/x`` and the bare ``booster_train=/x``.
    Anything that is not a paths.* key is ignored, so a caller can pass its
    whole ``--set`` list through.
    """
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key.startswith("paths."):
            key = key[len("paths."):]
        if key in REPOS:
            out[key] = value.strip()
    return out


def git_head(path: Path) -> str | None:
    """Current commit of a checkout, or None if it is not a git work tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass
class BoosterPaths:
    """Resolved locations of the three checkouts."""

    booster_train: Path
    booster_assets: Path
    booster_deploy: Path
    pinned_commits: dict = field(default_factory=dict)
    subpaths: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)   # repo -> where the value came from

    def path(self, repo: str) -> Path:
        if repo not in REPOS:
            raise KeyError(f"Unknown Booster repo: {repo!r}")
        return getattr(self, repo)

    def sub(self, key: str) -> Path:
        """Absolute path of a subpath declared in configs/paths.json."""
        if key not in self.subpaths:
            raise KeyError(
                f"Unknown subpath {key!r}; configs/paths.json declares "
                f"{sorted(self.subpaths)}"
            )
        repo = SUBPATH_REPO.get(key)
        if repo is None:
            raise KeyError(f"Subpath {key!r} is not mapped to a checkout")
        return self.path(repo) / self.subpaths[key]

    # Named accessors for the subpaths other stages use.
    @property
    def motions_dir(self) -> Path:
        return self.sub("motions_dir")

    @property
    def task_root(self) -> Path:
        return self.sub("task_root")

    @property
    def task_registry(self) -> Path:
        return self.sub("task_registry")

    @property
    def train_script(self) -> Path:
        return self.sub("train_script")

    @property
    def csv_to_npz_script(self) -> Path:
        return self.sub("csv_to_npz_script")

    def describe(self) -> str:
        lines = []
        for repo in REPOS:
            head = git_head(self.path(repo)) or "?"
            lines.append(
                f"  {repo:<15} {self.path(repo)}\n"
                f"  {'':<15} commit {head[:12]}  (from {self.sources.get(repo, '?')})"
            )
        return "\n".join(lines)


def _resolve_one(repo: str, cli: dict, env_root: str | None, file_value: str) -> tuple[Path, str]:
    if repo in cli:
        return Path(cli[repo]).expanduser().resolve(), "--set paths." + repo
    if env_root:
        return (Path(env_root).expanduser() / repo).resolve(), "EM_BOOSTER_ROOT"
    value = Path(file_value).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve(), "configs/paths.json"


def resolve_paths(
    config: Any = None,
    overrides: Iterable[str] | Mapping[str, str] | None = None,
    *,
    paths_file: Path | str | None = None,
    verify: bool = True,
    check_commits: bool = True,
) -> BoosterPaths:
    """Resolve the three checkouts.

    ``config`` may be a PipelineConfig, a plain dict, or None; it is consulted
    only for a ``paths`` block, so callers that have one can pass it and callers
    that do not can omit it.  ``overrides`` is a list of ``--set`` strings or a
    plain {repo: path} mapping.
    """
    data = load_paths_file(paths_file)

    # A pipeline config may carry a paths block; it sits below --set but above
    # the file, in the same slot as the file's own values.
    if config is not None:
        block = None
        if hasattr(config, "get"):
            block = config.get("paths")
        elif isinstance(config, Mapping):
            block = config.get("paths")
        if isinstance(block, Mapping):
            for repo in REPOS:
                if repo in block:
                    data[repo] = block[repo]

    if isinstance(overrides, Mapping):
        cli = {k: v for k, v in overrides.items() if k in REPOS}
    else:
        cli = parse_overrides(overrides)

    env_root = os.environ.get("EM_BOOSTER_ROOT") or None

    resolved: dict[str, Path] = {}
    sources: dict[str, str] = {}
    for repo in REPOS:
        default = data.get(repo, f"external/booster/{repo}")
        resolved[repo], sources[repo] = _resolve_one(repo, cli, env_root, default)

    paths = BoosterPaths(
        booster_train=resolved["booster_train"],
        booster_assets=resolved["booster_assets"],
        booster_deploy=resolved["booster_deploy"],
        pinned_commits=data.get("pinned_commits", {}),
        subpaths=data.get("subpaths", {}),
        sources=sources,
    )
    if verify:
        verify_paths(paths, check_commits=check_commits)
    return paths


def verify_paths(paths: BoosterPaths, *, check_commits: bool = True) -> None:
    """Raise one BoosterPathError describing every problem found."""
    problems: list[str] = []

    for repo in REPOS:
        target = paths.path(repo)
        origin = paths.sources.get(repo, "?")
        if not target.is_dir():
            problems.append(
                f"  {repo}: directory does not exist\n"
                f"    {target}   (from {origin})"
            )
            continue

        head = git_head(target)
        if head is None:
            problems.append(
                f"  {repo}: not a git checkout, so its commit cannot be verified\n"
                f"    {target}   (from {origin})"
            )
            continue

        pinned = paths.pinned_commits.get(repo)
        if check_commits:
            if not pinned:
                problems.append(
                    f"  {repo}: no pinned commit is declared in configs/paths.json\n"
                    f"    {target}   (from {origin})"
                )
            elif head != pinned:
                problems.append(
                    f"  {repo}: commit does not match the pin\n"
                    f"    {target}   (from {origin})\n"
                    f"    found  {head}\n"
                    f"    pinned {pinned}"
                )

    if not problems:
        return

    raise BoosterPathError(
        "Booster checkouts are not usable ("
        + str(len(problems))
        + " problem" + ("s" if len(problems) != 1 else "") + "):\n"
        + "\n".join(problems)
        + "\n\nFix with one of:\n"
          "  git submodule update --init --recursive\n"
          "      (checks out the pinned commits under external/booster/)\n"
          "  git -C <checkout> checkout <pinned commit>\n"
          "  export EM_BOOSTER_ROOT=/path/to/the/parent/of/the/three/checkouts\n"
          "  --set paths.<repo>=/path/to/<repo>\n"
          "  or edit configs/paths.json, and update pinned_commits with it\n"
          "Pinned commits are recorded in configs/paths.json."
    )


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Resolve and verify the three Booster checkouts."
    )
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        metavar="paths.KEY=VALUE")
    parser.add_argument("--paths-config", default=None)
    parser.add_argument("--no-commit-check", action="store_true")
    args = parser.parse_args(argv)

    try:
        paths = resolve_paths(
            overrides=args.overrides,
            paths_file=args.paths_config,
            check_commits=not args.no_commit_check,
        )
    except BoosterPathError as exc:
        print(str(exc))
        return 1

    print("Booster checkouts OK:")
    print(paths.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
