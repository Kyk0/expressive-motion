"""Video metadata probing and 30 fps staging."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    fps_exact: Fraction
    frame_count: int
    width: int
    height: int
    duration: float

def _ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if not executable:
        raise ProbeError("ffprobe not found on PATH; install ffmpeg.")
    return executable


def probe(video: Path) -> VideoInfo:
    """Read frame rate, frame count and geometry from a video file."""
    video = Path(video)
    if not video.is_file():
        raise ProbeError(f"Video not found: {video}")

    result = subprocess.run(
        [
            _ffprobe(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed for {video}: {result.stderr.strip()}")

    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise ProbeError(f"No video stream in {video}")

    stream = streams[0]

    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    try:
        fps_exact = Fraction(rate)
    except (ZeroDivisionError, ValueError):
        fps_exact = Fraction(0)

    if fps_exact <= 0:
        try:
            fps_exact = Fraction(stream.get("r_frame_rate", "0/1"))
        except (ZeroDivisionError, ValueError):
            fps_exact = Fraction(0)

    if fps_exact <= 0:
        raise ProbeError(f"Could not determine frame rate for {video}")

    duration = 0.0
    for candidate in (stream.get("duration"), (payload.get("format") or {}).get("duration")):
        try:
            duration = float(candidate)
            break
        except (TypeError, ValueError):
            continue

    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = int(round(duration * float(fps_exact))) if duration else 0

    return VideoInfo(
        path=video,
        fps=float(fps_exact),
        fps_exact=fps_exact,
        frame_count=frame_count,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        duration=duration or (frame_count / float(fps_exact) if frame_count else 0.0),
    )


def resample_to_30(source: Path, target: Path, interpolate: bool = False) -> Path:
    """Retime a clip to a true 30 fps.

    GVHMR re-encodes its working copy at a hard-coded 30 fps and its temporal
    model and velocity-based contact detector both assume that rate.  Feeding a
    24 fps clip therefore makes the network see motion 25% too fast.  Resampling
    first keeps real-world velocities correct.

    ``interpolate`` uses ffmpeg motion interpolation (much better, much slower).
    The default duplicates frames, which is cheap and adequate at volume.
    """
    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("ffmpeg")
    if not executable:
        raise ProbeError("ffmpeg not found on PATH.")

    if interpolate:
        video_filter = "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
    else:
        video_filter = "fps=30"

    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            video_filter,
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            str(target),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise ProbeError(f"ffmpeg resample failed for {source}: {result.stderr.strip()}")

    return target
