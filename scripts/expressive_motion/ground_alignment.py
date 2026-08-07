"""Apply contact-informed ground alignment to retargeted robot motion.

The correction optionally levels the floating base against estimated foot
contact points and applies a low-frequency vertical root offset. Joint angles
are never modified, so the operation preserves the retargeted articulation.
This is a root-trajectory correction rather than foot locking or general
motion smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.signal import butter, filtfilt

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy ships with the GMR env
    _HAVE_SCIPY = False


@dataclass
class GroundResult:
    root_pos: np.ndarray
    root_rot_wxyz: np.ndarray
    level_rotation_deg: float
    offset_curve: np.ndarray
    contact_frames: int
    strategy: str = "time_varying"
    confidence: str = "high"
    stats: dict = field(default_factory=dict)


def _lowpass(signal: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Zero-phase low-pass.  Falls back to a Gaussian blur without scipy."""
    n = len(signal)
    if n < 9 or cutoff_hz <= 0:
        return np.full(n, float(np.median(signal)))

    nyquist = 0.5 * fps
    normalised = min(cutoff_hz / nyquist, 0.99)

    if _HAVE_SCIPY:
        padlen = min(3 * 4, n - 1)
        try:
            b, a = butter(2, normalised, btype="low")
            return filtfilt(b, a, signal, padlen=padlen)
        except Exception:
            pass

    sigma = max(1.0, fps / (2.0 * np.pi * max(cutoff_hz, 1e-6)))
    radius = int(min(4 * sigma, (n - 1) / 2))
    if radius < 1:
        return signal.copy()
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(signal, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _interpolate_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid samples by linear interpolation, holding the edges."""
    filled = values.astype(np.float64).copy()
    if not valid.any():
        return np.zeros_like(filled)

    index = np.arange(len(filled))
    filled[~valid] = np.interp(index[~valid], index[valid], filled[valid])
    return filled


def detect_contacts(
    sole_z: dict[str, np.ndarray],
    contact_xy: dict[str, np.ndarray],
    fps: float,
    speed_threshold: float = 0.12,
    height_percentile: float = 35.0,
    height_fraction: float = 0.45,
    speed_percentile: float = 70.0,
    adaptive: bool = True,
) -> dict[str, np.ndarray]:
    """Geometric contact detection, used when GVHMR labels are missing or sparse.

    A foot counts as planted when its sole is low within that foot's own swing
    range and its contact patch is moving slowly for that clip.

    Both tests are relative rather than absolute.  A fixed speed threshold
    rejects genuine contacts on any clip where the retargeted foot scuffs, and a
    fixed height percentile caps coverage at that percentile no matter how much
    of the clip is actually weight-bearing.  Measured on walking clips, the
    absolute form found 13-22% of frames planted where roughly 60-80% should be;
    the relative form finds 35-53%.
    """
    contacts: dict[str, np.ndarray] = {}

    for name, z in sole_z.items():
        count = len(z)
        speed = np.zeros(count)
        if count > 1:
            speed[1:] = np.linalg.norm(np.diff(contact_xy[name], axis=0), axis=1) * fps
            speed[0] = speed[1]

        detrended = z - _lowpass(z, fps, 0.30)

        if not adaptive:
            low = detrended < np.percentile(detrended, height_percentile)
            contacts[name] = low & (speed < speed_threshold)
            continue

        floor = np.percentile(detrended, 2.0)
        ceiling = np.percentile(detrended, 98.0)
        low = detrended < floor + height_fraction * max(ceiling - floor, 1e-6)

        if low.any():
            limit = max(float(np.percentile(speed[low], speed_percentile)), speed_threshold * 0.5)
        else:
            limit = speed_threshold

        contacts[name] = low & (speed < limit)

    return contacts


def _fit_level_rotation(points: np.ndarray, max_deg: float) -> tuple[np.ndarray, float]:
    """Robustly fit a plane to contact points and return a levelling rotation."""
    if len(points) < 25:
        return np.eye(3), 0.0

    keep = np.ones(len(points), bool)
    coef = np.zeros(3)

    for _ in range(8):
        design = np.c_[points[keep, 0], points[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(design, points[keep, 2], rcond=None)
        residual = points[:, 2] - (np.c_[points[:, 0], points[:, 1], np.ones(len(points))] @ coef)
        spread = residual[keep].std()
        keep = np.abs(residual) < max(0.02, 1.4 * spread)
        if keep.sum() < 25:
            return np.eye(3), 0.0

    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)

    angle = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))
    if angle < 0.05 or angle > max_deg:
        return np.eye(3), 0.0

    axis = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(3), 0.0

    axis /= norm
    theta = np.arctan2(norm, float(np.dot(normal, np.array([0.0, 0.0, 1.0]))))
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    return rotation, angle


def _quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    return R.from_matrix(matrix).as_quat(scalar_first=True)


def align_to_ground(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    sole_z: dict[str, np.ndarray],
    contact_xy: dict[str, np.ndarray],
    fps: float,
    contacts: dict[str, np.ndarray] | None = None,
    cutoff_hz: float = 0.30,
    max_level_deg: float = 12.0,
    max_offset: float = 1.0,
    level: bool = True,
    min_contact_coverage: float = 0.15,
    min_frames_for_timevarying: int = 45,
    min_frames_for_level: int = 25,
    recompute: callable | None = None,
) -> GroundResult:
    """Level and vertically anchor a retargeted motion.

    ``recompute`` optionally re-runs FK after the levelling rotation so the
    vertical correction is measured on the rotated geometry.  It receives the
    new ``(root_pos, root_rot_wxyz)`` and returns fresh ``(sole_z, contact_xy)``.
    """
    root_pos = np.asarray(root_pos, dtype=np.float64).copy()
    root_rot_wxyz = np.asarray(root_rot_wxyz, dtype=np.float64).copy()
    frame_count = len(root_pos)
    foot_names = list(sole_z.keys())

    if contacts is None:
        contacts = detect_contacts(sole_z, contact_xy, fps)

    any_contact = np.zeros(frame_count, bool)
    for name in foot_names:
        any_contact |= contacts[name]

    # ---- decide how much correction the evidence supports ----
    coverage = float(any_contact.mean()) if frame_count else 0.0

    if not any_contact.any():
        strategy, confidence = "constant_min", "low"
    elif coverage < min_contact_coverage or frame_count < min_frames_for_timevarying:
        strategy, confidence = "constant", "low"
    elif coverage < 0.45:
        strategy, confidence = "time_varying", "medium"
    else:
        strategy, confidence = "time_varying", "high"

    allow_level = (
        level
        and any_contact.sum() >= min_frames_for_level
        and frame_count >= min_frames_for_level
    )

    # ---- 1. global re-levelling ----
    rotation = np.eye(3)
    level_deg = 0.0

    if allow_level and any_contact.any():
        points = []
        for name in foot_names:
            mask = contacts[name]
            if mask.any():
                points.append(
                    np.c_[contact_xy[name][mask], sole_z[name][mask]]
                )
        if points:
            rotation, level_deg = _fit_level_rotation(np.concatenate(points, axis=0), max_level_deg)

    if level_deg > 0.0:
        pivot = root_pos[:, :2].mean(axis=0)
        centre = np.array([pivot[0], pivot[1], 0.0])
        root_pos = (root_pos - centre) @ rotation.T + centre

        from scipy.spatial.transform import Rotation as R

        delta = R.from_matrix(rotation)
        rotated = delta * R.from_quat(root_rot_wxyz, scalar_first=True)
        root_rot_wxyz = rotated.as_quat(scalar_first=True)

        if recompute is not None:
            sole_z, contact_xy = recompute(root_pos, root_rot_wxyz)

    # ---- 2. time-varying vertical anchor ----
    observation = np.zeros(frame_count)
    valid = np.zeros(frame_count, bool)

    for index in range(frame_count):
        heights = [sole_z[name][index] for name in foot_names if contacts[name][index]]
        if heights:
            observation[index] = min(heights)
            valid[index] = True

    stacked_all = np.stack([sole_z[name] for name in foot_names], axis=0)

    if strategy == "constant_min":
        # Nothing was ever detected as planted. The safest assumption is that
        # the clip's lowest sole touched the floor at least once, which is what
        # GVHMR itself assumes. One constant offset, no shape imposed.
        offset = np.full(frame_count, float(np.percentile(stacked_all.min(axis=0), 2)))
    elif strategy == "constant":
        # Too little contact evidence to trust a time-varying curve; a single
        # robust offset cannot invent motion that is not there.
        filled = _interpolate_gaps(observation, valid)
        offset = np.full(frame_count, float(np.median(filled[valid])))
    else:
        filled = _interpolate_gaps(observation, valid)
        offset = _lowpass(filled, fps, cutoff_hz)

    offset = np.clip(offset, -max_offset, max_offset)

    root_pos[:, 2] -= offset

    before = stacked_all.min(axis=0)
    after = before - offset

    result = GroundResult(
        root_pos=root_pos,
        root_rot_wxyz=root_rot_wxyz,
        level_rotation_deg=level_deg,
        offset_curve=offset,
        contact_frames=int(any_contact.sum()),
        strategy=strategy,
        confidence=confidence,
        stats={
            "strategy": strategy,
            "confidence": confidence,
            "contact_coverage": coverage,
            "offset_min": float(offset.min()),
            "offset_max": float(offset.max()),
            "offset_range": float(np.ptp(offset)),
            "clearance_range_before": float(np.ptp(before)),
            "clearance_range_after": float(np.ptp(after)),
            "clearance_mean_before": float(before.mean()),
            "clearance_mean_after": float(after.mean()),
            "level_rotation_deg": level_deg,
            "cutoff_hz": cutoff_hz,
        },
    )
    return result
