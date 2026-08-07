"""Provide MuJoCo kinematics and mesh utilities for robot-foot alignment.

The helpers locate foot bodies, read mesh vertices, and trace foot poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np


FOOT_BODY_GUESSES = (
    ("left_foot_link", "right_foot_link"),
    ("left_ankle_roll_link", "right_ankle_roll_link"),
    ("left_foot", "right_foot"),
    ("L_foot", "R_foot"),
)


def load_model(xml_path: str | Path) -> mj.MjModel:
    return mj.MjModel.from_xml_path(str(Path(xml_path).resolve()))


def body_names(model: mj.MjModel) -> list[str]:
    return [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or ""
        for i in range(model.nbody)
    ]


def find_foot_bodies(model: mj.MjModel) -> tuple[str, str] | None:
    """Best-effort discovery of the two foot bodies for an arbitrary robot."""
    names = set(body_names(model))

    for left, right in FOOT_BODY_GUESSES:
        if left in names and right in names:
            return left, right

    lefts = sorted(n for n in names if "foot" in n.lower() and n.lower().startswith(("l", "left")))
    rights = sorted(n for n in names if "foot" in n.lower() and n.lower().startswith(("r", "right")))
    if lefts and rights:
        return lefts[0], rights[0]

    return None


def mesh_vertices(model: mj.MjModel, body_name: str) -> np.ndarray:
    """All mesh vertices of a body, expressed in that body's local frame."""
    body_id = model.body(body_name).id
    chunks: list[np.ndarray] = []

    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id:
            continue
        if model.geom_type[geom_id] != mj.mjtGeom.mjGEOM_MESH:
            continue

        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[start : start + count].reshape(-1, 3).astype(np.float64)

        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        chunks.append(verts @ rotation.reshape(3, 3).T + model.geom_pos[geom_id])

    if not chunks:
        raise ValueError(f"Body {body_name!r} has no mesh geoms to measure.")

    return np.concatenate(chunks, axis=0)


@dataclass
class FootTrace:
    """Per-frame sole geometry for one foot."""

    sole_z: np.ndarray          # (N,) lowest sole vertex height
    contact_xy: np.ndarray      # (N, 2) centroid of the lowest vertices
    body_xyz: np.ndarray        # (N, 3) body origin (ankle)
    tilt_deg: np.ndarray        # (N,) angle between sole normal and world up


def trace_feet(
    model: mj.MjModel,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    foot_bodies: tuple[str, str],
    patch_fraction: float = 0.05,
) -> dict[str, FootTrace]:
    """Run FK over a motion and record sole height, contact patch and tilt."""
    data = mj.MjData(model)
    frame_count = len(root_pos)

    local_verts = {name: mesh_vertices(model, name) for name in foot_bodies}
    traces = {
        name: FootTrace(
            sole_z=np.zeros(frame_count),
            contact_xy=np.zeros((frame_count, 2)),
            body_xyz=np.zeros((frame_count, 3)),
            tilt_deg=np.zeros(frame_count),
        )
        for name in foot_bodies
    }

    for index in range(frame_count):
        data.qpos[:3] = root_pos[index]
        data.qpos[3:7] = root_rot_wxyz[index]
        data.qpos[7:] = dof_pos[index]
        mj.mj_forward(model, data)

        for name in foot_bodies:
            body_id = model.body(name).id
            origin = data.xpos[body_id]
            rotation = data.xmat[body_id].reshape(3, 3)
            world = local_verts[name] @ rotation.T + origin

            trace = traces[name]
            trace.sole_z[index] = world[:, 2].min()
            trace.body_xyz[index] = origin

            take = max(1, int(patch_fraction * len(world)))
            lowest = np.argsort(world[:, 2])[:take]
            trace.contact_xy[index] = world[lowest, :2].mean(axis=0)

            normal = rotation[:, 2]
            trace.tilt_deg[index] = np.degrees(
                np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))
            )

    return traces


def standing_sole_offset(model: mj.MjModel, foot_body: str) -> float:
    """Vertical drop from a foot body origin to its lowest sole vertex at rest."""
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    body_id = model.body(foot_body).id
    origin = data.xpos[body_id]
    rotation = data.xmat[body_id].reshape(3, 3)
    world = mesh_vertices(model, foot_body) @ rotation.T + origin
    return float(origin[2] - world[:, 2].min())


def max_leg_reach(model: mj.MjModel, hip_body: str, foot_body: str) -> float:
    """Straight-leg distance from hip to foot body origin at the rest pose."""
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    hip = data.xpos[model.body(hip_body).id]
    foot = data.xpos[model.body(foot_body).id]
    return float(np.linalg.norm(hip - foot))
