"""Derive robot-foot IK offsets from robot and SMPL-X geometry.

The mapped SMPL-X foot joint and a robot foot body's origin represent
different anatomical points. This module derives the offset from the robot
sole centroid and the scaled SMPL-X foot geometry instead of applying a
robot-specific hard-coded translation. Offsets are expressed in the robot
foot body's local frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import robot_kinematics


# SMPL-X joint indices for the lower limb.
SMPLX_JOINTS = {
    "left": {"ankle": 7, "toe": 10},
    "right": {"ankle": 8, "toe": 11},
}


@dataclass
class FootOffset:
    robot_body: str
    human_body: str
    offset: np.ndarray
    sole_centroid_local: np.ndarray
    human_span_m: float

    def as_dict(self) -> dict:
        return {
            "robot_body": self.robot_body,
            "human_body": self.human_body,
            "offset": [round(float(v), 5) for v in self.offset],
            "sole_centroid_local": [round(float(v), 5) for v in self.sole_centroid_local],
            "human_foot_span_m": round(float(self.human_span_m), 5),
        }


def sole_centroid_local(model, foot_body: str, patch_fraction: float = 0.15) -> np.ndarray:
    """Centroid of the lowest slice of a foot mesh, in the body's local frame."""
    verts = robot_kinematics.mesh_vertices(model, foot_body)
    take = max(1, int(patch_fraction * len(verts)))
    lowest = np.argsort(verts[:, 2])[:take]
    return verts[lowest].mean(axis=0)


def derive(
    model,
    foot_bodies: dict[str, str],
    human_joints_zup: np.ndarray,
    scale: float,
    rot_offsets: dict[str, np.ndarray],
    include_vertical: bool = True,
) -> dict[str, FootOffset]:
    """Compute a position offset for each mapped foot body.

    ``foot_bodies`` maps ``side -> robot body name`` (``{"left": ..., "right": ...}``).
    ``human_joints_zup`` is (F, J, 3) SMPL-X joints already in the z-up frame the
    retargeter uses.  ``rot_offsets`` gives the per-side rotation matrix that
    takes the target frame to the robot foot frame, averaged over the clip.
    """
    offsets: dict[str, FootOffset] = {}

    for side, robot_body in foot_bodies.items():
        indices = SMPLX_JOINTS.get(side)
        if indices is None:
            continue

        ankle = human_joints_zup[:, indices["ankle"], :]
        toe = human_joints_zup[:, indices["toe"], :]

        # Human foot centre expressed relative to the mapped joint (the toe),
        # in world axes, scaled to robot size.
        centre_world = 0.5 * (ankle - toe) * scale
        span = float(np.linalg.norm((ankle - toe) * scale, axis=1).mean())

        rotation = rot_offsets.get(side)
        if rotation is None:
            continue

        # Into the foot's local frame, averaged over the clip. The per-frame
        # spread of this quantity is tiny because it is a skeletal constant.
        local = (rotation.transpose(0, 2, 1) @ centre_world[:, :, None]).squeeze(-1)
        local_mean = local.mean(axis=0)

        centroid = sole_centroid_local(model, robot_body)
        offset = local_mean - centroid

        if not include_vertical:
            offset = offset.copy()
            offset[2] = 0.0

        offsets[side] = FootOffset(
            robot_body=robot_body,
            human_body=f"{side}_foot",
            offset=offset,
            sole_centroid_local=centroid,
            human_span_m=span,
        )

    return offsets


def measured_height(vertices_zup: np.ndarray) -> float:
    """Subject height from the posed SMPL-X mesh.

    GMR's ``1.66 + 0.1 * beta0`` heuristic was 9 cm low on our sample, which
    biases every scale by about 5%.  Measuring the mesh is exact and costs
    nothing since we already build it.
    """
    per_frame = np.ptp(vertices_zup[:, :, 2], axis=1)
    return float(np.percentile(per_frame, 95))
