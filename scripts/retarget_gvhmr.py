#!/usr/bin/env python3
"""Retarget a GVHMR result onto a robot and align it with the ground.

The retargeter measures subject height from the SMPL-X mesh, derives foot IK
offsets from robot geometry, and applies a contact-based ground correction.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from expressive_motion import (  # noqa: E402
    foot_alignment,
    ground_alignment,
    robot_kinematics,
)
from expressive_motion.pipeline_config import PipelineConfig  # noqa: E402


ZUP_FROM_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
IK_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "ik_configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-robots", action="store_true")
    parser.add_argument("--list-robots-json", action="store_true")
    parser.add_argument("--gvhmr-result", type=Path)
    parser.add_argument("--robot")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--body-models", type=Path)
    parser.add_argument("--fps", type=float, default=None, help="True source frame rate.")
    parser.add_argument("--clip-name", default=None)
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--no-ground-fix", action="store_true")
    return parser.parse_args()


def available_robots() -> list[str]:
    from general_motion_retargeting.params import IK_CONFIG_DICT

    return sorted(IK_CONFIG_DICT["smplx"].keys())


def load_gvhmr(result_path: Path, body_models: Path, use_mesh_height: bool):
    """Build SMPL-X output and the frame dicts GMR consumes."""
    import smplx

    prediction = torch.load(result_path, map_location="cpu", weights_only=False)

    if "smpl_params_global" not in prediction:
        raise KeyError("GVHMR output is missing 'smpl_params_global'.")

    params = prediction["smpl_params_global"]
    for key in ("betas", "body_pose", "global_orient", "transl"):
        if key not in params:
            raise KeyError(f"GVHMR output is missing SMPL-X field: {key}")

    betas = params["betas"][0].detach().cpu().numpy()
    betas = np.pad(betas, (0, max(0, 16 - betas.shape[0])))[:16]

    body_pose = params["body_pose"].detach().cpu()
    global_orient = params["global_orient"].detach().cpu()
    transl = params["transl"].detach().cpu()
    frame_count = body_pose.shape[0]

    body_model = smplx.create(
        str(body_models),
        model_type="smplx",
        gender="neutral",
        ext="npz",
        use_pca=False,
        num_betas=16,
    )

    zeros = lambda width: torch.zeros(frame_count, width)
    output = body_model(
        betas=torch.tensor(betas, dtype=torch.float32).view(1, -1),
        global_orient=global_orient.float(),
        body_pose=body_pose.float(),
        transl=transl.float(),
        left_hand_pose=zeros(45),
        right_hand_pose=zeros(45),
        jaw_pose=zeros(3),
        leye_pose=zeros(3),
        reye_pose=zeros(3),
        return_full_pose=True,
    )

    vertices_zup = output.vertices.detach().numpy() @ ZUP_FROM_YUP.T
    joints_zup = output.joints.detach().numpy() @ ZUP_FROM_YUP.T

    heuristic_height = 1.66 + 0.1 * float(betas[0])
    mesh_height = foot_alignment.measured_height(vertices_zup)
    height = mesh_height if use_mesh_height else heuristic_height

    smplx_data = {
        "pose_body": body_pose.numpy(),
        "betas": betas,
        "root_orient": global_orient.numpy(),
        "trans": transl.numpy(),
        # GVHMR always works at 30 fps internally; this field only drives GMR's
        # resampling, which we keep disabled so no frames are invented.
        "mocap_frame_rate": torch.tensor(30),
    }

    return smplx_data, body_model, output, height, joints_zup, prediction


def gvhmr_contacts(prediction: dict, frame_count: int, threshold: float):
    """Per-foot contact labels from GVHMR's static confidence head."""
    net = prediction.get("net_outputs") or {}
    logits = net.get("static_conf_logits")
    if logits is None:
        return None

    values = logits[0].detach().cpu().numpy() if hasattr(logits, "detach") else np.asarray(logits[0])
    if values.ndim != 2 or values.shape[0] < frame_count or values.shape[1] < 4:
        return None

    probability = 1.0 / (1.0 + np.exp(-values[:frame_count]))
    # Column order is [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist].
    return {
        "left": (probability[:, [0, 1]] > threshold).any(axis=1),
        "right": (probability[:, [2, 3]] > threshold).any(axis=1),
    }


def average_target_rotations(frames: list[dict], human_bodies: dict[str, str], rot_offset):
    """Per-frame rotation matrices of the foot IK targets, per side."""
    from scipy.spatial.transform import Rotation as R

    result: dict[str, np.ndarray] = {}
    for side, body in human_bodies.items():
        mats = []
        for frame in frames:
            entry = frame.get(body)
            if entry is None:
                continue
            quat = np.asarray(entry[1], dtype=np.float64)
            mats.append((R.from_quat(quat, scalar_first=True) * rot_offset).as_matrix())
        if mats:
            result[side] = np.stack(mats, axis=0)
    return result


def main() -> None:
    args = parse_args()

    if args.list_robots_json:
        print(json.dumps(available_robots()))
        return
    if args.list_robots:
        print("\n".join(available_robots()))
        return

    missing = [
        name
        for name, value in (
            ("--gvhmr-result", args.gvhmr_result),
            ("--robot", args.robot),
            ("--output", args.output),
            ("--body-models", args.body_models),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

    config = PipelineConfig.load(args.pipeline_config, args.overrides)

    if not args.gvhmr_result.is_file():
        raise SystemExit(f"GVHMR result not found: {args.gvhmr_result}")

    robots = available_robots()
    if args.robot not in robots:
        raise SystemExit(
            f"Unsupported robot: {args.robot}\nAvailable robots:\n  " + "\n  ".join(robots)
        )

    neutral = args.body_models / "smplx" / "SMPLX_NEUTRAL.npz"
    if not neutral.is_file():
        raise SystemExit(f"Missing SMPL-X body model: {neutral}")

    clip_name = args.clip_name or args.gvhmr_result.parent.name

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
    from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast
    from scipy.spatial.transform import Rotation as R
    from tqdm import tqdm

    smplx_data, body_model, smplx_output, height, joints_zup, prediction = load_gvhmr(
        args.gvhmr_result,
        args.body_models,
        bool(config.get("retarget.use_smplx_mesh_height", True)),
    )

    frames, _ = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    if not frames:
        raise SystemExit("GVHMR motion contains no frames.")

    frame_count = len(frames)
    fps = float(args.fps) if args.fps else 30.0

    # ---- derive foot offsets from geometry ----
    ik_path = Path(IK_CONFIG_DICT["smplx"][args.robot])
    ik_config = json.loads(ik_path.read_text(encoding="utf-8"))
    model = robot_kinematics.load_model(ROBOT_XML_DICT[args.robot])
    foot_pair = robot_kinematics.find_foot_bodies(model)

    if foot_pair and config.get("retarget.auto_foot_offset", True):
        side_bodies = {"left": foot_pair[0], "right": foot_pair[1]}
        human_bodies = {}
        rot_offsets_raw = {}

        for side, robot_body in side_bodies.items():
            for table_name in ("ik_match_table2", "ik_match_table1"):
                entry = (ik_config.get(table_name) or {}).get(robot_body)
                if entry:
                    human_bodies[side] = entry[0]
                    rot_offsets_raw[side] = entry[4]
                    break

        if len(human_bodies) == 2:
            scale = (
                ik_config["human_scale_table"].get(human_bodies["left"], 1.0)
                * height
                / ik_config["human_height_assumption"]
            )
            rot_mats = {}
            for side in human_bodies:
                rot_offset = R.from_quat(rot_offsets_raw[side], scalar_first=True)
                rot_mats.update(
                    average_target_rotations(frames, {side: human_bodies[side]}, rot_offset)
                )

            derived = foot_alignment.derive(model, side_bodies, joints_zup, scale, rot_mats)
            for item in derived.values():
                robot_body = item.robot_body
                for table_name in ("ik_match_table1", "ik_match_table2"):
                    table = ik_config.get(table_name) or {}
                    if robot_body in table:
                        table[robot_body][3] = [float(v) for v in item.offset]

            patched = IK_CACHE_DIR / f"smplx_to_{args.robot}.auto.json"
            patched.parent.mkdir(parents=True, exist_ok=True)
            patched.write_text(json.dumps(ik_config, indent=2) + "\n", encoding="utf-8")
            IK_CONFIG_DICT["smplx"][args.robot] = str(patched)

    # ---- retarget ----
    retargeter = GMR(actual_human_height=height, src_human="smplx", tgt_robot=args.robot)

    qpos_list = [
        np.asarray(retargeter.retarget(frame)).copy()
        for frame in tqdm(frames, desc=f"Retargeting {clip_name} -> {args.robot}")
    ]
    qpos = np.stack(qpos_list)

    root_pos = qpos[:, :3].copy()
    root_rot_wxyz = qpos[:, 3:7].copy()
    dof_pos = qpos[:, 7:].copy()

    # ---- ground alignment ----
    ground_stats: dict = {}

    if not foot_pair:
        ground_stats = {
            "strategy": "unavailable",
            "confidence": "low",
            "reason": f"no foot bodies found for {args.robot}",
        }
    else:
        def measure(rp: np.ndarray, rr: np.ndarray):
            t = robot_kinematics.trace_feet(model, rp, rr, dof_pos, foot_pair)
            return (
                {n: t[n].sole_z for n in foot_pair},
                {n: t[n].contact_xy for n in foot_pair},
            )

        traces = robot_kinematics.trace_feet(model, root_pos, root_rot_wxyz, dof_pos, foot_pair)
        sole_z = {n: traces[n].sole_z for n in foot_pair}
        contact_xy = {n: traces[n].contact_xy for n in foot_pair}

        contact_source = config.get("contacts.source", "auto")
        labels = None
        if contact_source in ("auto", "gvhmr"):
            raw = gvhmr_contacts(
                prediction,
                frame_count,
                float(config.get("contacts.gvhmr_probability_threshold", 0.8)),
            )
            if raw is not None:
                labels = {foot_pair[0]: raw["left"], foot_pair[1]: raw["right"]}
                coverage = float((labels[foot_pair[0]] | labels[foot_pair[1]]).mean())
                if (
                    contact_source == "auto"
                    and coverage < float(config.get("contacts.min_gvhmr_coverage", 0.35))
                ):
                    labels = None

        if labels is None:
            labels = ground_alignment.detect_contacts(
                sole_z,
                contact_xy,
                fps,
                speed_threshold=float(config.get("contacts.geometric_speed_threshold_mps", 0.12)),
                height_percentile=float(config.get("contacts.geometric_height_percentile", 35.0)),
            )
            contact_method = "geometric"
        else:
            contact_method = "gvhmr"

        contacts = labels

        if config.get("ground.enabled", True) and not args.no_ground_fix:
            result = ground_alignment.align_to_ground(
                root_pos=root_pos,
                root_rot_wxyz=root_rot_wxyz,
                sole_z=sole_z,
                contact_xy=contact_xy,
                fps=fps,
                contacts=contacts,
                cutoff_hz=float(config.get("ground.cutoff_hz", 0.30)),
                max_level_deg=float(config.get("ground.max_level_deg", 12.0)),
                max_offset=float(config.get("ground.max_offset_m", 1.0)),
                level=bool(config.get("ground.level", True)),
                min_contact_coverage=float(config.get("ground.min_contact_coverage", 0.15)),
                min_frames_for_timevarying=int(
                    config.get("ground.min_frames_for_timevarying", 45)
                ),
                min_frames_for_level=int(config.get("ground.min_frames_for_level", 25)),
                recompute=measure,
            )
            root_pos = result.root_pos
            root_rot_wxyz = result.root_rot_wxyz
            ground_stats = dict(result.stats)
            ground_stats["contact_method"] = contact_method

        else:
            ground_stats = {"strategy": "disabled", "contact_method": contact_method}

    # ---- save motion ----
    motion = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot_wxyz[:, [1, 2, 3, 0]],  # wxyz -> xyzw for load_robot_motion
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
        "robot": args.robot,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(motion, handle)

    print(f"Saved robot motion: {args.output}")
    print(f"Frames: {frame_count}   FPS: {fps:.3f}   DoF: {dof_pos.shape[1]}")
    strategy = ground_stats.get("strategy")
    if strategy == "unavailable":
        print(f"Ground: unavailable ({ground_stats['reason']})")
    elif strategy == "disabled":
        print("Ground: disabled")
    elif ground_stats:
        print(
            "Ground: strategy={strategy} confidence={confidence} "
            "drift {before:.3f} -> {after:.3f} m".format(
                strategy=strategy,
                confidence=ground_stats.get("confidence", "?"),
                before=ground_stats.get("clearance_range_before", float("nan")),
                after=ground_stats.get("clearance_range_after", float("nan")),
            )
        )


if __name__ == "__main__":
    main()
