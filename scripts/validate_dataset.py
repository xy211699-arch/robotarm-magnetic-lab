"""Validate dataset files, shapes, timestamps and numerical ranges."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


parser = argparse.ArgumentParser(description="Validate robotarm magnetic dataset.")
parser.add_argument("dataset_root", type=Path)
parser.add_argument("--check_images", action="store_true")
# See build_finetune_index.py: the offline validator accepts but ignores the
# launcher's automatically appended Kit compatibility arguments.
parser.add_argument("--kit_args", default=None, help=argparse.SUPPRESS)
args = parser.parse_args()


def _finite_vector(value, length: int, label: str) -> None:
    if len(value) != length:
        raise ValueError(f"{label}: expected length {length}, got {len(value)}")
    if not all(math.isfinite(float(item)) for item in value):
        raise ValueError(f"{label}: non-finite value")


def main() -> None:
    manifest = json.loads((args.dataset_root / "dataset.json").read_text())
    rgb_shape = tuple(manifest["interface"]["inputs"]["rgb"]["shape"])
    depth_shape = tuple(manifest["interface"]["inputs"]["depth"]["shape"])
    schema_version = manifest["interface_schema_version"]
    rates = manifest["interface"]["rates_hz"]
    control_period = 1.0 / float(rates.get("control", rates.get("policy")))
    expected_camera_stride = int(rates.get("control", rates.get("policy"))) // int(
        rates["camera"]
    )
    camera_stride_tolerance = int(
        manifest["interface"].get("timing", {}).get(
            "camera_stride_tolerance_control_steps", 0
        )
    )
    episodes = [
        json.loads(line)
        for line in (args.dataset_root / "episodes.jsonl").read_text().splitlines()
    ]
    total_steps = 0
    for episode_ref in episodes:
        episode_dir = args.dataset_root / episode_ref["path"]
        meta = json.loads((episode_dir / "episode.json").read_text())
        steps = [
            json.loads(line)
            for line in (episode_dir / "steps.jsonl").read_text().splitlines()
        ]
        if len(steps) != meta["num_steps"] or len(steps) != episode_ref["num_steps"]:
            raise ValueError(f"{episode_ref['episode_id']}: inconsistent step count")
        previous_time = -math.inf
        last_camera_frame = -1
        last_camera_timestamp = -math.inf
        last_rgb_path = None
        last_depth_path = None
        new_camera_steps = []
        for expected_step, step in enumerate(steps):
            prefix = f"{episode_ref['episode_id']}:{expected_step}"
            if step["step"] != expected_step:
                raise ValueError(f"{prefix}: non-contiguous step")
            timestamp = float(step.get("control_time_s", step.get("sim_time_s")))
            if timestamp <= previous_time:
                raise ValueError(f"{prefix}: non-increasing timestamp")
            if expected_step and not math.isclose(
                timestamp - previous_time, control_period, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError(f"{prefix}: control timestamp is not on the configured rate")
            previous_time = timestamp
            _finite_vector(step["policy_state"], 31, f"{prefix}:policy_state")
            _finite_vector(step["action_command"], 9, f"{prefix}:action")
            _finite_vector(
                step["action_applied_joint_target_rad"], 9, f"{prefix}:target"
            )
            if max(abs(float(value)) for value in step["action_command"]) > 1.0001:
                raise ValueError(f"{prefix}: action outside [-1,1]")
            rgb_path = episode_dir / step["rgb_path"]
            depth_path = episode_dir / step["depth_path"]
            if not rgb_path.is_file() or not depth_path.is_file():
                raise FileNotFoundError(f"{prefix}: missing image")
            if schema_version == "2.0.0":
                frame_id = int(step["camera_frame_id"])
                camera_timestamp = float(step["camera_timestamp_s"])
                is_new = bool(step["camera_is_new"])
                if is_new:
                    if frame_id != last_camera_frame + 1:
                        raise ValueError(f"{prefix}: non-contiguous camera frame id")
                    if camera_timestamp < last_camera_timestamp:
                        raise ValueError(f"{prefix}: camera timestamp moved backwards")
                    new_camera_steps.append(expected_step)
                    last_camera_frame = frame_id
                    last_camera_timestamp = camera_timestamp
                    last_rgb_path = step["rgb_path"]
                    last_depth_path = step["depth_path"]
                else:
                    if frame_id != last_camera_frame:
                        raise ValueError(f"{prefix}: stale row changed camera frame id")
                    if camera_timestamp != last_camera_timestamp:
                        raise ValueError(f"{prefix}: stale row changed camera timestamp")
                    if step["rgb_path"] != last_rgb_path or step["depth_path"] != last_depth_path:
                        raise ValueError(f"{prefix}: stale row must reuse the latest image paths")
            if args.check_images and (
                schema_version != "2.0.0" or bool(step["camera_is_new"])
            ):
                with Image.open(rgb_path) as image:
                    if (image.height, image.width, 3) != rgb_shape:
                        raise ValueError(f"{prefix}: RGB dimensions mismatch")
                with Image.open(depth_path) as image:
                    if (image.height, image.width) != depth_shape:
                        raise ValueError(f"{prefix}: depth dimensions mismatch")
            total_steps += 1
        if schema_version == "2.0.0":
            if not new_camera_steps or new_camera_steps[0] != 0:
                raise ValueError(f"{episode_ref['episode_id']}: first row is not a new camera frame")
            camera_strides = [
                right - left for left, right in zip(new_camera_steps, new_camera_steps[1:])
            ]
            if any(
                abs(stride - expected_camera_stride) > camera_stride_tolerance
                for stride in camera_strides
            ):
                raise ValueError(
                    f"{episode_ref['episode_id']}: camera stride mismatch {camera_strides}"
                )
            expected_frames = int(meta.get("num_camera_frames", -1))
            if expected_frames != len(new_camera_steps):
                raise ValueError(f"{episode_ref['episode_id']}: camera frame count mismatch")
    print(
        f"[DATASET_VALID] episodes={len(episodes)} steps={total_steps} "
        f"schema={manifest['interface_schema_version']} root={args.dataset_root}"
    )


if __name__ == "__main__":
    main()
