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
        for expected_step, step in enumerate(steps):
            prefix = f"{episode_ref['episode_id']}:{expected_step}"
            if step["step"] != expected_step:
                raise ValueError(f"{prefix}: non-contiguous step")
            if float(step["sim_time_s"]) <= previous_time:
                raise ValueError(f"{prefix}: non-increasing timestamp")
            previous_time = float(step["sim_time_s"])
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
            if args.check_images:
                with Image.open(rgb_path) as image:
                    if (image.height, image.width, 3) != rgb_shape:
                        raise ValueError(f"{prefix}: RGB dimensions mismatch")
                with Image.open(depth_path) as image:
                    if (image.height, image.width) != depth_shape:
                        raise ValueError(f"{prefix}: depth dimensions mismatch")
            total_steps += 1
    print(
        f"[DATASET_VALID] episodes={len(episodes)} steps={total_steps} "
        f"schema={manifest['interface_schema_version']} root={args.dataset_root}"
    )


if __name__ == "__main__":
    main()
