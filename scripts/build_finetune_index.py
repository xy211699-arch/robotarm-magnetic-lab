"""Build temporal image/state -> action-chunk samples from committed episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description="Build VLA/BC temporal sample index.")
parser.add_argument("dataset_root", type=Path)
parser.add_argument("--history", type=int, default=4)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--stride", type=int, default=1)
# Accepted for compatibility with the project launcher, which injects Kit
# workarounds for every ``-p some_script.py`` invocation. This offline tool
# does not start Kit and intentionally ignores the value.
parser.add_argument("--kit_args", default=None, help=argparse.SUPPRESS)
args = parser.parse_args()


def main() -> None:
    if args.history < 1 or args.horizon < 1 or args.stride < 1:
        raise ValueError("history, horizon and stride must be positive")
    output_path = args.dataset_root / "finetune_index.jsonl"
    sample_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for episode_line in (args.dataset_root / "episodes.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            episode_ref = json.loads(episode_line)
            if not episode_ref["success"]:
                continue
            episode_dir = args.dataset_root / episode_ref["path"]
            episode_meta = json.loads(
                (episode_dir / "episode.json").read_text(encoding="utf-8")
            )
            steps = [
                json.loads(line)
                for line in (episode_dir / "steps.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            first_anchor = args.history - 1
            last_anchor = len(steps) - args.horizon
            for anchor in range(first_anchor, last_anchor + 1, args.stride):
                history = steps[anchor - args.history + 1 : anchor + 1]
                future = steps[anchor : anchor + args.horizon]
                record = {
                    "schema_version": episode_meta["schema_version"],
                    "episode_id": episode_ref["episode_id"],
                    "anchor_step": anchor,
                    "instruction": episode_meta["metadata"].get(
                        "language_instruction", ""
                    ),
                    "rgb_history": [
                        f"{episode_ref['path']}/{step['rgb_path']}" for step in history
                    ],
                    "depth_history": [
                        f"{episode_ref['path']}/{step['depth_path']}" for step in history
                    ],
                    "policy_state_history": [
                        step["policy_state"] for step in history
                    ],
                    "target_action_chunk": [
                        step["action_command"] for step in future
                    ],
                    "target_joint_position_chunk_rad": [
                        step["action_applied_joint_target_rad"] for step in future
                    ],
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                sample_count += 1
    print(
        f"[FINETUNE_INDEX] samples={sample_count} history={args.history} "
        f"horizon={args.horizon} output={output_path}"
    )


if __name__ == "__main__":
    main()
