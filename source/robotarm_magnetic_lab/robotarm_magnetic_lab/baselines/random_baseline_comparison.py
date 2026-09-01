"""Pure artifact selection helpers for the frozen 20-pose baseline comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_best_entries(entries: Iterable[dict[str, Any]]) -> tuple[dict[str, dict], dict]:
    rows = [row for row in entries if str(row.get("policy_id", "")).startswith("R")]
    by_policy: dict[str, dict] = {}
    for policy_index in range(1, 8):
        policy_id = f"R{policy_index}"
        candidates = [row for row in rows if row.get("policy_id") == policy_id]
        if len(candidates) != 20:
            raise ValueError(f"{policy_id} requires exactly 20 completed pose records")
        by_policy[policy_id] = sorted(
            candidates,
            key=lambda row: (-float(row["C_final_reachable"]), str(row["pose_id"])),
        )[0]
    overall = sorted(
        rows,
        key=lambda row: (
            -float(row["C_final_reachable"]), str(row["policy_id"]), str(row["pose_id"])
        ),
    )[0]
    return by_policy, overall


def preserve_best_snapshot_images(
    run_directory: Path,
    entries: Iterable[dict[str, Any]],
    snapshot_times_s: Iterable[int],
) -> Path:
    """Copy only selected scheduled snapshots to a stable, auditable directory."""
    run_directory = Path(run_directory)
    by_policy, overall = select_best_entries(entries)
    destination = run_directory / "best_pose_snapshots"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    selections = {**by_policy, "OVERALL": overall}
    artifacts: list[dict[str, Any]] = []
    for label, row in selections.items():
        source = run_directory / "coverage" / str(row["episode_id"])
        target = destination / label.lower()
        target.mkdir()
        for second in snapshot_times_s:
            matches = sorted(source.glob(f"snapshot_*_candidate_{int(second):03d}s.png"))
            if len(matches) != 1:
                raise ValueError(
                    f"{row['episode_id']} has {len(matches)} snapshots at t={int(second)} s"
                )
            output = target / f"coverage_{int(second):03d}s.png"
            shutil.copy2(matches[0], output)
            artifacts.append(
                {
                    "label": label,
                    "policy_id": row["policy_id"],
                    "pose_id": row["pose_id"],
                    "episode_id": row["episode_id"],
                    "time_s": int(second),
                    "path": str(output.resolve()),
                    "bytes": output.stat().st_size,
                    "sha256": file_sha256(output),
                }
            )

    manifest = {
        "schema": "robotarm_magnetic_lab.task009c_20pose_best_snapshots",
        "version": 1,
        "selection_metric": "C_final_reachable",
        "per_policy": {
            key: {
                "pose_id": row["pose_id"],
                "episode_id": row["episode_id"],
                "final_reachable_coverage": float(row["C_final_reachable"]),
            }
            for key, row in by_policy.items()
        },
        "overall": {
            "policy_id": overall["policy_id"],
            "pose_id": overall["pose_id"],
            "episode_id": overall["episode_id"],
            "final_reachable_coverage": float(overall["C_final_reachable"]),
        },
        "artifacts": artifacts,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Scheduled candidate images are temporary selection inputs. The normal
    # final snapshot and all numeric audit artifacts remain in every episode.
    for source in (run_directory / "coverage").glob("*/snapshot_*_candidate_*s.*"):
        if source.suffix in (".png", ".json"):
            source.unlink()
    return manifest_path
