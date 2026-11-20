"""Restore and verify the original input frames from the extracted GitHub Release."""

import argparse
import csv
import hashlib
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def under(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes its root: {relative}")
    return path


def restore(archive, output, task, manifest_name, input_name):
    manifest = ROOT / task / manifest_name
    destination = output / task
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        frame = under(archive, row["source_image"])
        with frame.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError(f"Frame SHA-256 mismatch: {frame}")
        for source_field, dest_field in (("source_image", "image_path"), ("source_metadata", "metadata_path")):
            target = under(destination, row[dest_field])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(under(archive, row[source_field]), target)
    shutil.copyfile(manifest, destination / input_name)
    print(f"{task}: restored {len(rows)} verified original frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path, help="Extracted release directory containing capture_source/")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive = (args.archive_root / "capture_source").resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    restore(archive, output, "lesion", "original-candidates.csv", "classification.csv")
    restore(archive, output, "wall-distance", "original-split.csv", "regression.csv")


if __name__ == "__main__":
    main()
