"""Verify published bytes, split isolation and test metrics without loading model files."""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("normal", "raised", "ulcerated")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_run(task, samples, originals):
    root = ROOT / task
    manifest = read_csv(root / "split-manifest.csv")
    split = {r["sample_id"]: r for r in manifest}
    parents = {key: r for key, r in split.items() if key == r["parent_sample_id"]}
    frozen = read_json(root / "frozen-selection.json")
    protocol = read_json(root / "protocol.json")
    require(len(split) == len(manifest), f"{task}: duplicate sample")
    require(Counter(r["split"] for r in manifest) == samples, f"{task}: split size")
    require(Counter(r["split"] for r in parents.values()) == originals, f"{task}: original count")
    for field in ("group_id", "image_path", "sha256"):
        boundaries = {}
        for row in manifest:
            prior = boundaries.setdefault(row[field], row["split"])
            require(prior == row["split"], f"{task}: {field} crossed splits")
    for row in manifest:
        parent = parents[row["parent_sample_id"]]
        require(parent["split"] == row["split"], f"{task}: parent crossed splits")
        require(parent["sha256"] == row["parent_sha256"], f"{task}: parent identity")
        augmented = row["sample_id"] != row["parent_sample_id"]
        require(augmented == (row["is_augmented"].lower() == "true"), f"{task}: augmentation flag")
        require(not augmented or row["split"] == "train", f"{task}: augmented holdout")
    split_hash = protocol.get("manifest_sha256") or protocol["evidence"]["manifest_sha256"]
    require(digest(root / "split-manifest.csv") == split_hash, f"{task}: manifest digest")
    require(digest(root / "protocol.json") == frozen["protocol_sha256"], f"{task}: protocol digest")
    require(digest(root / "frozen-linear-head.joblib") == frozen["model_sha256"], f"{task}: model digest")
    test = {key: r for key, r in split.items() if r["split"] == "test"}
    predictions = read_csv(root / "test-predictions.csv")
    require(len(predictions) == len(test) and {r["sample_id"] for r in predictions} == test.keys(),
            f"{task}: test predictions do not match split")
    return predictions, test, read_json(root / "test-metrics.json"), frozen


def verify_lesion():
    predictions, test, metrics, _ = verify_run(
        "lesion", {"train": 1300, "val": 35, "test": 35}, {"train": 130, "val": 35, "test": 35}
    )
    require(all(r["label"] == test[r["sample_id"]]["label"] for r in predictions), "Lesion: test label differs")
    matrix = [[sum(r["label"] == truth and r["prediction"] == pred for r in predictions)
               for pred in CLASSES] for truth in CLASSES]
    correct = sum(matrix[i][i] for i in range(len(CLASSES)))
    f1 = sum(2 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix))
             for i in range(len(CLASSES))) / len(CLASSES)
    require(matrix == metrics["confusion_matrix"], "Lesion: confusion matrix")
    require(math.isclose(correct / len(predictions), metrics["accuracy"]), "Lesion: accuracy")
    require(math.isclose(f1, metrics["macro_f1"]), "Lesion: macro-F1")
    print(f"Lesion: {correct}/{len(predictions)} correct, macro-F1 {f1:.3f}")


def verify_wall():
    predictions, test, metrics, frozen = verify_run(
        "wall-distance", {"train": 2000, "val": 40, "test": 64}, {"train": 208, "val": 40, "test": 64}
    )
    for r in predictions:
        row = test[r["sample_id"]]
        require(r["sha256"] == row["sha256"] and float(r["distance_true_mm"]) == float(row["distance"]),
                "Wall distance: test image or truth differs")
    for group in [metrics, *metrics["per_interval"]]:
        lower = group.get("lower_mm", 30)
        upper = group.get("upper_mm", 60)
        errors, baseline = [], []
        for r in predictions:
            truth = float(r["distance_true_mm"])
            if lower <= truth < upper or truth == upper == 60:
                errors.append(float(r["distance_predicted_mm"]) - truth)
                baseline.append(frozen["train_target_mean_mm"] - truth)
        require(len(errors) == group["n"], "Wall distance: interval count")
        for values, expected in ((errors, group), (baseline, group["mean_baseline"])):
            mae = sum(map(abs, values)) / len(values)
            rmse = math.sqrt(sum(x * x for x in values) / len(values))
            require(math.isclose(mae, expected["mae"]), "Wall distance: MAE")
            require(math.isclose(rmse, expected["rmse"]), "Wall distance: RMSE")
    print(f'Wall distance: {metrics["n"]} test frames, MAE {metrics["mae"]:.4f} mm, RMSE {metrics["rmse"]:.4f} mm')


def main():
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, path = line.split("  ", 1)
        require(digest(ROOT / path) == expected, f"Checksum mismatch: {path}")
    verify_lesion()
    verify_wall()
    print("Published file checksums and split boundaries verified.")


if __name__ == "__main__":
    main()
