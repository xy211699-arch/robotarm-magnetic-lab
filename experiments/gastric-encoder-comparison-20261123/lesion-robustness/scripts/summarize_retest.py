"""Recompute lesion robustness statistics from the test manifest and predictions."""

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys


MODELS = ("dinov3", "radio-v2.5-b", "siglip2-base")
CLASSES = ("normal", "raised", "ulcerated")
ANGLES = (0, 90, 180, 270)
EXPOSURES = (-0.25, 0.0, 0.25)
EXPECTED = {"original": 35, "exposure_only": 70, "rotation_only": 105, "rotation_and_exposure": 210}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, description):
    if not condition:
        raise ValueError(description)


def metrics(rows):
    matrix = {actual: {prediction: 0 for prediction in CLASSES} for actual in CLASSES}
    for row in rows:
        matrix[row["truth"]][row["prediction"]] += 1
    correct = sum(matrix[c][c] for c in CLASSES)
    f1 = sum(2 * matrix[c][c] / (sum(matrix[c].values()) + sum(matrix[d][c] for d in CLASSES)) for c in CLASSES) / 3
    return {"n": len(rows), "correct": correct, "errors": len(rows) - correct,
            "accuracy": correct / len(rows), "macro_f1": f1}, matrix


def main():
    root, baseline = map(Path, sys.argv[1:])
    protocol = read_json(root / "test-protocol.json")
    lock = read_json(root / "frozen-baseline.json")
    require(sha(root / "frozen-baseline.json") == protocol["frozen_baseline_sha256"], "Frozen model registry changed")
    require(sha(root / "test-manifest.csv") == protocol["manifest_sha256"], "Test manifest changed")
    require(sha(baseline) == protocol["source_split_sha256"], "The original split changed")
    canonical = read_csv(baseline)
    non_test_hashes = {r["sha256"] for r in canonical if r["split"] != "test"}
    original = {r["sample_id"]: r for r in canonical if r["split"] == "test"}
    manifest = read_csv(root / "test-manifest.csv")
    by_id = {r["sample_id"]: r for r in manifest}
    require(len(manifest) == len(by_id) == 420, "The test suite must have 420 unique records")
    require(Counter(r["condition"] for r in manifest) == EXPECTED, "The condition counts changed")
    require(len({r["sha256"] for r in manifest}) == len(manifest), "Duplicate image bytes")
    families = defaultdict(set)
    for row in manifest:
        parent = row["parent_sample_id"]
        source = original[parent]
        require(row["split"] == "test" and row["sha256"] not in non_test_hashes, "Split overlap found")
        require((row["parent_sha256"], row["label"], row["group_id"]) ==
                (source["sha256"], source["label"], source["group_id"]), "Wrong parent identity or label")
        if row["condition"] == "original":
            require(row["sha256"] == source["sha256"] and row["sample_id"] == parent, "Wrong untouched original")
        families[parent].add((int(row["rotation_cw_deg"]), float(row["exposure_ev"])))
    grid = set(itertools.product(ANGLES, EXPOSURES))
    require(families.keys() == original.keys() and all(values == grid for values in families.values()),
            "The expected transformation grid is incomplete")

    overview, condition_detail, parent_detail, confusion, incorrect = [], [], [], [], []
    for model in MODELS:
        identity = read_json(root / model / "identity.json")
        require(identity["test_manifest_sha256"] == protocol["manifest_sha256"] and
                identity["model_sha256"] == lock["models"][model]["frozen-linear-head.joblib"] and
                identity["original_predictions_reproduced"] == 35, "Wrong model or manifest")
        rows = read_csv(root / model / "predictions.csv")
        require(len(rows) == len(by_id) and {r["sample_id"] for r in rows} == by_id.keys(), "Wrong prediction identities")
        for row in rows:
            source = by_id[row["sample_id"]]
            shared = ("parent_sample_id", "group_id", "rotation_cw_deg", "exposure_ev", "condition")
            require(all(row[k] == source[k] for k in shared) and row["truth"] == source["label"], "Changed test labels or transforms")
        perturbed = [r for r in rows if r["condition"] != "original"]
        for label in EXPECTED:
            scores, _ = metrics([r for r in rows if r["condition"] == label])
            overview.append({"encoder": model, "condition": label, **scores})
        for label, subset in (("augmentation_only", perturbed), ("all_views", rows)):
            scores, matrix = metrics(subset)
            overview.append({"encoder": model, "condition": label, **scores})
            for actual in CLASSES:
                for prediction in CLASSES:
                    confusion.append({"encoder": model, "condition": label, "truth": actual,
                                      "prediction": prediction, "n": matrix[actual][prediction]})
        for angle, ev in sorted(grid):
            scores, _ = metrics([r for r in rows if (int(r["rotation_cw_deg"]), float(r["exposure_ev"])) == (angle, ev)])
            condition_detail.append({"encoder": model, "rotation_cw_deg": angle, "exposure_ev": ev, **scores})
        wrong_by_parent = Counter()
        for row in perturbed:
            if row["prediction"] != row["truth"]:
                wrong_by_parent[row["parent_sample_id"]] += 1
                incorrect.append({"encoder": model, **row})
        for parent, info in original.items():
            n_wrong = wrong_by_parent[parent]
            parent_detail.append({"encoder": model, "parent_sample_id": parent, "group_id": info["group_id"],
                                  "truth": info["label"], "augmented_views": 11, "errors": n_wrong,
                                  "all_augmented_views_correct": n_wrong == 0})
        total, _ = metrics(perturbed)
        print(f'{model}: {total["correct"]}/{total["n"]} augmented, Macro-F1 {total["macro_f1"]:.6f}, '
              f'{len(original) - len(wrong_by_parent)}/{len(original)} originals correct in every transformed view')
    write_csv(root / "summary.csv", overview)
    write_csv(root / "by-transform.csv", condition_detail)
    write_csv(root / "by-original.csv", parent_detail)
    write_csv(root / "confusion-counts.csv", confusion)
    if incorrect:
        write_csv(root / "misclassifications.csv", incorrect)


if __name__ == "__main__":
    main()
