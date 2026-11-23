"""Audit paired evaluations against the original splits and write comparison CSVs."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
MODELS = ("dinov3", "radio-v2.5-b", "siglip2-base")
TASKS = ("lesion", "wall-distance")
CLASSES = ("normal", "raised", "ulcerated")


def need(condition, detail):
    if not condition:
        raise ValueError(detail)


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(x, y):
    return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)


def verified_predictions(file, expected, task):
    records = {r["sample_id"]: r for r in expected}
    rows = csv_rows(file)
    need(len(rows) == len(records) and {r["sample_id"] for r in rows} == records.keys(), f"{file}: wrong frames")
    truth, pred = [], []
    for row in rows:
        source = records[row["sample_id"]]
        need(row["group_id"] == source["group_id"], f"{file}: changed group")
        if task == "lesion":
            y = row.get("truth", row.get("label"))
            p = row["prediction"]
            need(y == source["label"] and p in CLASSES, f"{file}: changed or invalid class")
        else:
            y = float(row.get("truth", row.get("distance_true_mm")))
            p = float(row.get("prediction", row.get("distance_predicted_mm")))
            need(y == float(source["distance"]) and math.isfinite(p), f"{file}: changed distance or invalid prediction")
        if "image_sha256" in row or "sha256" in row:
            need(row.get("image_sha256", row.get("sha256")) == source["sha256"], f"{file}: changed image identity")
        truth.append(y)
        pred.append(p)
    return truth, pred


def error_metrics(truth, predictions):
    errors = [p - y for y, p in zip(truth, predictions, strict=True)]
    n = len(errors)
    return {"n": n, "mae": sum(map(abs, errors)) / n, "rmse": math.sqrt(sum(e * e for e in errors) / n),
            "signed_bias_mm": sum(errors) / n}


def verify_regression(truth, pred, reported, train_mean):
    model = error_metrics(truth, pred)
    baseline = error_metrics(truth, [train_mean] * len(truth))
    for key, value in model.items():
        need(close(value, reported[key]), f"Wrong regression metric: {key}")
    for key in ("mae", "rmse"):
        need(close(baseline[key], reported["mean_baseline"][key]), f"Wrong baseline metric: {key}")
    return model


def verify_metrics(task, truth, pred, reported, mean):
    if task == "lesion":
        matrix = [[sum(y == i and p == j for y, p in zip(truth, pred, strict=True)) for j in CLASSES] for i in CLASSES]
        n = len(truth)
        f1 = sum(2 * matrix[i][i] / (sum(matrix[i]) + sum(row[i] for row in matrix))
                 for i in range(len(CLASSES))) / len(CLASSES)
        accuracy = sum(matrix[i][i] for i in range(len(CLASSES))) / n
        need(matrix == reported["confusion_matrix"], "Wrong confusion matrix")
        need(close(accuracy, reported["accuracy"]) and close(f1, reported["macro_f1"]), "Wrong classification metric")
        return {"n": n, "accuracy": accuracy, "macro_f1": f1}
    return verify_regression(truth, pred, reported, mean)


def verify_selection(directory, task, selected):
    trials = csv_rows(directory / "validation-search.csv")
    strength = "C" if task == "lesion" else "alpha"
    need(len(trials) == 10, f"Incomplete validation search: {directory}")

    def order(row):
        rate = ((-float(row["macro_f1"]), -float(row["accuracy"])) if task == "lesion"
                else (float(row["mae"]), float(row["rmse"])))
        return *rate, row["readout"] != "summary", float(row[strength])

    best = min(trials, key=order)
    need(best["readout"] == selected["readout"] and float(best[strength]) == selected[strength],
         f"Frozen selection disagrees with validation search: {directory}")


def summarize(baseline, output):
    scores, intervals, clarity = [], [], []
    for task in TASKS:
        manifest = baseline / task / "split-manifest.csv"
        split = csv_rows(manifest)
        validation = [r for r in split if r["split"] == "val"]
        test = [r for r in split if r["split"] == "test"]
        train_mean = (sum(float(r["distance"]) for r in split if r["split"] == "train")
                      / sum(r["split"] == "train" for r in split)) if task == "wall-distance" else None
        for model in MODELS:
            old = model == "dinov3"
            directory = baseline / task if old else output / "runs" / model / task
            protocol = read_json(directory / "protocol.json")
            selected = read_json(directory / "frozen-selection.json")
            expected_hash = protocol.get("manifest_sha256") or protocol["evidence"]["manifest_sha256"]
            need(expected_hash == sha(manifest), "Manifest identity mismatch")
            need(sha(directory / "protocol.json") == selected["protocol_sha256"], "Protocol identity mismatch")
            need(sha(directory / "frozen-linear-head.joblib") == selected["model_sha256"], "Head identity mismatch")
            if old:
                feat = f'block_{selected.get("cls_layer", selected.get("layer"))}_cls'
                test_file = directory / "test-predictions.csv"
                metrics = read_json(directory / "test-metrics.json")
                eval_cost = read_json(directory / "test-extraction.json")
            else:
                feat = selected["readout"]
                test_file = directory / "test/predictions.csv"
                metrics = read_json(directory / "test/metrics.json")
                eval_cost = read_json(directory / "test/test-extraction.json")
                need(metrics["frozen_selection_sha256"] == sha(directory / "frozen-selection.json"),
                     "Test does not identify frozen validation selection")
                verify_selection(directory, task, selected)
                truth, pred = verified_predictions(directory / "validation-predictions.csv", validation, task)
                verify_metrics(task, truth, pred, selected["validation"], train_mean)
            truth, pred = verified_predictions(test_file, test, task)
            reported = verify_metrics(task, truth, pred, metrics, train_mean)
            fit_cost = read_json(directory / "train-val-extraction.json")
            scores.append({
                "encoder": model, "task": task, "readout": feat,
                "C": selected.get("C", ""), "alpha": selected.get("alpha", ""),
                "validation_n": len(validation), "test_n": len(test),
                "validation_accuracy": selected["validation"].get("accuracy", ""),
                "validation_macro_f1": selected["validation"].get("macro_f1", ""),
                "validation_mae_mm": selected["validation"].get("mae", ""),
                "validation_rmse_mm": selected["validation"].get("rmse", ""),
                "test_accuracy": reported.get("accuracy", ""), "test_macro_f1": reported.get("macro_f1", ""),
                "test_mae_mm": reported.get("mae", ""), "test_rmse_mm": reported.get("rmse", ""),
                "test_bias_mm": reported.get("signed_bias_mm", ""),
                "train_val_extraction_s": fit_cost["seconds_extraction_including_io_sha_and_decode"],
                "test_extraction_s": eval_cost["seconds_extraction_including_io_sha_and_decode"],
                "test_peak_allocated_mib": eval_cost["peak_allocated_bytes"] / (1024 ** 2),
                "test_peak_reserved_mib": eval_cost["peak_reserved_bytes"] / (1024 ** 2),
                "head_fit_s": selected["fit_seconds"], "test_gpu": eval_cost["gpu"],
            })
            if task == "lesion":
                for i, values in enumerate(metrics["confusion_matrix"]):
                    for j, count in enumerate(values):
                        clarity.append({"encoder": model, "truth": CLASSES[i], "prediction": CLASSES[j], "n": count})
            else:
                for group in metrics["per_interval"]:
                    low, high = group["lower_mm"], group["upper_mm"]
                    values = [(y, p) for y, p in zip(truth, pred, strict=True)
                              if low <= y < high or y == high == 60]
                    actual = verify_regression(*zip(*values, strict=True), group, train_mean)
                    intervals.append({"encoder": model, "lower_mm": low, "upper_mm": high, **actual,
                                      "mean_baseline_mae_mm": group["mean_baseline"]["mae"]})
    write_csv(output / "summary.csv", scores)
    write_csv(output / "wall-distance-intervals.csv", intervals)
    write_csv(output / "lesion-confusion.csv", clarity)
    for row in scores:
        task = row["task"]
        result = (f'Accuracy {row["test_accuracy"]:.4f}, macro-F1 {row["test_macro_f1"]:.4f}' if task == "lesion"
                  else f'MAE {row["test_mae_mm"]:.4f} mm, RMSE {row["test_rmse_mm"]:.4f} mm')
        print(f'{row["encoder"]} {task}: {result} (n={row["test_n"]})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=HERE.parent / "dinov3-gastric-20261120")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    summarize(args.baseline, args.output)


if __name__ == "__main__":
    main()
