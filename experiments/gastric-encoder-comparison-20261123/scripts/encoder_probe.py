"""Frozen vision linear probes on the immutable DINOv3 project splits."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


SEED = 42
BATCH = 32
SIZE = 256
CLASSES = ("normal", "raised", "ulcerated")
READOUTS = ("summary", "patch_mean")
DATASETS = {
    "lesion": {
        "manifest": "runs/dinov3-lesion-within-source-20261120/split-manifest.csv",
        "sha256": "819a5163a9f7d0771656c7c12ab814bb4bf2f09b75f140cf9408bc3fdafae89b",
        "counts": {"train": 1300, "val": 35, "test": 35},
        "originals": {"train": 130, "val": 35, "test": 35},
        "search": (0.01, 0.1, 1.0, 10.0, 100.0),
        "selection": "Maximize validation macro-F1, then accuracy, then favor summary and smaller C.",
    },
    "wall-distance": {
        "manifest": "runs/dinov3-wall-cap40-aug-20261119/manifest.csv",
        "sha256": "5ff80ce9d3de66e570f92458fab10442ab679c6a326a9b6b1b41e859771ba4f2",
        "counts": {"train": 2000, "val": 40, "test": 64},
        "originals": {"train": 208, "val": 40, "test": 64},
        "search": (0.1, 1.0, 10.0, 100.0, 1000.0),
        "selection": "Minimize validation MAE, then RMSE, then favor summary and smaller alpha.",
    },
}
RADIO_CODE = "4ffaf15d0ff48197c194e306772bdb5fb92db230"
RADIO_REV = "5409f826455e642d64d8beeeef14e67e0e8e9709"
RADIO_SHA = "6bff4bd732d815136652d454598e8fe6c6f4e658716d5e6a697f0e6b60bd8a98"
SIGLIP_SHA = "6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bytes(path, expected):
    check(digest(path) == expected, f"SHA-256 mismatch: {path}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_split(path, task):
    specification = DATASETS[task]
    verify_bytes(path, specification["sha256"])
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    check(Counter(r["split"] for r in rows) == specification["counts"], f"{task}: unexpected split size")
    check(len({r["sample_id"] for r in rows}) == len(rows), f"{task}: duplicate sample ID")
    origins = {r["sample_id"]: r for r in rows if r["sample_id"] == r["parent_sample_id"]}
    check(Counter(r["split"] for r in origins.values()) == specification["originals"], f"{task}: original count")
    for field in ("group_id", "image_path", "sha256", "parent_sample_id"):
        split_by_key = {}
        for row in rows:
            check(split_by_key.setdefault(row[field], row["split"]) == row["split"], f"{task}: shared {field}")
    for row in rows:
        origin = origins[row["parent_sample_id"]]
        augmented = row["sample_id"] != row["parent_sample_id"]
        check(origin["sha256"] == row["parent_sha256"], f"{task}: changed parent")
        check(augmented == (row["is_augmented"].lower() == "true"), f"{task}: invalid enhancement flag")
        check(not augmented or row["split"] == "train", f"{task}: augmentation in holdout")
        key = "label" if task == "lesion" else "distance"
        check(row[key] == origin[key], f"{task}: changed target")
    return rows


def load_encoder(encoder, job):
    import torch
    from torchvision.transforms import InterpolationMode, functional

    if encoder == "radio-v2.5-b":
        repo = job / "radio-src"
        revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        check(revision == RADIO_CODE, "RADIO source revision changed")
        check(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"],
                                          text=True).strip(), "RADIO source has local changes")
        weight_dir = job / "torch/hub"
        verify_bytes(weight_dir / "checkpoints/radio-v2.5-b_half.pth.tar", RADIO_SHA)
        torch.hub.set_dir(str(weight_dir))
        model = torch.hub.load(str(repo), "radio_model", source="local", version="radio_v2.5-b", progress=False)

        def preprocess(images):
            return torch.stack([
                functional.to_tensor(functional.resize(im, [SIZE, SIZE], InterpolationMode.BILINEAR, antialias=True))
                for im in images
            ])

        def embedding(batch):
            summary, patches = model(batch)
            return {"summary": summary.float(), "patch_mean": patches.float().mean(1)}

        metadata = {
            "model_id": "NVlabs/RADIO radio_v2.5-b",
            "license": "NVIDIA Source Code License",
            "weights_source": f"https://huggingface.co/nvidia/RADIO/tree/{RADIO_REV}",
            "weights_sha256": RADIO_SHA,
            "source_commit": revision,
            "preprocess": "RGB square bilinear antialias 256, [0, 1] input to official RADIO internal conditioning.",
            "readouts": {"summary": "Official final backbone summary, no teacher adaptor.",
                         "patch_mean": "Mean of official final backbone spatial tokens."},
        }
    else:
        from transformers import AutoImageProcessor, SiglipConfig, SiglipModel

        folder = job / "siglip2"
        verify_bytes(folder / "model.safetensors", SIGLIP_SHA)
        revision = (folder / "revision.txt").read_text().strip()
        cfg = SiglipConfig.from_pretrained(folder, local_files_only=True)
        check(cfg.vision_config.image_size == SIZE and cfg.vision_config.patch_size == 16,
              "Unexpected SigLIP2 vision configuration")
        processor = AutoImageProcessor.from_pretrained(folder, local_files_only=True, use_fast=False)
        model = SiglipModel.from_pretrained(folder, local_files_only=True, use_safetensors=True).vision_model

        def preprocess(images):
            return processor(images=images, return_tensors="pt")["pixel_values"]

        def embedding(batch):
            output = model(pixel_values=batch)
            return {"summary": output.pooler_output.float(), "patch_mean": output.last_hidden_state.float().mean(1)}

        metadata = {
            "model_id": "google/siglip2-base-patch16-256",
            "license": "Apache-2.0",
            "weights_source": f"https://huggingface.co/google/siglip2-base-patch16-256/tree/{revision}",
            "weights_sha256": SIGLIP_SHA,
            "source_commit": revision,
            "config_sha256": digest(folder / "config.json"),
            "processor_sha256": digest(folder / "preprocessor_config.json"),
            "processor_config": read_json(folder / "preprocessor_config.json"),
            "preprocess": "Official Transformers vision image processor, 256 × 256.",
            "readouts": {"summary": "Official final attention-pooler output of the vision tower.",
                         "patch_mean": "Mean of the final post-layer-norm spatial tokens."},
        }
    check(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 is required for comparable timing")
    model.eval().requires_grad_(False).to("cuda")
    metadata["vision_parameters"] = sum(p.numel() for p in model.parameters())
    metadata["software"] = {name: importlib.metadata.version(name)
                            for name in ("torch", "torchvision", "Pillow", "numpy", "scikit-learn", "timm", "transformers")}
    return preprocess, embedding, metadata


def image_batch(root, rows):
    images = []
    for row in rows:
        path = (root / row["image_path"]).resolve()
        check(path.is_relative_to(root), f"Path outside image root: {path}")
        raw = path.read_bytes()
        check(hashlib.sha256(raw).hexdigest() == row["sha256"], f"Image identity mismatch: {path}")
        with Image.open(BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
    return images


def extract(rows, image_root, preprocess, embedding, readouts, directory, tag):
    import torch

    collected = {readout: [] for readout in readouts}

    def forward(chunk):
        batch = preprocess(image_batch(image_root, chunk))
        check(batch.shape == (len(chunk), 3, SIZE, SIZE), f"Unexpected input shape: {batch.shape}")
        return embedding(batch.to("cuda"))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        warmup = rows[:BATCH]
        for _ in range(2):
            forward(warmup)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            output = forward(chunk)
            for name in readouts:
                values = output[name].cpu().numpy()
                check(values.ndim == 2 and values.shape[0] == len(chunk) and values.shape[1] > 0
                      and np.isfinite(values).all(), f"Invalid features: {name}")
                collected[name].append(values)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    features = {name: np.concatenate(chunks) for name, chunks in collected.items()}
    for values in features.values():
        if len(values) > 1:
            check(np.any(np.ptp(values, axis=0) > 0), "Degenerate features: all frames produced identical vectors")
    metrics = {
        "image_count": len(rows), "batch_size": BATCH, "image_size": SIZE, "warmup_batches": 2,
        "precision": "BF16 autocast with FP32 pooled features",
        "feature_dimensions": {name: values.shape[1] for name, values in features.items()},
        "seconds_extraction_including_io_sha_and_decode": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(), "cuda_runtime": torch.version.cuda,
    }
    np.savez_compressed(directory / f"{tag}-features.npz", sample_id=np.array([r["sample_id"] for r in rows]), **features)
    write_json(directory / f"{tag}-extraction.json", metrics)
    print(f"{directory.name}/{tag}: {len(rows)} frames in {elapsed:.2f} s", flush=True)
    return features


def labels(rows, task):
    if task == "lesion":
        return np.array([row["label"] for row in rows])
    return np.array([float(row["distance"]) for row in rows], dtype=np.float64)


def errors(truth, predicted):
    err = predicted - truth
    return {"mae": float(np.abs(err).mean()), "rmse": math.sqrt(float((err ** 2).mean()))}


def score(task, truth, pred, train_mean=None):
    if task == "lesion":
        return {
            "accuracy": float(accuracy_score(truth, pred)),
            "macro_f1": float(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)),
            "class_names": list(CLASSES),
            "support": {c: int(np.sum(truth == c)) for c in CLASSES},
            "confusion_matrix": confusion_matrix(truth, pred, labels=CLASSES).tolist(),
        }
    return {
        **errors(truth, pred), "signed_bias_mm": float((pred - truth).mean()),
        "mean_baseline": errors(truth, np.full_like(truth, train_mean)), "n": len(truth), "unit": "mm",
    }


def prediction_rows(rows, truth, values):
    return [{"sample_id": row["sample_id"], "group_id": row["group_id"], "image_sha256": row["sha256"],
             "truth": y.item(), "prediction": p.item()}
            for row, y, p in zip(rows, truth, values, strict=True)]


def build_head(task, value):
    if task == "lesion":
        return make_pipeline(StandardScaler(), LogisticRegression(C=value, max_iter=2000, random_state=SEED))
    return make_pipeline(StandardScaler(), Ridge(alpha=value, solver="svd"))


def validate(task, rows, directory, image_root, encoder, protocol):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol)
    fitted = [row for row in rows if row["split"] != "test"]
    is_train = np.array([row["split"] == "train" for row in fitted])
    val_rows = [row for row in fitted if row["split"] == "val"]
    y = labels(fitted, task)
    train_mean = None if task == "lesion" else float(y[is_train].mean())
    features = extract(fitted, image_root, *encoder[:2], READOUTS, directory, "train-val")
    trials = []
    best = None
    with threadpool_limits(limits=6):
        for i, name in enumerate(READOUTS):
            x = features[name].astype(np.float64)
            for strength in DATASETS[task]["search"]:
                model = build_head(task, strength)
                start = time.perf_counter()
                model.fit(x[is_train], y[is_train])
                elapsed = time.perf_counter() - start
                if task == "lesion":
                    check(np.max(model[-1].n_iter_) < 2000, "Logistic regression did not converge")
                check(np.allclose(model[0].mean_, x[is_train].mean(0)), "Scaler must use training data only")
                predicted = model.predict(x[~is_train])
                metrics = score(task, y[~is_train], predicted, train_mean)
                if task == "lesion":
                    summary = {key: metrics[key] for key in ("macro_f1", "accuracy")}
                    order = (-metrics["macro_f1"], -metrics["accuracy"], i, strength)
                    regularization = {"C": strength}
                else:
                    summary = {key: metrics[key] for key in ("mae", "rmse")}
                    order = (metrics["mae"], metrics["rmse"], i, strength)
                    regularization = {"alpha": strength}
                trial = {"readout": name, **regularization, **summary, "fit_seconds": elapsed}
                trials.append(trial)
                if best is None or order < best[0]:
                    best = (order, model, trial, metrics, predicted)
    _, head, selected, metrics, predicted = best
    write_csv(directory / "validation-search.csv", trials)
    write_csv(directory / "validation-predictions.csv", prediction_rows(val_rows, y[~is_train], predicted))
    joblib.dump(head, directory / "frozen-linear-head.joblib")
    frozen = {
        **selected, "validation": metrics, "train_target_mean_mm": train_mean,
        "protocol_sha256": digest(directory / "protocol.json"),
        "model_sha256": digest(directory / "frozen-linear-head.joblib"),
        "features_sha256": digest(directory / "train-val-features.npz"),
    }
    write_json(directory / "frozen-selection.json", frozen)


def evaluate(task, rows, directory, image_root, encoder, protocol):
    selected = read_json(directory / "frozen-selection.json")
    for name, field in (("protocol.json", "protocol_sha256"), ("frozen-linear-head.joblib", "model_sha256"),
                        ("train-val-features.npz", "features_sha256")):
        verify_bytes(directory / name, selected[field])
    check(read_json(directory / "protocol.json") == protocol, "Validation and test configuration differ")
    target = directory / "test"
    target.mkdir(exist_ok=False)
    rows = [row for row in rows if row["split"] == "test"]
    readout = selected["readout"]
    features = extract(rows, image_root, *encoder[:2], (readout,), target, "test")
    head = joblib.load(directory / "frozen-linear-head.joblib")
    with threadpool_limits(limits=6):
        predicted = head.predict(features[readout].astype(np.float64))
    truth = labels(rows, task)
    mean = selected["train_target_mean_mm"]
    metrics = score(task, truth, predicted, mean)
    if task == "wall-distance":
        metrics["per_interval"] = []
        for low, high in ((30, 40), (40, 50), (50, 60)):
            within = (truth >= low) & ((truth <= high) if high == 60 else (truth < high))
            metrics["per_interval"].append({
                "lower_mm": low, "upper_mm": high, **score(task, truth[within], predicted[within], mean)
            })
    else:
        groups = np.array([row["group_id"] for row in rows])
        metrics["group_accuracy"] = [
            {"group_id": group, "n": int(np.sum(groups == group)),
             "accuracy": float(np.mean(predicted[groups == group] == truth[groups == group]))}
            for group in sorted(set(groups))
        ]
    write_csv(target / "predictions.csv", prediction_rows(rows, truth, predicted))
    write_json(target / "metrics.json", {
        **metrics, "protocol_sha256": selected["protocol_sha256"], "model_sha256": selected["model_sha256"],
        "frozen_selection_sha256": digest(directory / "frozen-selection.json"), "test_command": sys.argv,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "smoke", "validate", "test"))
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--encoder", choices=("radio-v2.5-b", "siglip2-base"), required=True)
    args = parser.parse_args()
    datasets = {task: read_split(args.lab / spec["manifest"], task) for task, spec in DATASETS.items()}
    if args.phase == "audit":
        for task, rows in datasets.items():
            print(task, len(rows), DATASETS[task]["sha256"])
        return
    import torch

    torch.set_num_threads(6)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    image_root = (args.lab / "datasets/raw/gastric-aug-light-roll-v1").resolve()
    encoder = load_encoder(args.encoder, args.job.resolve())
    run = args.job / "runs" / args.encoder
    if args.phase == "smoke":
        smoke = run / "smoke"
        smoke.mkdir(parents=True, exist_ok=False)
        train = [row for row in datasets["lesion"] if row["split"] == "train"][:4]
        extract(train, image_root, *encoder[:2], READOUTS, smoke, "smoke")
        write_json(smoke / "backbone.json", encoder[2])
        return
    for task, rows in datasets.items():
        spec = DATASETS[task]
        protocol = {
            "task": task, "manifest_sha256": spec["sha256"], "split_counts": spec["counts"],
            "original_split_counts": spec["originals"], "augmentation_only_in_train": True,
            "backbone": encoder[2], "seed": SEED, "script_sha256": digest(__file__),
            "candidates": list(spec["search"]), "selection": spec["selection"], "batch_size": BATCH,
            "scope": "Same-source lesion view holdout, capture labels pending semantic review."
                     if task == "lesion" else "Grouped simulation holdout; some scene IDs shared.",
            "linear_head": "StandardScaler + L2 logistic regression (LBFGS)" if task == "lesion"
                           else "StandardScaler + Ridge (SVD)",
            "fitted_split": "train", "hyperparameter_selection_split": "val",
        }
        operation = validate if args.phase == "validate" else evaluate
        operation(task, rows, run / task, image_root, encoder, protocol)


if __name__ == "__main__":
    main()
