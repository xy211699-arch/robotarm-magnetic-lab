"""Paired test-only robustness assessment for the three frozen lesion classifiers."""

import argparse
from collections import Counter
import hashlib
import itertools
from io import BytesIO
from pathlib import Path
import re
import sys

import joblib
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits


ANGLES = (0, 90, 180, 270)
LEVELS = (-0.25, 0.0, 0.25)
MODELS = ("dinov3", "radio-v2.5-b", "siglip2-base")
ROTATE = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}
ORIGIN_COUNTS = {"normal": 20, "raised": 7, "ulcerated": 8}


def exposure_lut(ev):
    rgb = np.arange(256, dtype=np.float64) / 255
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    linear = np.clip(linear * (2 ** ev), 0, 1)
    s = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return np.rint(255 * s).astype(int).tolist() * 3


def condition(angle, exposure):
    if angle and exposure:
        return "rotation_and_exposure"
    if angle:
        return "rotation_only"
    if exposure:
        return "exposure_only"
    return "original"


def resolved_within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes its image root: {relative}")
    return path


def prepare(root, canonical, work, probe):
    clean = [row for row in canonical if row["split"] == "test"]
    probe.check(Counter(r["label"] for r in clean) == ORIGIN_COUNTS, "Unexpected original counts")
    probe.check(all(r["sample_id"] == r["parent_sample_id"] for r in clean), "Original input must contain originals")
    other_hashes = {r["sha256"] for r in canonical if r["split"] != "test"}
    store = work / "images"
    store.mkdir(exist_ok=False)
    variants = []
    for row in clean:
        sid = row["sample_id"]
        probe.check(re.fullmatch(r"[A-Za-z0-9_.-]+", sid), "Invalid sample ID")
        original_bytes = resolved_within(root, row["image_path"]).read_bytes()
        probe.check(hashlib.sha256(original_bytes).hexdigest() == row["sha256"], "Original identity changed")
        with Image.open(BytesIO(original_bytes)) as im:
            probe.check(im.size == (801, 801), "Unexpected original shape")
            image = im.convert("RGB")
        suffix = ("original", *(f"cw{a:03d}_ev{ev:+.2f}" for a, ev in itertools.product(ANGLES, LEVELS) if a or ev))
        combinations = ((0, 0.0), *((a, ev) for a, ev in itertools.product(ANGLES, LEVELS) if a or ev))
        seen = {hashlib.sha256(image.tobytes()).hexdigest()}
        for (angle, ev), tag in zip(combinations, suffix, strict=True):
            name = f"{sid}__{tag}" if tag != "original" else sid
            relative = f"images/{name}.png"
            target = work / relative
            if tag == "original":
                target.write_bytes(original_bytes)
            else:
                view = image.transpose(ROTATE[angle]) if angle else image
                if ev:
                    view = view.point(exposure_lut(ev))
                pixels = hashlib.sha256(view.tobytes()).hexdigest()
                probe.check(pixels not in seen, "Identical pixel data in a test family")
                seen.add(pixels)
                view.save(target, format="PNG", compress_level=2)
            sha = probe.digest(target)
            probe.check(sha not in other_hashes, "Test image bytes overlap training or validation")
            variants.append({
                "sample_id": name, "parent_sample_id": sid, "group_id": row["group_id"],
                "label": row["label"], "split": "test", "sha256": sha, "parent_sha256": row["sha256"],
                "image_path": relative, "source_image_path": row["image_path"],
                "rotation_cw_deg": angle, "exposure_ev": ev, "condition": condition(angle, ev),
            })
    counts = Counter(row["condition"] for row in variants)
    probe.check(counts == {"original": 35, "exposure_only": 70, "rotation_only": 105, "rotation_and_exposure": 210},
                "Unexpected condition counts")
    probe.check(len({r["sha256"] for r in variants}) == len(variants), "Duplicate file content in the test suite")
    probe.write_csv(work / "test-manifest.csv", variants)
    probe.write_json(work / "test-protocol.json", {
        "manifest_sha256": probe.digest(work / "test-manifest.csv"),
        "frozen_baseline_sha256": probe.digest(work / "frozen-baseline.json"),
        "source_split_sha256": probe.DATASETS["lesion"]["sha256"], "source_original_counts": ORIGIN_COUNTS,
        "augmentation_only": "Test-only; no model fitting or configuration selection.",
        "angles_clockwise_degrees": ANGLES, "exposure_ev": LEVELS,
        "exposure": "Decode sRGB, multiply linear RGB by 2**EV, clip and re-encode sRGB.",
        "n_unique_originals": len(clean), "n_variants": len(variants) - len(clean), "condition_counts": dict(counts),
        "scope": "Within-source lesion viewing robustness. Transformed views are paired with 35 existing originals.",
    })
    print(f"Prepared {len(variants)} test images from {len(clean)} originals", flush=True)


def load_dino(lab, protocol, selected, probe):
    import torch

    sys.path.insert(0, str(lab / "prep"))
    from training_prep import load_backbone, transform_from

    cfg = protocol["config"]
    probe.check(cfg["image_size"] == probe.SIZE and cfg["precision"] == "bf16", "Unexpected DINOv3 protocol")
    model = load_backbone(cfg)
    transform = transform_from(cfg)
    layer = selected["cls_layer"]
    key = f"block_{layer}_cls"

    def preprocess(images):
        return torch.stack([transform(image) for image in images])

    def embedding(batch):
        (_, cls), = model.get_intermediate_layers(batch, n=[layer - 1], return_class_token=True, norm=True)
        return {key: cls.float()}

    return preprocess, embedding, key


def test_model(args, work, canonical, locked, probe):
    import csv
    import torch

    model_name = args.encoder
    if model_name == "dinov3":
        source = args.lab / "runs/dinov3-lesion-within-source-20261120"
    else:
        source = args.comparison / "runs" / model_name / "lesion"
    for relative, sha in locked["models"][model_name].items():
        probe.verify_bytes(source / relative, sha)
    meta = probe.read_json(source / "protocol.json")
    selection = probe.read_json(source / "frozen-selection.json")
    manifest_hash = meta.get("manifest_sha256") or meta["evidence"]["manifest_sha256"]
    probe.check(manifest_hash == probe.DATASETS["lesion"]["sha256"], "Model was validated on different data")
    probe.verify_bytes(source / "frozen-linear-head.joblib", selection["model_sha256"])
    protocol = probe.read_json(work / "test-protocol.json")
    probe.verify_bytes(work / "test-manifest.csv", protocol["manifest_sha256"])
    probe.verify_bytes(work / "frozen-baseline.json", protocol["frozen_baseline_sha256"])
    with (work / "test-manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    test_originals = {row["sample_id"]: (row["sha256"], row["label"], row["group_id"])
                      for row in canonical if row["split"] == "test"}
    probe.check(Counter(r["parent_sample_id"] for r in rows) == dict.fromkeys(test_originals, 12),
                "Unexpected test family membership")
    probe.check(all((r["parent_sha256"], r["label"], r["group_id"]) == test_originals[r["parent_sample_id"]]
                    and r["split"] == "test" for r in rows), "Changed parent, label or split")
    torch.set_num_threads(6)
    if model_name == "dinov3":
        preprocess, embedding, readout = load_dino(args.lab, meta, selection, probe)
    else:
        preprocess, embedding, observed = probe.load_encoder(model_name, args.comparison.resolve())
        probe.check(observed["weights_sha256"] == meta["backbone"]["weights_sha256"], "Weights do not match the frozen run")
        for field in ("source_commit", "config_sha256", "processor_sha256"):
            probe.check(observed.get(field) == meta["backbone"].get(field), f"Runtime identity mismatch: {field}")
        readout = selection["readout"]
    out = work / model_name
    out.mkdir(exist_ok=False)
    features = probe.extract(rows, work, preprocess, embedding, (readout,), out, "test")
    head = joblib.load(source / "frozen-linear-head.joblib")
    with threadpool_limits(limits=6):
        pred = head.predict(features[readout].astype(np.float64))
    expected_path = source / ("test/predictions.csv")
    with expected_path.open(newline="", encoding="utf-8") as file:
        previous = {r["sample_id"]: r["prediction"] for r in csv.DictReader(file)}
    records = []
    for r, value in zip(rows, pred, strict=True):
        if r["condition"] == "original":
            probe.check(value == previous[r["sample_id"]], "Original-frame predictions did not reproduce")
        records.append({
            "sample_id": r["sample_id"], "parent_sample_id": r["parent_sample_id"],
            "group_id": r["group_id"], "truth": r["label"], "prediction": str(value),
            "rotation_cw_deg": r["rotation_cw_deg"], "exposure_ev": r["exposure_ev"], "condition": r["condition"],
        })
    probe.write_csv(out / "predictions.csv", records)
    probe.write_json(out / "identity.json", {
        "model": model_name, "readout": readout, "model_sha256": selection["model_sha256"],
        "test_manifest_sha256": protocol["manifest_sha256"], "original_predictions_reproduced": len(previous),
    })
    print(f"{model_name}: froze the existing head and reproduced {len(previous)} original predictions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "test"))
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--encoder", choices=MODELS)
    args = parser.parse_args()
    work = args.work.resolve()
    sys.path.insert(0, str(args.comparison))
    import encoder_probe as probe

    locked = probe.read_json(work / "frozen-baseline.json")
    probe.verify_bytes(args.comparison / "encoder_probe.py", locked["shared_probe_sha256"])
    canonical = probe.read_split(args.lab / probe.DATASETS["lesion"]["manifest"], "lesion")
    if args.stage == "prepare":
        prepare((args.lab / "datasets/raw/gastric-aug-light-roll-v1").resolve(), canonical, work, probe)
    else:
        if not args.encoder:
            parser.error("--encoder is required for the test stage")
        test_model(args, work, canonical, locked, probe)


if __name__ == "__main__":
    main()
