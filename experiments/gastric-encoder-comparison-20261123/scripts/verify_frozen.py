import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

root = Path(sys.argv[1])
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
for model in ("radio-v2.5-b", "siglip2-base"):
    for task in ("lesion", "wall-distance"):
        run = root / "runs" / model / task
        frozen = json.loads((run / "frozen-selection.json").read_text())
        assert digest(run / "protocol.json") == frozen["protocol_sha256"]
        assert digest(run / "frozen-linear-head.joblib") == frozen["model_sha256"]
        assert digest(run / "train-val-features.npz") == frozen["features_sha256"]
        with (run / "test/predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        head = joblib.load(run / "frozen-linear-head.joblib")
        with np.load(run / "test/test-features.npz", allow_pickle=False) as features:
            assert list(features["sample_id"]) == [row["sample_id"] for row in rows]
            with threadpool_limits(limits=6):
                predictions = head.predict(features[frozen["readout"]].astype(np.float64))
        if task == "lesion":
            assert list(predictions) == [row["prediction"] for row in rows]
        else:
            np.testing.assert_allclose(predictions, [float(row["prediction"]) for row in rows], atol=1e-11, rtol=1e-12)
        print(f"{model}/{task}: {len(rows)} stored predictions reproduced from the frozen head and test features")
