"""Canonical constants and validation for the robotarm magnetic policy API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


INTERFACE_SCHEMA_VERSION = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0", INTERFACE_SCHEMA_VERSION}
JOINT_NAMES = (
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
    "j6",
    "ballxj",
    "ballyj",
    "ballzj",
)
ACTION_DIM = 9
POLICY_STATE_DIM = 31
RGB_SHAPE = (720, 1280, 3)
DEPTH_SHAPE = (720, 1280)


def load_interface_spec(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and structurally validate an interface JSON file.

    Returns the decoded specification and its SHA-256 digest.  The digest is
    copied into every dataset manifest so a run cannot silently mix schemas.
    """
    path = Path(path)
    raw = path.read_bytes()
    spec = json.loads(raw)
    version = spec.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Supported schemas are {sorted(SUPPORTED_SCHEMA_VERSIONS)}, got {version!r}"
        )
    if tuple(spec["inputs"]["joint_order"]) != JOINT_NAMES:
        raise ValueError("Interface joint order does not match the runtime contract")
    if int(spec["outputs"]["action"]["shape"][0]) != ACTION_DIM:
        raise ValueError("Interface action dimension must be 9")
    if int(spec["inputs"]["policy_state"]["shape"][0]) != POLICY_STATE_DIM:
        raise ValueError("Interface policy-state dimension must be 31")
    if version == "2.0.0":
        rates = spec["rates_hz"]
        if int(rates["camera"]) != 1 or int(rates["policy"]) != 1:
            raise ValueError("Schema v2 requires 1 Hz camera and policy inference")
        if int(rates["control"]) != 20 or int(rates["physics"]) != 240:
            raise ValueError("Schema v2 requires 20 Hz control and 240 Hz physics")
        timing = spec["timing"]
        if int(timing["nominal_camera_to_control_ratio"]) != 20:
            raise ValueError("Schema v2 requires a nominal 20 control rows per camera frame")
        if int(timing["camera_stride_tolerance_control_steps"]) != 1:
            raise ValueError("Schema v2 must declare the measured +/-1 control-step jitter")
        chunk = spec["outputs"]["action_chunk_default"]
        if int(chunk["horizon_steps"]) != 20 or float(chunk["duration_s"]) != 1.0:
            raise ValueError("Schema v2 requires a 20-step, 1-second action chunk")
    return spec, hashlib.sha256(raw).hexdigest()
