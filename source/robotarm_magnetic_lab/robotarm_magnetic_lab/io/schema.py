"""Canonical constants and validation for the robotarm magnetic policy API."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


INTERFACE_SCHEMA_VERSION = "1.0.0"
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
    if spec.get("schema_version") != INTERFACE_SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema {INTERFACE_SCHEMA_VERSION}, got {spec.get('schema_version')!r}"
        )
    if tuple(spec["inputs"]["joint_order"]) != JOINT_NAMES:
        raise ValueError("Interface joint order does not match the runtime contract")
    if int(spec["outputs"]["action"]["shape"][0]) != ACTION_DIM:
        raise ValueError("Interface action dimension must be 9")
    if int(spec["inputs"]["policy_state"]["shape"][0]) != POLICY_STATE_DIM:
        raise ValueError("Interface policy-state dimension must be 31")
    return spec, hashlib.sha256(raw).hexdigest()
