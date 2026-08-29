#!/usr/bin/env python3
"""Read-only dependency and interface audit for TASK-010."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))


def _source_sha256(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode("utf-8")).hexdigest()


def _interface(owner: Any, method_name: str | None = None) -> dict[str, str]:
    value = getattr(owner, method_name) if method_name else owner
    return {
        "qualified_name": f"{value.__module__}.{value.__qualname__}",
        "signature": str(inspect.signature(value)),
        "source_sha256": _source_sha256(value),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kit_args", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import CNNModel, MLPModel, RNNModel
    from rsl_rl.modules import RNN
    from rsl_rl.runners import OnPolicyRunner
    from rsl_rl.storage import RolloutStorage
    from robotarm_magnetic_lab.runtime.task010_config import load_task010_config

    config = load_task010_config(args.config)
    interfaces = {
        name: _interface(owner, method)
        for name, owner, method in (
            ("OnPolicyRunner.learn", OnPolicyRunner, "learn"),
            ("OnPolicyRunner.save", OnPolicyRunner, "save"),
            ("PPO.act", PPO, "act"),
            ("PPO.process_env_step", PPO, "process_env_step"),
            ("PPO.update", PPO, "update"),
            ("RolloutStorage.recurrent_mini_batch_generator", RolloutStorage, "recurrent_mini_batch_generator"),
            ("RNN.forward", RNN, "forward"),
            ("RNN.reset", RNN, "reset"),
            ("RNNModel.forward", RNNModel, "forward"),
            ("MLPModel.forward", MLPModel, "forward"),
            ("CNNModel.forward", CNNModel, "forward"),
        )
    }
    weights = ResNet18_Weights.IMAGENET1K_V1
    payload = {
        "schema": "robotarm_magnetic_lab.task010_prerequisites",
        "config_sha256": config.config_sha256,
        "python": platform.python_version(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "torchvision", "rsl-rl-lib", "isaaclab")
        },
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torchvision_module_version": torchvision.__version__,
        "resnet18_weights": {
            "enum": "IMAGENET1K_V1",
            "available": weights is not None,
            "url": weights.url,
        },
        "interfaces": interfaces,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
