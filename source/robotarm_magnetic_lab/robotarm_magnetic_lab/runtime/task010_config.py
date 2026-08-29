"""Strict, immutable TASK-010 CNN-GRU development configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


TASK010_SCHEMA = "robotarm_magnetic_lab.task010_cnn_gru_development"
TASK010_VERSION = 1
TASK010_TASK_ID = "Template-Robotarm-Magnetic-Task010-CNN-GRU-Coverage-Lab-v0"
TASK010_BASE_COMMIT = "1533bfa59f3d2d7b2f1769a9890efb354a5e4de6"
TASK010_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs/task010/cnn_gru_development_v1.json"


@dataclass(frozen=True)
class ClocksConfig:
    physics_hz: int
    control_hz: int
    physics_steps_per_action: int


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    crop_size: int
    model_input_size: int
    hz: int
    fov_deg: float


@dataclass(frozen=True)
class ModelConfig:
    resnet_weights: str
    visual_feature_dim: int
    actor_observation_dim: int
    critic_observation_dim: int
    visual_projection_dim: int
    action_feature_dim: int
    gru_input_dim: int
    gru_hidden_dim: int
    gru_layers: int


@dataclass(frozen=True)
class ActionConfig:
    mode_names: tuple[str, ...]
    mode_ids: tuple[int, ...]
    force_ratio_mg: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class EpisodeConfig:
    duration_s: float
    formal_steps: int
    coverage_points: int
    reset_hold_steps: int
    true_terminal: bool


@dataclass(frozen=True)
class RecoveryConfig:
    window_s: float
    position_threshold_m: float
    rotation_threshold_deg: float
    coverage_threshold: float
    escape_position_m: float
    escape_rotation_deg: float
    coverage_wait_s: float
    reward_limit_s: float


@dataclass(frozen=True)
class PPOConfig:
    rollout_steps: int
    num_learning_epochs: int
    num_mini_batches: int
    clip_param: float
    use_clipped_value_loss: bool
    value_loss_coef: float
    learning_rate: float
    schedule: str
    desired_kl: float
    max_grad_norm: float
    gamma: float
    lam: float
    entropy_coef: float


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    num_envs: int
    max_updates: int
    transitions: int
    synchronous_resets: bool


@dataclass(frozen=True)
class ValidationConfig:
    updates: tuple[int, ...]
    pose_ids: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointConfig:
    rolling_interval: int
    permanent_updates: tuple[int, ...]


@dataclass(frozen=True)
class Task010Config:
    schema: str
    version: int
    config_sha256: str
    task_id: str
    base_commit: str
    clocks: ClocksConfig
    camera: CameraConfig
    model: ModelConfig
    action: ActionConfig
    episode: EpisodeConfig
    coverage: Mapping[str, Any]
    recovery: RecoveryConfig
    reward: Mapping[str, float]
    ppo: PPOConfig
    training: TrainingConfig
    validation: ValidationConfig
    checkpoints: CheckpointConfig
    image_augmentation: bool
    environment_randomization: bool
    artifact_subdirectory: str


_TOP_LEVEL = {field.name for field in fields(Task010Config)}


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    return value


def canonical_config_sha256(config: Task010Config | Mapping[str, Any]) -> str:
    payload = _canonical(config)
    payload.pop("config_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact(record: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = sorted(set(record) - expected)
    missing = sorted(expected - set(record))
    if unknown:
        if "augmentation" in unknown:
            raise ValueError("augmentation must remain disabled")
        raise ValueError(f"{name} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing fields: {missing}")


def _section(cls, value: Any, name: str):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    _exact(value, {field.name for field in fields(cls)}, name)
    return cls(**value)


def _finite_tree(value: Any, name: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def load_task010_config(path: Path = TASK010_CONFIG_PATH) -> Task010Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("TASK-010 config must be an object")
    _exact(raw, _TOP_LEVEL, "TASK-010 config")
    if raw["schema"] != TASK010_SCHEMA or int(raw["version"]) != TASK010_VERSION:
        raise ValueError("TASK-010 schema/version mismatch")
    if raw["task_id"] != TASK010_TASK_ID or raw["base_commit"] != TASK010_BASE_COMMIT:
        raise ValueError("TASK-010 identity mismatch")
    if raw["image_augmentation"] is not False or raw["environment_randomization"] is not False:
        raise ValueError("augmentation must remain disabled")
    action = dict(raw["action"])
    _exact(action, {"mode_names", "mode_ids", "force_ratio_mg"}, "action")
    action_cfg = ActionConfig(
        tuple(action["mode_names"]),
        tuple(int(item) for item in action["mode_ids"]),
        MappingProxyType({key: tuple(float(item) for item in value) for key, value in action["force_ratio_mg"].items()}),
    )
    validation = dict(raw["validation"])
    checkpoints = dict(raw["checkpoints"])
    cfg = Task010Config(
        schema=raw["schema"], version=int(raw["version"]), config_sha256=raw["config_sha256"],
        task_id=raw["task_id"], base_commit=raw["base_commit"],
        clocks=_section(ClocksConfig, raw["clocks"], "clocks"),
        camera=_section(CameraConfig, raw["camera"], "camera"),
        model=_section(ModelConfig, raw["model"], "model"), action=action_cfg,
        episode=_section(EpisodeConfig, raw["episode"], "episode"),
        coverage=MappingProxyType(dict(raw["coverage"])),
        recovery=_section(RecoveryConfig, raw["recovery"], "recovery"),
        reward=MappingProxyType({key: float(value) for key, value in raw["reward"].items()}),
        ppo=_section(PPOConfig, raw["ppo"], "ppo"),
        training=_section(TrainingConfig, raw["training"], "training"),
        validation=ValidationConfig(tuple(validation["updates"]), tuple(validation["pose_ids"])),
        checkpoints=CheckpointConfig(int(checkpoints["rolling_interval"]), tuple(checkpoints["permanent_updates"])),
        image_augmentation=raw["image_augmentation"], environment_randomization=raw["environment_randomization"],
        artifact_subdirectory=raw["artifact_subdirectory"],
    )
    _finite_tree(_canonical(cfg))
    expected = {
        "clocks": (cfg.clocks.physics_hz, cfg.clocks.control_hz, cfg.clocks.physics_steps_per_action),
        "model": (cfg.model.resnet_weights, cfg.model.actor_observation_dim, cfg.model.critic_observation_dim),
        "training": (cfg.training.seed, cfg.training.num_envs, cfg.training.max_updates),
        "ppo": (cfg.ppo.rollout_steps, cfg.ppo.gamma, cfg.ppo.lam),
    }
    if expected != {"clocks": (240, 10, 24), "model": ("IMAGENET1K_V1", 519, 65), "training": (991000, 12, 1000), "ppo": (64, 0.999, 0.95)}:
        raise ValueError("TASK-010 frozen values differ from the approved contract")
    if cfg.action.mode_ids != (0, 1, 2, 3, 4, 5) or cfg.episode.formal_steps != 1200:
        raise ValueError("TASK-010 action or episode contract mismatch")
    if cfg.config_sha256 != canonical_config_sha256(cfg):
        raise ValueError("TASK-010 deterministic config hash mismatch")
    return cfg
