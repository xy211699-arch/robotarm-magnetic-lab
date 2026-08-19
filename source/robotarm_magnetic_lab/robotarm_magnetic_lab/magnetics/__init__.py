"""Repository-local finite magnetic model used by TASK-007."""

from .config import config_sha256, load_config
from .field_models import FiniteMagnetSystem, magpylib_version

__all__ = ["FiniteMagnetSystem", "config_sha256", "load_config", "magpylib_version"]
