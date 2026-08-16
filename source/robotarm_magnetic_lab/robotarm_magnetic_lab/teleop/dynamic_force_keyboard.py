"""Level-triggered six-direction keyboard state for TASK-003."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force import (
    normalize_force_direction,
)
from .atomic_keyboard import normalize_key


class DynamicForceCommandKind(str, Enum):
    CLEAR = "CLEAR"
    RESET = "RESET"
    SNAPSHOT = "SNAPSHOT"
    EXIT = "EXIT"


@dataclass(frozen=True)
class DynamicForceCommand:
    kind: DynamicForceCommandKind
    key: str


FORCE_KEYS = {
    "W": np.asarray([1.0, 0.0, 0.0]),
    "S": np.asarray([-1.0, 0.0, 0.0]),
    "A": np.asarray([0.0, 1.0, 0.0]),
    "D": np.asarray([0.0, -1.0, 0.0]),
    "Q": np.asarray([0.0, 0.0, 1.0]),
    "E": np.asarray([0.0, 0.0, -1.0]),
}
SPECIAL_KEYS = {
    "BACKSPACE": DynamicForceCommandKind.RESET,
    "F12": DynamicForceCommandKind.SNAPSHOT,
    "ESC": DynamicForceCommandKind.EXIT,
    "ESCAPE": DynamicForceCommandKind.EXIT,
}


class DynamicForceKeyboard:
    """Maintain held directions; emit commands only for non-force controls."""

    def __init__(self) -> None:
        self._held_force_keys: set[str] = set()
        self._held_special_keys: set[str] = set()

    @property
    def direction(self) -> np.ndarray:
        if not self._held_force_keys:
            return np.zeros(3, dtype=np.float64)
        summed = np.sum([FORCE_KEYS[key] for key in sorted(self._held_force_keys)], axis=0)
        return normalize_force_direction(summed)

    def key_event(self, key: str, is_down: bool) -> DynamicForceCommand | None:
        normalized = normalize_key(key)
        if normalized in FORCE_KEYS:
            if is_down:
                self._held_force_keys.add(normalized)
            else:
                self._held_force_keys.discard(normalized)
            return None
        if normalized == "SPACE":
            if not is_down:
                self._held_special_keys.discard(normalized)
                return None
            if normalized in self._held_special_keys:
                return None
            self._held_special_keys.add(normalized)
            self._held_force_keys.clear()
            return DynamicForceCommand(DynamicForceCommandKind.CLEAR, normalized)
        if normalized in SPECIAL_KEYS:
            if not is_down:
                self._held_special_keys.discard(normalized)
                return None
            if normalized in self._held_special_keys:
                return None
            self._held_special_keys.add(normalized)
            return DynamicForceCommand(SPECIAL_KEYS[normalized], normalized)
        return None

    def release_all(self) -> None:
        self._held_force_keys.clear()
        self._held_special_keys.clear()
