"""Held-key input state for the 10 Hz parameterized-force controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import (
    ParameterizedForceMode,
)

from .atomic_keyboard import normalize_key


ACTION_KEYS = {
    "D": ParameterizedForceMode.MOVE_POS,
    "A": ParameterizedForceMode.MOVE_NEG,
    "E": ParameterizedForceMode.VIEW_POS,
    "Q": ParameterizedForceMode.VIEW_NEG,
    "W": ParameterizedForceMode.UP,
    "SPACE": ParameterizedForceMode.HOLD,
}
ALPHA_KEYS = {"Z": 0.0, "X": 0.5, "C": 1.0}


class ParameterizedKeyboardEventKind(str, Enum):
    CONTROL_CHANGED = "CONTROL_CHANGED"
    RESET = "RESET"
    SNAPSHOT = "SNAPSHOT"
    EXIT = "EXIT"


@dataclass(frozen=True)
class ParameterizedKeyboardEvent:
    kind: ParameterizedKeyboardEventKind
    key: str
    mode: ParameterizedForceMode
    alpha: float


class ParameterizedForceKeyboard:
    """Resolve held English keys into one mode plus one global alpha value."""

    def __init__(self, alpha: float = 0.5) -> None:
        if float(alpha) not in (0.0, 0.5, 1.0):
            raise ValueError("visualization alpha must be one of 0, 0.5, 1")
        self.alpha = float(alpha)
        self._down: set[str] = set()
        self._held_action_order: list[str] = []

    @property
    def mode(self) -> ParameterizedForceMode:
        if not self._held_action_order:
            return ParameterizedForceMode.HOLD
        return ACTION_KEYS[self._held_action_order[-1]]

    @property
    def command(self) -> tuple[ParameterizedForceMode, float]:
        return self.mode, self.alpha

    def key_event(self, key: str, is_down: bool) -> ParameterizedKeyboardEvent | None:
        normalized = normalize_key(key)
        if not is_down:
            self._down.discard(normalized)
            if normalized in self._held_action_order:
                self._held_action_order.remove(normalized)
                return self._event(ParameterizedKeyboardEventKind.CONTROL_CHANGED, normalized)
            return None
        if normalized in self._down:
            return None
        self._down.add(normalized)
        if normalized in ACTION_KEYS:
            if normalized in self._held_action_order:
                self._held_action_order.remove(normalized)
            self._held_action_order.append(normalized)
            return self._event(ParameterizedKeyboardEventKind.CONTROL_CHANGED, normalized)
        if normalized in ALPHA_KEYS:
            self.alpha = ALPHA_KEYS[normalized]
            return self._event(ParameterizedKeyboardEventKind.CONTROL_CHANGED, normalized)
        if normalized in ("R", "BACKSPACE"):
            return self._event(ParameterizedKeyboardEventKind.RESET, normalized)
        if normalized == "P":
            return self._event(ParameterizedKeyboardEventKind.SNAPSHOT, normalized)
        if normalized in ("ESC", "ESCAPE"):
            return self._event(ParameterizedKeyboardEventKind.EXIT, normalized)
        return None

    def release_all(self) -> None:
        self._down.clear()
        self._held_action_order.clear()

    def _event(self, kind: ParameterizedKeyboardEventKind, key: str) -> ParameterizedKeyboardEvent:
        return ParameterizedKeyboardEvent(kind, key, self.mode, self.alpha)
