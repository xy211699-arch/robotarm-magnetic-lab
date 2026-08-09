"""Fixed key mapping with rising-edge and OS-repeat suppression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandKind(str, Enum):
    ACTION = "ACTION"
    RESET = "RESET"
    SNAPSHOT = "SNAPSHOT"
    EXIT = "EXIT"


@dataclass(frozen=True)
class KeyCommand:
    kind: CommandKind
    key: str
    action_id: int | None = None


ACTION_KEYS = {
    "W": 1,
    "S": 2,
    "D": 3,
    "A": 4,
    "E": 5,
    "Q": 6,
    "C": 7,
    "Z": 8,
    "R": 9,
    "F": 10,
    "SPACE": 0,
}
SPECIAL_KEYS = {
    "BACKSPACE": CommandKind.RESET,
    "F12": CommandKind.SNAPSHOT,
    "ESC": CommandKind.EXIT,
    "ESCAPE": CommandKind.EXIT,
}


def normalize_key(key: str) -> str:
    value = str(key).strip().upper()
    return "SPACE" if value in ("", " ", "SPACEBAR") else value


class AtomicKeyboard:
    """Convert raw down/up events into at most one command per key press."""

    def __init__(self) -> None:
        self._down: set[str] = set()

    def key_event(self, key: str, is_down: bool) -> KeyCommand | None:
        normalized = normalize_key(key)
        if not is_down:
            self._down.discard(normalized)
            return None
        if normalized in self._down:
            return None
        self._down.add(normalized)
        if normalized in ACTION_KEYS:
            return KeyCommand(CommandKind.ACTION, normalized, ACTION_KEYS[normalized])
        if normalized in SPECIAL_KEYS:
            return KeyCommand(SPECIAL_KEYS[normalized], normalized)
        return None

    def release_all(self) -> None:
        self._down.clear()
