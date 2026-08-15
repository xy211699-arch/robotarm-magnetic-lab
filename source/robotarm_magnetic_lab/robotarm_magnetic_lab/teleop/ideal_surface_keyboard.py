"""Fifteen-action keyboard with canonical numpad aliases and edge suppression."""

from __future__ import annotations

from .atomic_keyboard import CommandKind, KeyCommand, SPECIAL_KEYS, normalize_key


ACTION_KEYS = {
    "SPACE": 0,
    # Letter compass layout for keyboards whose numpad events are unavailable:
    # R T Y
    # F   H
    # V B N
    "T": 1,
    "Y": 2,
    "H": 3,
    "N": 4,
    "B": 5,
    "V": 6,
    "F": 7,
    "R": 8,
    "NUMPAD8": 1,
    "NUMPAD9": 2,
    "NUMPAD6": 3,
    "NUMPAD3": 4,
    "NUMPAD2": 5,
    "NUMPAD1": 6,
    "NUMPAD4": 7,
    "NUMPAD7": 8,
    "W": 9,
    "S": 10,
    "D": 11,
    "A": 12,
    "E": 13,
    "Q": 14,
}


def normalize_ideal_surface_key(key: str) -> str:
    value = normalize_key(key).replace("NUMPAD_", "NUMPAD")
    if value.startswith("KP") and value[2:].isdigit():
        value = "NUMPAD" + value[2:]
    elif len(value) == 1 and value.isdigit():
        value = "NUMPAD" + value
    return value


class IdealSurfaceKeyboard:
    def __init__(self) -> None:
        self._down: set[str] = set()

    def key_event(self, key: str, is_down: bool) -> KeyCommand | None:
        normalized = normalize_ideal_surface_key(key)
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
