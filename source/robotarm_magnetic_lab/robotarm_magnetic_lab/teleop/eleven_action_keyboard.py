"""TASK-005 numeric-grid keyboard with physical key edge suppression."""

from __future__ import annotations

from .atomic_keyboard import CommandKind, KeyCommand, SPECIAL_KEYS, normalize_key


ACTION_KEYS = {
    "NUMPAD8": 1,
    "NUMPAD9": 2,
    "NUMPAD6": 3,
    "NUMPAD3": 4,
    "NUMPAD2": 5,
    "NUMPAD1": 6,
    "NUMPAD4": 7,
    "NUMPAD7": 8,
    "NUMPAD5": 0,
    "E": 9,
    "Q": 10,
}


def normalize_eleven_action_key(key: str) -> str:
    value = normalize_key(key).replace("NUMPAD_", "NUMPAD")
    if value.startswith("KP") and value[2:].isdigit():
        value = "NUMPAD" + value[2:]
    elif len(value) == 1 and value.isdigit():
        value = "NUMPAD" + value
    return value


class ElevenActionKeyboard:
    def __init__(self) -> None:
        self._down: set[str] = set()

    def key_event(self, key: str, is_down: bool) -> KeyCommand | None:
        normalized = normalize_eleven_action_key(key)
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

