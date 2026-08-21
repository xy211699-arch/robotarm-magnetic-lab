"""TASK-008 one-press/one-macro keyboard contract."""

from __future__ import annotations

from .atomic_keyboard import CommandKind, KeyCommand, SPECIAL_KEYS, normalize_key


ACTION_KEYS = {
    "SPACE": 0,
    "D": 1,
    "A": 2,
    "E": 3,
    "Q": 4,
    "W": 5,
}


class DynamicForceMacroKeyboard:
    """Emit only rising-edge commands and suppress OS key repeat."""

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
