"""Edge-triggered keyboard protocol for the four TASK-004 primitives."""

from __future__ import annotations

from .atomic_keyboard import CommandKind, KeyCommand, SPECIAL_KEYS, normalize_key


ACTION_KEYS = {
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "KEY1": 0,
    "KEY2": 1,
    "KEY3": 2,
    "KEY4": 3,
}

RESET_EVENT = None


def parse_local_primitive_sequence(value: str) -> list[int | None]:
    """Parse ``0,1;reset;0,2`` into primitive IDs and reset boundaries."""

    events: list[int | None] = []
    if not value.strip():
        return events
    for group in value.split(";"):
        token = group.strip().lower()
        if not token:
            raise ValueError("scripted sequence contains an empty group")
        if token == "reset":
            events.append(RESET_EVENT)
            continue
        for item in token.split(","):
            try:
                primitive_id = int(item.strip())
            except ValueError as exc:
                raise ValueError(f"invalid primitive ID: {item!r}") from exc
            if primitive_id not in range(4):
                raise ValueError("primitive IDs must be 0, 1, 2, or 3")
            events.append(primitive_id)
    return events


def normalize_local_primitive_key(key: str) -> str:
    value = normalize_key(key).replace("_", "")
    if value.startswith("NUMPAD") and value[-1:] in "1234":
        return value[-1]
    if value.startswith("KP") and value[2:] in "1234":
        return value[2:]
    return value


class LocalPrimitiveKeyboard:
    """Emit one command per physical key press and suppress OS repeats."""

    def __init__(self) -> None:
        self._down: set[str] = set()

    def key_event(self, key: str, is_down: bool) -> KeyCommand | None:
        normalized = normalize_local_primitive_key(key)
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
