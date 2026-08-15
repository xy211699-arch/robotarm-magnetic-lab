from __future__ import annotations

from robotarm_magnetic_lab.teleop import CommandKind, IdealSurfaceKeyboard


def test_fifteen_action_key_map_and_repeat_suppression():
    expected = {
        "SPACE": 0,
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
    keyboard = IdealSurfaceKeyboard()
    for key, action_id in expected.items():
        command = keyboard.key_event(key, True)
        assert command.action_id == action_id
        assert keyboard.key_event(key, True) is None
        assert keyboard.key_event(key, False) is None


def test_numpad_aliases_are_identical():
    keyboard = IdealSurfaceKeyboard()
    for alias in ("NUMPAD_8", "NUMPAD8", "KP8", "8"):
        command = keyboard.key_event(alias, True)
        assert command.action_id == 1
        keyboard.key_event(alias, False)


def test_reset_snapshot_and_exit_are_preserved():
    keyboard = IdealSurfaceKeyboard()
    assert keyboard.key_event("BACKSPACE", True).kind is CommandKind.RESET
    assert keyboard.key_event("F12", True).kind is CommandKind.SNAPSHOT
    assert keyboard.key_event("ESCAPE", True).kind is CommandKind.EXIT
