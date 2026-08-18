from pathlib import Path

from robotarm_magnetic_lab.teleop.atomic_keyboard import CommandKind
from robotarm_magnetic_lab.teleop.local_primitive_keyboard import (
    LocalPrimitiveKeyboard,
    parse_local_primitive_sequence,
)


def test_keys_one_to_four_map_to_primitive_ids_and_suppress_repeat():
    keyboard = LocalPrimitiveKeyboard()
    for key, expected in (("1", 0), ("KEY_2", 1), ("NUMPAD3", 2), ("KP4", 3)):
        command = keyboard.key_event(key, True)
        assert command.kind is CommandKind.ACTION
        assert command.action_id == expected
        assert keyboard.key_event(key, True) is None
        assert keyboard.key_event(key, False) is None


def test_reset_snapshot_and_exit_keys_are_available():
    keyboard = LocalPrimitiveKeyboard()
    assert keyboard.key_event("BACKSPACE", True).kind is CommandKind.RESET
    assert keyboard.key_event("F12", True).kind is CommandKind.SNAPSHOT
    assert keyboard.key_event("ESC", True).kind is CommandKind.EXIT


def test_scripted_sequence_preserves_reset_boundaries():
    assert parse_local_primitive_sequence("0,1;reset;0,2;reset;0,2,3") == [
        0, 1, None, 0, 2, None, 0, 2, 3,
    ]


def test_launcher_terminal_output_is_event_driven():
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "local_primitives"
        / "teleop_local_primitives.py"
    ).read_text(encoding="utf-8")

    assert "LOCAL_PRIMITIVE_STATE" not in source
    assert "step % 30" not in source
    for event in (
        "LOCAL_PRIMITIVE_REQUEST",
        "LOCAL_PRIMITIVE_OUTCOME",
        "LOCAL_PRIMITIVE_RESET",
        "LOCAL_PRIMITIVE_SNAPSHOT",
        "LOCAL_PRIMITIVES_FINISHED",
    ):
        assert event in source
