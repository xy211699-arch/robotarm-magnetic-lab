import pytest

from robotarm_magnetic_lab.teleop import CommandKind, ElevenActionKeyboard


@pytest.mark.parametrize(
    ("key", "action_id"),
    [("8", 1), ("9", 2), ("6", 3), ("3", 4), ("2", 5), ("1", 6), ("4", 7), ("7", 8), ("5", 0), ("E", 9), ("Q", 10)],
)
def test_numpad_grid_and_move_keys(key, action_id):
    keyboard = ElevenActionKeyboard()
    command = keyboard.key_event(key, True)
    assert command.kind is CommandKind.ACTION
    assert command.action_id == action_id


def test_one_physical_press_emits_once_and_release_rearms():
    keyboard = ElevenActionKeyboard()
    assert keyboard.key_event("NUMPAD8", True) is not None
    assert keyboard.key_event("NUMPAD8", True) is None
    assert keyboard.key_event("NUMPAD8", False) is None
    assert keyboard.key_event("KP8", True) is not None


@pytest.mark.parametrize(
    ("key", "kind"),
    [("BACKSPACE", CommandKind.RESET), ("F12", CommandKind.SNAPSHOT), ("ESC", CommandKind.EXIT)],
)
def test_special_keys_have_no_overlay_dependency(key, kind):
    keyboard = ElevenActionKeyboard()
    assert keyboard.key_event(key, True).kind is kind

