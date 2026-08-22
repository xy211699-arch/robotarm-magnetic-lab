from robotarm_magnetic_lab.teleop import CommandKind, DynamicForceMacroKeyboard


def test_exact_action_mapping_and_repeat_suppression():
    keyboard = DynamicForceMacroKeyboard()
    expected = {
        "SPACE": 0,
        "D": 1,
        "A": 2,
        "E": 3,
        "Q": 4,
        "W": 5,
        "L": 6,
        "J": 7,
        "O": 8,
        "U": 9,
        "K": 10,
        "H": 11,
        "I": 12,
        "Y": 13,
    }
    for key, action_id in expected.items():
        command = keyboard.key_event(key, True)
        assert command.kind is CommandKind.ACTION
        assert command.action_id == action_id
        assert keyboard.key_event(key, True) is None
        assert keyboard.key_event(key, False) is None


def test_reset_snapshot_exit_mapping():
    keyboard = DynamicForceMacroKeyboard()
    assert keyboard.key_event("BACKSPACE", True).kind is CommandKind.RESET
    assert keyboard.key_event("F12", True).kind is CommandKind.SNAPSHOT
    assert keyboard.key_event("ESCAPE", True).kind is CommandKind.EXIT
