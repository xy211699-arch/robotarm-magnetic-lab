from pathlib import Path

from robotarm_magnetic_lab.teleop import (
    CommandKind,
    DynamicForceMacroKeyboard,
    ParameterizedForceKeyboard,
    ParameterizedKeyboardEventKind,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.parameterized_force import ParameterizedForceMode


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


def test_parameterized_keyboard_is_held_and_release_returns_hold():
    keyboard = ParameterizedForceKeyboard(alpha=0.5)
    assert keyboard.command == (ParameterizedForceMode.HOLD, 0.5)
    assert keyboard.key_event("D", True).kind is ParameterizedKeyboardEventKind.CONTROL_CHANGED
    assert keyboard.command == (ParameterizedForceMode.MOVE_POS, 0.5)
    assert keyboard.key_event("D", True) is None
    keyboard.key_event("D", False)
    assert keyboard.command == (ParameterizedForceMode.HOLD, 0.5)


def test_parameterized_keyboard_strength_and_last_pressed_mode():
    keyboard = ParameterizedForceKeyboard()
    keyboard.key_event("A", True)
    keyboard.key_event("E", True)
    keyboard.key_event("C", True)
    assert keyboard.command == (ParameterizedForceMode.VIEW_POS, 1.0)
    keyboard.key_event("E", False)
    assert keyboard.command == (ParameterizedForceMode.MOVE_NEG, 1.0)
    keyboard.key_event("SPACE", True)
    assert keyboard.command == (ParameterizedForceMode.HOLD, 1.0)
    keyboard.key_event("SPACE", False)
    assert keyboard.command == (ParameterizedForceMode.MOVE_NEG, 1.0)


def test_parameterized_keyboard_uses_letter_reset_snapshot_exit():
    keyboard = ParameterizedForceKeyboard()
    assert keyboard.key_event("R", True).kind is ParameterizedKeyboardEventKind.RESET
    assert keyboard.key_event("P", True).kind is ParameterizedKeyboardEventKind.SNAPSHOT
    assert keyboard.key_event("ESCAPE", True).kind is ParameterizedKeyboardEventKind.EXIT


def test_10hz_visual_launcher_does_not_use_macro_runner_or_popup():
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/parameterized_force/teleop_table_10hz.py"
    ).read_text(encoding="utf-8")
    assert "ParameterizedForceKeyboard" in source
    assert "parameterized_force" in source
    assert "configure_capsule_camera_view(cfg)" in source
    assert "configure_capsule_pose_view(cfg)" in source
    assert source.index("configure_capsule_pose_view(cfg)") < source.index("gym.make(args_cli.task, cfg=cfg)")
    assert "attach_capsule_pose_view(env)" in source
    assert "SynchronousMacroRunner" not in source
    assert "omni.ui.Window" not in source
    assert "physics_steps_per_control" in source
    assert '"--render_fps"' in source
    assert "cfg.sim.render_interval = PHYSICS_HZ // args_cli.render_fps" in source
