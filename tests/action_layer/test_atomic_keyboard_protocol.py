"""Pure fixed-key and non-preemptive session protocol tests."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "source" / "robotarm_magnetic_lab"
sys.path.insert(0, str(PACKAGE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "robotarm_magnetic_lab" or module_name.startswith("robotarm_magnetic_lab."):
        del sys.modules[module_name]

from robotarm_magnetic_lab.teleop.atomic_keyboard import AtomicKeyboard, CommandKind
from robotarm_magnetic_lab.teleop.session_controller import RequestOutcome, SessionController


def test_all_frozen_key_mappings_and_key_down_edges():
    keyboard = AtomicKeyboard()
    expected = {
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
    for key, action_id in expected.items():
        command = keyboard.key_event(key, is_down=True)
        assert command.kind is CommandKind.ACTION and command.action_id == action_id
        assert keyboard.key_event(key, is_down=True) is None  # repeat suppression
        assert keyboard.key_event(key, is_down=False) is None
    assert keyboard.key_event("BACKSPACE", True).kind is CommandKind.RESET
    keyboard.key_event("BACKSPACE", False)
    assert keyboard.key_event("F12", True).kind is CommandKind.SNAPSHOT
    keyboard.key_event("F12", False)
    assert keyboard.key_event("ESC", True).kind is CommandKind.EXIT


def test_busy_mask_reset_termination_and_no_queue():
    session = SessionController()
    mask = [True] * 11
    accepted = session.request_action(1, mask, timestamp_s=0.0)
    assert accepted.outcome is RequestOutcome.ACCEPTED
    assert session.request_action(2, mask, 0.1).outcome is RequestOutcome.IGNORED_WHILE_BUSY
    assert session.request_reset(0.2).outcome is RequestOutcome.RESET_WHILE_BUSY
    completed = session.acknowledge("DONE", 1.0)
    assert completed is not None and completed.request_id == accepted.request_id
    assert session.acknowledge("DONE", 1.1) is None  # exactly one acknowledgement
    # Busy request was discarded rather than queued; the next accepted request is explicit.
    masked = mask.copy()
    masked[2] = False
    assert session.request_action(2, masked, 1.2).outcome is RequestOutcome.MASKED_ACTION
    next_request = session.request_action(3, mask, 1.3)
    assert next_request.outcome is RequestOutcome.ACCEPTED
    session.acknowledge("HARD_FAILURE", 2.0)
    assert session.request_action(0, mask, 2.1).outcome is RequestOutcome.EPISODE_TERMINATED
    assert session.request_reset(2.2).outcome is RequestOutcome.RESET_ACCEPTED
    assert session.request_action(0, mask, 2.3).outcome is RequestOutcome.ACCEPTED
