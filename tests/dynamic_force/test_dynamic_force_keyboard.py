"""Held-key semantics for continuous six-direction force input."""

from __future__ import annotations

import numpy as np
import pytest

from robotarm_magnetic_lab.teleop.dynamic_force_keyboard import (
    DynamicForceCommandKind,
    DynamicForceKeyboard,
)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("W", [1, 0, 0]),
        ("S", [-1, 0, 0]),
        ("A", [0, 1, 0]),
        ("D", [0, -1, 0]),
        ("Q", [0, 0, 1]),
        ("E", [0, 0, -1]),
    ],
)
def test_each_key_selects_one_world_axis(key, expected):
    keyboard = DynamicForceKeyboard()
    assert keyboard.key_event(key, True) is None
    np.testing.assert_array_equal(keyboard.direction, expected)


def test_release_and_opposites_return_zero():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    keyboard.key_event("S", True)
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])
    keyboard.key_event("S", False)
    np.testing.assert_array_equal(keyboard.direction, [1, 0, 0])
    keyboard.key_event("W", False)
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])


def test_diagonal_is_norm_limited():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    keyboard.key_event("A", True)
    np.testing.assert_allclose(keyboard.direction, [2.0**-0.5, 2.0**-0.5, 0.0])


def test_repeated_press_is_level_triggered_and_release_clears():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    keyboard.key_event("W", True)
    np.testing.assert_array_equal(keyboard.direction, [1, 0, 0])
    keyboard.key_event("W", False)
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])


def test_space_clears_force_without_latching():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    command = keyboard.key_event("SPACE", True)
    assert command.kind is DynamicForceCommandKind.CLEAR
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("BACKSPACE", DynamicForceCommandKind.RESET),
        ("F12", DynamicForceCommandKind.SNAPSHOT),
        ("ESC", DynamicForceCommandKind.EXIT),
        ("ESCAPE", DynamicForceCommandKind.EXIT),
    ],
)
def test_special_commands(key, kind):
    keyboard = DynamicForceKeyboard()
    assert keyboard.key_event(key, True).kind is kind


def test_release_all_clears_force_and_special_key_state():
    keyboard = DynamicForceKeyboard()
    keyboard.key_event("W", True)
    keyboard.key_event("F12", True)
    keyboard.release_all()
    np.testing.assert_array_equal(keyboard.direction, [0, 0, 0])
    assert keyboard.key_event("F12", True).kind is DynamicForceCommandKind.SNAPSHOT
