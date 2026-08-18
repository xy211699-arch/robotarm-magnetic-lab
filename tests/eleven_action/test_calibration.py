from scripts.eleven_action.calibrate_eleven_action import (
    VIEW_CANONICAL_STATES,
    authorized_view_grid,
    choose_smallest_shared_move_k,
    choose_view_candidate,
)


def test_view_canonical_states_cover_tilt_azimuth_and_material_roll():
    assert len(VIEW_CANONICAL_STATES) == 8
    assert {state[0] for state in VIEW_CANONICAL_STATES} >= {0.0, 45.0, 75.0, 90.0}
    assert len({state[1] for state in VIEW_CANONICAL_STATES}) == 8
    assert len({state[2] for state in VIEW_CANONICAL_STATES}) == 8


def test_authorized_view_grid_is_complete_deterministic_and_contains_profile_default():
    grid = authorized_view_grid()
    assert len(grid) == 81
    assert grid[0] == (0.005, 0.0008, 5.0, 0.2)
    assert (0.02, 0.0016, 10.0, 0.4) in grid


def test_view_candidate_selection_uses_max_error_drift_then_wrench_lexicographic_order():
    candidates = [
        {"gains": (0.005, 0.0008, 5.0, 0.2), "passed": True, "max_angle_error_deg": 1.0, "max_support_drift_m": 0.001, "wrench_integral": 3.0},
        {"gains": (0.01, 0.0016, 10.0, 0.4), "passed": True, "max_angle_error_deg": 0.5, "max_support_drift_m": 0.0015, "wrench_integral": 5.0},
    ]
    assert choose_view_candidate(candidates)["gains"] == candidates[1]["gains"]
    assert choose_view_candidate([{**candidates[0], "passed": False}]) is None


def test_move_k_selection_is_smallest_shared_grid_value():
    trials = {
        0.9: {"positive_rate": 1.0, "negative_rate": 0.8},
        1.0: {"positive_rate": 0.9, "negative_rate": 0.9},
        1.1: {"positive_rate": 1.0, "negative_rate": 1.0},
    }
    assert choose_smallest_shared_move_k(trials) == 1.0
    assert choose_smallest_shared_move_k({3.0: {"positive_rate": 0.8, "negative_rate": 1.0}}) is None
