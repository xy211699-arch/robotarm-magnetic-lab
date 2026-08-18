import pytest

from scripts.local_primitives.calibrate_simulation_authority import (
    AuthorityCandidate, CALIBRATED_ENDPOINT_DAMPING_NS_PER_M, CalibrationRecord, EXPANDED_PIN_FORCES_N,
    EXPANDED_TORQUES_NM, PRIMARY_PIN_FORCES_N, PRIMARY_TORQUES_NM,
    candidate_grid, select_candidate,
)


def _record(torque, pin, status, completion):
    return CalibrationRecord(
        AuthorityCandidate(torque, pin), status, completion,
        "succeeded_holding" if status == "pass" else "timeout",
    )


def test_primary_and_expanded_grids_are_exact_and_deterministic():
    primary = candidate_grid(PRIMARY_TORQUES_NM, PRIMARY_PIN_FORCES_N)
    expanded = candidate_grid(EXPANDED_TORQUES_NM, EXPANDED_PIN_FORCES_N)
    assert len(primary) == 20
    assert primary[0] == AuthorityCandidate(1.0e-4, 0.05)
    assert primary[-1] == AuthorityCandidate(5.0e-3, 0.50)
    assert expanded == (
        AuthorityCandidate(0.01, 1.0), AuthorityCandidate(0.01, 2.0),
        AuthorityCandidate(0.02, 1.0), AuthorityCandidate(0.02, 2.0),
    )
    assert CALIBRATED_ENDPOINT_DAMPING_NS_PER_M == pytest.approx(20.0)


def test_selector_chooses_lowest_authority_pass():
    records = [
        _record(1e-4, 0.1, "fail", 8.0),
        _record(3e-4, 0.1, "pass", 5.0),
        _record(3e-4, 0.05, "pass", 6.0),
    ]
    selected = select_candidate(records)
    assert selected.pose_torque_limit_nm == pytest.approx(3e-4)
    assert selected.endpoint_pin_force_n == pytest.approx(0.05)


def test_selector_returns_none_without_a_pass():
    assert select_candidate([_record(0.02, 2.0, "fail", None)]) is None
