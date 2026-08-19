from scripts.virtual_magnet.common import summarize_trials, terminal_pass


def _audit(**updates):
    value = {
        "physics_substeps": 240,
        "result": "completed",
        "linear_speed_m_s": 0.001,
        "angular_speed_rad_s": 0.05,
        "optical_axis_error_deg": 2.0,
        "tangent_drift_m": 0.001,
        "move_signed_displacement_m": 0.005,
    }
    value.update(updates)
    return value


def test_exact_action_metrics_and_invalid_move():
    assert terminal_pass(0, _audit())
    assert terminal_pass(1, _audit())
    assert terminal_pass(9, _audit())
    assert terminal_pass(10, _audit(result="rejected"), invalid_move=True)
    assert not terminal_pass(9, _audit(move_signed_displacement_m=0.0039))
    assert not terminal_pass(1, _audit(angular_speed_rad_s=0.1001))


def test_summary_requires_sixteen_of_twenty_per_class():
    rows = [{"action_id": 0, "pass": index < 16, "result": "completed"} for index in range(20)]
    assert summarize_trials(rows)["all_classes_pass"]
    rows[-5]["pass"] = False
    assert not summarize_trials(rows)["all_classes_pass"]

