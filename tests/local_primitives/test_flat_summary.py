import copy
import math

from scripts.local_primitives.validate_local_primitives_flat import evaluate_flat_summary


def valid_flat_summary():
    primitive = {
        "status": "succeeded_holding", "completion_time_s": 5.0,
        "nonfinite_samples": 0, "max_force_n": 0.01, "force_bound_n": 0.02,
        "max_torque_nm": 1e-5, "torque_bound_nm": 2e-5,
        "max_physics_step_displacement_m": 1e-5,
        "max_force_slew_n_per_s": 1.0, "force_slew_bound_n_per_s": 20.0,
        "max_torque_slew_nm_per_s": 0.001, "torque_slew_bound_nm_per_s": 0.05,
        "camera_hemisphere_load_samples": 0, "late_dominant_non_camera": True,
        "actual_cone_coverage_rad": 2 * math.pi, "cone_tilt_rmse_rad": 0.01,
    }
    return {
        "preflight": {"gate": {"status": "pass"}},
        "runtime_contract": {"forbidden_calls": []},
        "contact": {"max_force_n": 0.0},
        "primitives": {name: copy.deepcopy(primitive) for name in (
            "side_to_upright", "upright_to_side", "upright_to_30_deg", "cone_30_deg",
        )},
    }


def test_flat_summary_requires_four_sub_ten_second_successes():
    summary = valid_flat_summary()
    assert evaluate_flat_summary(summary)["status"] == "pass"
    summary["primitives"]["cone_30_deg"]["completion_time_s"] = 10.0
    assert evaluate_flat_summary(summary)["status"] == "fail"


def test_flat_summary_reports_missing_completion_without_crashing():
    summary = valid_flat_summary()
    summary["primitives"]["side_to_upright"]["status"] = "timed_out"
    summary["primitives"]["side_to_upright"]["completion_time_s"] = None
    result = evaluate_flat_summary(summary)
    assert result["status"] == "fail"
    assert any("completion time" in failure for failure in result["failures"])


def test_rise_rejects_camera_hemisphere_support():
    summary = valid_flat_summary()
    summary["primitives"]["side_to_upright"]["camera_hemisphere_load_samples"] = 1
    assert evaluate_flat_summary(summary)["status"] == "fail"


def test_contact_does_not_fail_without_tracking_failure():
    summary = valid_flat_summary()
    summary["contact"]["max_force_n"] = 100.0
    assert evaluate_flat_summary(summary)["status"] == "pass"


def test_unrealistic_but_allowed_wrench_does_not_fail():
    summary = valid_flat_summary()
    summary["wrench"] = {"max_force_n": 4.9, "max_torque_nm": 0.019}
    assert evaluate_flat_summary(summary)["status"] == "pass"


def test_numerical_discontinuity_still_fails():
    summary = valid_flat_summary()
    summary["continuity"] = {"max_physics_step_displacement_m": 0.0051}
    assert evaluate_flat_summary(summary)["status"] == "fail"
