from scripts.eleven_action.validate_eleven_action_flat import evaluate_samples


def _row(action_id, *, category="valid", result="completed", angle=15.0, drift=0.001, displacement=0.006):
    return {
        "action_id": action_id,
        "category": category,
        "result": result,
        "substeps": 240,
        "constrained": False,
        "angle_delta_deg": angle,
        "max_support_drift_m": drift,
        "move_signed_displacement_m": displacement,
        "fault": False,
    }


def _passing_samples():
    rows = []
    for action in range(1, 9):
        rows.extend(_row(action) for _ in range(10))
    rows.extend(_row(0, angle=2.0) for _ in range(10))
    for action in (9, 10):
        rows.extend(_row(action) for _ in range(9))
        rows.append(_row(action, displacement=0.004))
        rows.extend(_row(action, category="invalid_angle", result="rejected", displacement=0.0) for _ in range(5))
        rows.extend(_row(action, category="invalid_contact", result="rejected", displacement=0.0) for _ in range(5))
    return rows


def test_exact_flat_acceptance_gate_passes_required_counts_and_ninety_percent_move():
    summary = evaluate_samples(_passing_samples())
    assert summary["status"] == "pass"
    assert summary["fault_count"] == 0
    assert summary["actions"]["9"]["valid_success_rate"] == 0.9


def test_low_effect_move_remains_completed_but_fails_batch_and_fault_always_fails():
    samples = _passing_samples()
    samples[90]["move_signed_displacement_m"] = 0.001
    assert evaluate_samples(samples)["status"] == "fail"
    samples = _passing_samples()
    samples[0]["fault"] = True
    samples[0]["result"] = "fault"
    assert evaluate_samples(samples)["status"] == "fail"


def test_constrained_view_is_excluded_from_unblocked_angle_denominator():
    samples = _passing_samples()
    blocked = _row(1, angle=0.0)
    blocked["constrained"] = True
    samples.append(blocked)
    summary = evaluate_samples(samples)
    assert summary["status"] == "pass"
    assert summary["actions"]["1"]["blocked_count"] == 1

