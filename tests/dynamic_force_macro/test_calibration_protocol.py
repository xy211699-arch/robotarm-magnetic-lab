from scripts.dynamic_force_macro.common import (
    coarse_candidates,
    make_manifest,
    manifest_sha256,
    midpoint_refinements,
    replacement_trial,
)


def test_manifests_are_disjoint_and_deterministic():
    calibration = make_manifest("calibration", 20, 8008)
    held = make_manifest("held-out", 20, 18008)
    assert len(calibration) == len(held) == 100
    assert {row.seed for row in calibration}.isdisjoint({row.seed for row in held})
    assert manifest_sha256(calibration) == manifest_sha256(make_manifest("calibration", 20, 8008))


def test_candidate_schedule_and_refinements():
    values = coarse_candidates()
    assert values[0] == 0.9 and values[-1] == 3.0
    assert all(a < b for a, b in zip(values, values[1:]))
    mids = midpoint_refinements(0.9, 1.125, 3)
    assert len(mids) == 3
    assert all(0.9 < value < 1.125 for value in mids)


def test_invalid_setup_replacement_is_deterministic_and_keeps_slot():
    original = make_manifest("calibration", 1, 8008, actions=(1,))[0]
    replacement = replacement_trial(original, 1)
    assert replacement == replacement_trial(original, 1)
    assert replacement.seed != original.seed
    assert replacement.split == original.split
    assert replacement.action_id == original.action_id
    assert replacement.trial_index == original.trial_index
