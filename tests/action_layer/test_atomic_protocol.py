from dataclasses import fields

import action_layer


def test_action_ids_are_frozen():
    assert {item.name: int(item) for item in action_layer.AtomicAction} == {
        "HOLD": 0,
        "TILT_POS": 1,
        "TILT_NEG": 2,
        "AZIMUTH_POS": 3,
        "AZIMUTH_NEG": 4,
        "ROLL_POS": 5,
        "ROLL_NEG": 6,
        "TURN_POS": 7,
        "TURN_NEG": 8,
        "APPROACH": 9,
        "RETREAT": 10,
    }


def test_result_contract_has_only_two_statuses_and_no_effect_grade():
    assert {item.value for item in action_layer.ActionStatus} == {"DONE", "HARD_FAILURE"}
    names = {item.name for item in fields(action_layer.ActionResult)}
    forbidden = {"effect_grade", "success", "acceptable", "capsule_pose", "capsule_velocity"}
    assert names.isdisjoint(forbidden)


def test_deployable_snapshot_excludes_capsule_truth():
    names = {item.name.lower() for item in fields(action_layer.DeviceSnapshot)}
    forbidden_tokens = ("capsule", "contact", "depth", "stomach", "magnetic_force")
    assert not any(token in name for name in names for token in forbidden_tokens)
