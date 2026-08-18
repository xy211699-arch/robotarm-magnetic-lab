from pathlib import Path

import numpy as np
import pytest

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.eleven_action import (
    ElevenActionRequestDecoder,
    RequestGate,
    contact_records_for_capsule,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.eleven_action import ElevenActionId


def test_public_decoder_accepts_only_minus_one_or_exact_ids_zero_through_ten():
    decoder = ElevenActionRequestDecoder()
    assert decoder.decode([-1.0]) is None
    for value in range(11):
        assert decoder.decode([float(value)]) is ElevenActionId(value)
    for invalid in (-2.0, 11.0, 1.25, np.nan, np.inf):
        with pytest.raises(ValueError):
            decoder.decode([invalid])


def test_request_gate_drops_executing_requests_without_queue_and_repeats_same_id_at_ready():
    gate = RequestGate()
    action = ElevenActionId.VIEW_UP
    assert gate.offer(action, ready=True) is action
    assert gate.take() is action
    assert gate.offer(action, ready=False) is None
    assert gate.take() is None
    assert gate.discarded_request_count == 1
    assert gate.offer(action, ready=True) is action
    assert gate.take() is action


def test_contact_parser_keeps_only_capsule_headers_and_all_contact_points():
    class Header:
        def __init__(self, a, b, offset, count):
            self.collider0 = a
            self.collider1 = b
            self.contact_data_offset = offset
            self.num_contact_data = count

    class Contact:
        def __init__(self, position, normal, impulse):
            self.position = position
            self.normal = normal
            self.impulse = impulse

    paths = {1: "/World/envs/env_0/Scene/MagneticDemo/target_magnet/Collider", 2: "/World/Floor", 3: "/World/Other"}
    headers = [Header(1, 2, 0, 2), Header(2, 3, 2, 1)]
    contacts = [
        Contact([1, 2, 3], [0, 0, 1], [0, 0, 0.1]),
        Contact([4, 5, 6], [0, 1, 0], [0, 0.2, 0]),
        Contact([9, 9, 9], [1, 0, 0], [1, 0, 0]),
    ]
    records = contact_records_for_capsule(
        headers,
        contacts,
        capsule_prim_path="/World/envs/env_0/Scene/MagneticDemo/target_magnet",
        path_resolver=paths.__getitem__,
    )
    assert len(records) == 2
    np.testing.assert_allclose(records[0].position_world, [1, 2, 3])
    assert records[0].impulse_n_s == pytest.approx(0.1)


def test_action_term_source_has_scalar_action_and_copied_global_com_wrench():
    source = Path(
        "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/"
        "robotarm_magnetic_lab/mdp/eleven_action.py"
    ).read_text(encoding="utf-8")
    assert "def action_dim" in source and "return 1" in source
    assert "positions=None" in source
    assert "is_global=True" in source
    assert ".copy()" in source

