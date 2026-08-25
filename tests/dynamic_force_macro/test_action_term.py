from pathlib import Path

import numpy as np
import pytest

from robotarm_magnetic_lab.runtime.quaternion_conventions import (
    ISAAC_QUATERNION_ORDER,
    rotation_matrix_from_xyzw,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.controllers.dynamic_force_macro import (
    DynamicForceMacroConfig,
    resolved_force_levels_n,
)
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.dynamic_force_macro_action import DynamicForceMacroActionTermCfg


def test_action_term_contract_and_no_state_writers():
    cfg = DynamicForceMacroActionTermCfg()
    assert (cfg.move_force_ratio, cfg.move_force_ratio_medium, cfg.move_force_ratio_high) == (0.40, 0.50, 0.60)
    assert (cfg.view_force_ratio, cfg.view_force_ratio_medium, cfg.view_force_ratio_high) == (0.25, 0.35, 0.45)
    assert cfg.up_force_ratio == 0.85
    assert cfg.camera_sensor_name == "capsule_camera"
    source = Path(__file__).resolve().parents[2] / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_macro_action.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("write_root_pose", "write_root_velocity", "set_transforms", "set_velocities"):
        assert forbidden not in text
    assert "equivalent_com_wrench" in text
    assert "camera_sphere_centers_local" in text
    assert "positions=position_tensor" in text
    assert "torques=torque_tensor" in text
    assert "torque_tensor = None" in text
    assert "application_position = np.asarray(points[0].position_world" in text
    assert "CreateEnableCCDAttr" in text
    assert "ccd_attr.Set(True)" in text
def test_parameterized_action_uses_xyzw_and_has_no_state_writer():
    source = Path(__file__).resolve().parents[2] / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/parameterized_force_action.py"
    text = source.read_text(encoding="utf-8")
    assert "rotation_matrix_from_xyzw(link_quat)" in text
    assert "link_quat[[1, 2, 3, 0]]" not in text
    for forbidden in ("write_root_pose", "write_root_velocity", "set_transforms", "set_velocities"):
        assert forbidden not in text
    assert "positions=position_tensor" in text
    assert "self.physics_step_in_cycle += 1" in text
    assert "self.current_cycle_trace.append(self.last_telemetry)" in text
    assert text.count("set_forces_and_torques_index(") == 1
    assert "torque_tensor = None" in text
    assert "application_position = camera" in text


def test_isaac_quaternion_boundary_is_xyzw_and_rotates_local_axes_correctly():
    assert ISAAC_QUATERNION_ORDER == "xyzw"
    half_angle = np.sqrt(0.5)
    rotation = rotation_matrix_from_xyzw([0.0, 0.0, half_angle, half_angle])
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1.0e-12)
    assert rotation @ np.array([0.0, 1.0, 0.0]) == pytest.approx([-1.0, 0.0, 0.0], abs=1.0e-12)


def test_explicit_legacy_wxyz_conversion_round_trip():
    quaternion_xyzw = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float64)
    quaternion_xyzw /= np.linalg.norm(quaternion_xyzw)
    quaternion_wxyz = xyzw_to_wxyz(quaternion_xyzw)
    assert quaternion_wxyz == pytest.approx(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]]
    )
    assert wxyz_to_xyzw(quaternion_wxyz) == pytest.approx(quaternion_xyzw)


@pytest.mark.parametrize("invalid", ([0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0, 1.0]))
def test_invalid_isaac_quaternion_is_rejected(invalid):
    with pytest.raises(ValueError):
        rotation_matrix_from_xyzw(invalid)


def test_resolved_force_levels_report_total_and_endpoint_newtons():
    levels = resolved_force_levels_n(
        0.005735,
        DynamicForceMacroConfig(move_force_ratio=0.2, view_force_ratio=0.5, up_force_ratio=1.05),
    )
    assert levels["weight_n"] == pytest.approx(0.005735 * 9.81)
    assert levels["move_total_force_n"] == pytest.approx(0.2 * levels["weight_n"])
    assert levels["move_force_per_endpoint_n"] == pytest.approx(0.5 * levels["move_total_force_n"])
    assert levels["move_medium_total_force_n"] == pytest.approx(0.5 * levels["weight_n"])
    assert levels["move_high_total_force_n"] == pytest.approx(0.6 * levels["weight_n"])
    assert levels["view_camera_endpoint_force_n"] == pytest.approx(0.5 * levels["weight_n"])
    assert levels["view_medium_camera_endpoint_force_n"] == pytest.approx(0.35 * levels["weight_n"])
    assert levels["view_high_camera_endpoint_force_n"] == pytest.approx(0.45 * levels["weight_n"])
    assert levels["up_camera_endpoint_force_n"] == pytest.approx(1.05 * levels["weight_n"])
