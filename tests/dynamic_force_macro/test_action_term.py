from pathlib import Path

from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.mdp.dynamic_force_macro_action import DynamicForceMacroActionTermCfg


def test_action_term_contract_and_no_state_writers():
    cfg = DynamicForceMacroActionTermCfg()
    assert cfg.move_force_ratio == cfg.view_force_ratio == cfg.up_force_ratio == 0.9
    source = Path(__file__).resolve().parents[2] / "source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/dynamic_force_macro_action.py"
    text = source.read_text(encoding="utf-8")
    for forbidden in ("write_root_pose", "write_root_velocity", "set_transforms", "set_velocities"):
        assert forbidden not in text
    assert "equivalent_com_wrench" in text
    assert "positions=None" in text
