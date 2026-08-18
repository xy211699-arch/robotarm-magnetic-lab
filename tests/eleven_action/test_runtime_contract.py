from pathlib import Path


RUNTIME_FILES = [
    Path("source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/eleven_action.py"),
    Path("source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/eleven_action/controller.py"),
    Path("scripts/eleven_action/inspect_eleven_action_prerequisites.py"),
]


def test_runtime_files_do_not_write_state_or_control_robot_asm_or_external_magnet():
    forbidden = (
        "write_root_pose_to_sim",
        "write_root_velocity_to_sim",
        "set_world_pose",
        "set_linear_velocity",
        "set_angular_velocity",
        "kinematicEnabledAttr().Set",
        "robot.set_",
        "asm.set_",
        "external_magnet.set_",
    )
    for path in RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"forbidden runtime mutation {token!r} in {path}"


def test_runtime_reads_com_and_link_state_and_verifies_dynamic_invariants():
    source = RUNTIME_FILES[0].read_text(encoding="utf-8")
    for required in (
        "root_com_pose_w",
        "root_link_pose_w",
        "root_com_vel_w",
        "1.0 / 240.0",
        "non-kinematic",
        "gravity",
        "CCD",
        "num_envs",
    ):
        assert required in source


def test_runtime_does_not_add_policy_observation_or_action_mask():
    source = RUNTIME_FILES[0].read_text(encoding="utf-8")
    assert "ObservationTerm" not in source
    assert "action_mask" not in source

