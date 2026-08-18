from pathlib import Path


SOURCE = Path("scripts/eleven_action/teleop_eleven_action.py").read_text(encoding="utf-8")


def test_launcher_supports_required_render_rates_and_no_request_sentinel():
    assert "choices=(60, 120, 240)" in SOURCE
    assert "default=120" in SOURCE
    assert "240 // args_cli.render_fps" in SOURCE
    assert "-1.0" in SOURCE


def test_launcher_is_continuous_event_only_and_has_no_hud_or_state_writer():
    for event in ("READY", "REQUEST", "RESULT", "RESET", "SNAPSHOT", "FAULT", "SESSION"):
        assert f"ELEVEN_ACTION_{event}" in SOURCE
    assert "while simulation_app.is_running()" in SOURCE
    assert "overlay" not in SOURCE.lower()
    assert "write_root_" + "pose_to_sim" not in SOURCE
    assert "write_root_" + "velocity_to_sim" not in SOURCE

