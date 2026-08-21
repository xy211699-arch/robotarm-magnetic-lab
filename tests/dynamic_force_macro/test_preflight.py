from pathlib import Path


def test_preflight_records_required_geometry_and_api_path():
    source = (Path(__file__).resolve().parents[2] / "scripts/dynamic_force_macro/inspect_prerequisites.py").read_text(encoding="utf-8")
    for token in ("radius_m", "cylinder_height_m", "mass_kg", "inertia", "equivalent_com_wrench", "camera_side_local_axis_sign", "body_ccd", "scene_ccd"):
        assert token in source
