from scripts.local_primitives.inspect_local_primitives_prerequisites import (
    build_gate, scan_runtime_contract, source_report,
)


def valid_report():
    report = source_report()
    report["tasks"]["registered"] = [
        "Template-Robotarm-Magnetic-Local-Primitives-Flat-Lab-v0",
        "Template-Robotarm-Magnetic-Local-Primitives-Stomach-Lab-v0",
    ]
    report["contact_points"]["read_only_access"] = True
    report["isolation"]["flat_action_terms"] = ["local_primitive"]
    report["gate"] = build_gate(report)
    return report


def test_complete_preflight_contract_passes():
    report = valid_report()
    assert report["gate"] == {"status": "pass", "failures": []}


def test_contact_points_and_torque_are_decision_gates():
    report = valid_report()
    report["contact_points"]["read_only_access"] = False
    assert build_gate(report)["status"] == "needs_decision"
    report = valid_report()
    report["wrench_api"]["direct_force_and_torque"] = False
    assert build_gate(report)["status"] == "needs_decision"


def test_runtime_scan_has_no_state_writer():
    contract = scan_runtime_contract()
    assert contract["forbidden_calls"] == []
    assert contract["uses_com_wrench"] is True
