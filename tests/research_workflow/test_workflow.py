"""Behavior tests for the simulator-free research workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "scripts"))
import research_workflow as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        scratch = REPOSITORY / "artifacts/workflow-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "docs").mkdir()
        (self.root / "handoffs/active").mkdir(parents=True)
        (self.root / "docs/templates").mkdir()
        self.write("AGENTS.md", "# Rules\n\nRead the task first.\n")
        self.write("README.md", "# Project\n\n[Status](docs/STATUS.md).\n")
        self.write("docs/STATUS.md", "# Status\n\nPending execution.\n")
        self.write("handoffs/active/CAPACITY.md", "# Capacity\n\nAuthorized, no result.\n")
        self.write("docs/templates/task.md", "# {task_id}\n\n**状态：draft**\n\n{title}\n")
        self.layout = {
            "schema_version": 1,
            "required_paths": ["AGENTS.md", "README.md", "docs/STATUS.md"],
            "managed_documents": ["AGENTS.md", "README.md", "docs/STATUS.md"],
            "components": [{"path": "scripts", "purpose": "Entry points"}],
        }
        (self.root / "scripts").mkdir()
        self.state = {
            "schema_version": 1,
            "as_of": "2026-10-25",
            "active_task": "CAPACITY",
            "tasks": [{"id": "CAPACITY", "status": "authorized", "title": "Capacity",
                       "contract": "handoffs/active/CAPACITY.md", "plan": None, "report": None,
                       "summary": "Not yet run in this checkout."}],
        }
        self.save_registry()

    def write(self, path, content):
        (self.root / path).write_text(content, encoding="utf-8")

    def save_registry(self):
        self.write("docs/repository_layout.json", json.dumps(self.layout))
        self.write("docs/research_state.json", json.dumps(self.state))

    def test_valid_repository(self):
        self.assertEqual(workflow.check_repository(self.root), [])

    def test_missing_architecture_component(self):
        self.layout["components"][0]["path"] = "source/missing"
        self.save_registry()
        self.assertTrue(any("source/missing" in e for e in workflow.check_repository(self.root)))

    def test_broken_document_link(self):
        self.write("README.md", "# Project\n\n[Missing](docs/missing.md).\n")
        self.assertTrue(any("missing.md" in e for e in workflow.check_repository(self.root)))

    def test_fenced_examples_and_remote_links_are_not_local_targets(self):
        self.write("README.md", "# Project\n\n[Web](https://example.org).\n\n```md\n[X](absent.md)\n```\n")
        self.assertEqual(workflow.check_repository(self.root), [])

    def test_links_cannot_escape_repository(self):
        self.write("README.md", "# Project\n\n[Escape](../outside.md).\n")
        self.assertTrue(workflow.check_repository(self.root))

    def test_malformed_json_returns_diagnostic(self):
        self.write("docs/repository_layout.json", "{broken")
        self.assertTrue(any("repository_layout.json" in e for e in workflow.check_repository(self.root)))

    def test_wrong_registry_types_return_diagnostics(self):
        for value in ([], {"schema_version": 1, "required_paths": 1},
                      {"schema_version": 1, "required_paths": [], "managed_documents": [None], "components": []}):
            with self.subTest(value=value):
                self.write("docs/repository_layout.json", json.dumps(value))
                self.assertTrue(workflow.check_repository(self.root))

    def test_active_task_must_exist(self):
        self.state["active_task"] = "UNKNOWN"
        self.save_registry()
        self.assertTrue(any("UNKNOWN" in e for e in workflow.check_repository(self.root)))

    def test_missing_report_is_not_verified(self):
        self.state["tasks"][0]["report"] = "handoffs/reports/absent.md"
        self.save_registry()
        self.assertTrue(workflow.check_repository(self.root))

    def test_duplicate_task_ids_and_invalid_status_are_rejected(self):
        self.state["tasks"].append(dict(self.state["tasks"][0], status="probably_done"))
        self.save_registry()
        self.assertTrue(workflow.check_repository(self.root))

    def test_new_task_is_draft_and_does_not_change_authorization(self):
        before = (self.root / "docs/research_state.json").read_bytes()
        task = workflow.create_task(self.root, "RESEARCH-001", "A new question")
        self.assertIn("draft", task.read_text(encoding="utf-8"))
        self.assertIn("A new question", task.read_text(encoding="utf-8"))
        self.assertEqual((self.root / "docs/research_state.json").read_bytes(), before)
        with self.assertRaises((ValueError, FileExistsError)):
            workflow.create_task(self.root, "RESEARCH-001", "Do not overwrite")

    def test_task_ids_cannot_be_paths(self):
        for task_id in ("../ESCAPE", "/ESCAPE", "A/B", "A\\B", "", "a b"):
            with self.subTest(task_id=task_id), self.assertRaises(ValueError):
                workflow.create_task(self.root, task_id, "Invalid")

    def test_registered_task_id_cannot_create_a_second_contract(self):
        self.state["tasks"][0]["id"] = "ALREADY-REGISTERED"
        self.state["active_task"] = "ALREADY-REGISTERED"
        self.save_registry()
        with self.assertRaises(ValueError):
            workflow.create_task(self.root, "ALREADY-REGISTERED", "Duplicate registered identity")

    def test_new_run_initializes_metadata_without_claiming_execution(self):
        run = workflow.create_run(self.root, "EXP-001")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "initialized")
        self.assertEqual(manifest["artifacts"], [])
        self.assertIsNone(manifest["command"])
        self.assertTrue(workflow.verify_run(run), "No evidence must not pass verification")

    def test_nested_non_repository_root_does_not_claim_parent_git_identity(self):
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True, capture_output=True)
        subprocess.run(["git", "-c", f"safe.directory={self.root.as_posix()}", "-C", str(self.root),
                        "-c", "user.name=Workflow Test",
                        "-c", "user.email=workflow-test@example.invalid", "-c", "commit.gpgsign=false",
                        "-c", f"core.hooksPath={self.root / 'no-hooks'}", "commit", "--quiet", "--allow-empty",
                        "-m", "Temporary fixture"], check=True, capture_output=True)
        nested = self.root / "nested"
        nested.mkdir()
        run = workflow.create_run(nested, "EXP-NESTED")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["code"]["available"])

    def test_symlink_artifact_cannot_escape_run(self):
        run = self.prepared_run()
        outside = self.root / "outside.csv"
        outside.write_bytes((run / "summary.csv").read_bytes())
        link = run / "linked.csv"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable on this host")
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["path"] = "linked.csv"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(workflow.verify_run(run))

    def prepared_run(self):
        run = workflow.create_run(self.root, "EXP-002")
        data = b"measured evidence\n"
        (run / "summary.csv").write_bytes(data)
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"] = [{"path": "summary.csv", "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}]
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return run

    def test_verify_real_bytes_and_detect_tampering(self):
        run = self.prepared_run()
        self.assertEqual(workflow.verify_run(run), [])
        (run / "summary.csv").write_bytes(b"changed evidence\n")
        self.assertTrue(workflow.verify_run(run))

    def test_artifact_manifest_cannot_reference_parent(self):
        run = self.prepared_run()
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["path"] = "../outside.csv"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(workflow.verify_run(run))

    def test_duplicate_artifacts_and_invalid_size_are_rejected(self):
        run = self.prepared_run()
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"].append(dict(manifest["artifacts"][0], bytes=-1))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(workflow.verify_run(run))

    def test_cli_check_exit_code_and_context_from_another_directory(self):
        script = REPOSITORY / "scripts/research_workflow.py"
        result = subprocess.run([sys.executable, str(script), "--root", str(self.root), "context"],
                                cwd=self.root.parent, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CAPACITY", result.stdout)
        self.write("docs/STATUS.md", "[Broken](absent.md)\n")
        result = subprocess.run([sys.executable, str(script), "--root", str(self.root), "check"],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 1)

    def test_cli_record_lifecycle_and_tampered_evidence(self):
        script = REPOSITORY / "scripts/research_workflow.py"

        def cli(*args):
            return subprocess.run([sys.executable, str(script), "--root", str(self.root), *args],
                                  capture_output=True, text=True, encoding="utf-8")

        task = cli("new-task", "CLI-TASK", "--title", "CLI draft")
        self.assertEqual(task.returncode, 0, task.stderr)
        self.assertIn("draft", Path(task.stdout.strip()).read_text(encoding="utf-8"))
        duplicate = cli("new-task", "CLI-TASK", "--title", "Duplicate")
        self.assertEqual(duplicate.returncode, 1)
        created = cli("new-run", "CLI-RUN")
        self.assertEqual(created.returncode, 0, created.stderr)
        run = Path(created.stdout.strip())
        self.assertEqual(cli("verify-run", str(run)).returncode, 1)
        data = b"observed result\n"
        (run / "observed.txt").write_bytes(data)
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"] = [{"path": "observed.txt", "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()}]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(cli("verify-run", str(run)).returncode, 0)
        (run / "observed.txt").write_bytes(b"tampered\n")
        self.assertEqual(cli("verify-run", str(run)).returncode, 1)


if __name__ == "__main__":
    unittest.main()
