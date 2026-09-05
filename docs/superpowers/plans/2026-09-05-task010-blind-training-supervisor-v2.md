# TASK-010 Blind-GRU Three-Seed Training Supervisor V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training-only detached supervisor that runs Blind-GRU seeds 991001, 991002, and 991003 sequentially, detects incomplete zero exits and health degradation, supports audited checkpoint resume, and stops before validation.

**Architecture:** Keep the existing TASK-010 trainer as the only owner of Isaac Lab, PPO, and checkpoint writes. Add a pure-Python health/audit module and a new process supervisor that launches exactly one trainer child, derives progress from durable artifacts, and advances only after strict completion checks.

**Tech Stack:** Python 3.12, PyTorch checkpoint inspection, subprocess/fcntl process control, JSON/JSONL artifacts, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-task010-blind-training-supervisor-v2-design.md`

## Global Constraints

- The supervisor trains only seeds `991001`, `991002`, and `991003`, in that order.
- Every seed uses `visual_condition=blind`, 12 environments, 64 rollout steps, 1000 updates, checkpoint interval 50, `cuda:0`, and disabled validation.
- Do not modify CNN, GRU, Critic, PPO, reward, action, stomach scene, coverage, camera, dataset split, or environment configuration.
- The previous run `20260903T071833.170676Z-ff0184dd` is diagnostic only and must never be resumed by V2.
- V2 stops at `completed_training`; it must never launch validation, summary, or artifact-audit stages.
- Codex may run pure tests and help commands, but must not launch the three formal GPU seeds.

---

### Task 1: Implement Pure Training Health and Completion Audit

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_training_health.py`
- Create: `tests/stomach_coverage/test_task010_training_health.py`

**Interfaces:**
- Consumes: a seed training directory containing `metrics.jsonl`, `events.jsonl`, and `checkpoints/update_*.pt`.
- Produces: `parse_checkpoint_update(path) -> int`, `progress_snapshot(training_dir, now_epoch_s) -> dict`, `audit_completion(training_dir, expected) -> dict`, and `remaining_updates(checkpoint_update, target_update) -> int`.

- [ ] **Step 1: Write failing tests for update parsing and health states**

```python
def test_progress_classifies_degraded_and_stalled(tmp_path):
    training = write_metrics(tmp_path, updates=range(1, 11), tps=1.2)
    degraded = health.progress_snapshot(training, now_epoch_s=metric_time + 10)
    assert degraded["health"] == "degraded_performance"
    assert degraded["median_tps_last_10"] == pytest.approx(1.2)
    assert health.progress_snapshot(training, now_epoch_s=metric_time + 301)["health"] == "suspected_stall"
    assert health.progress_snapshot(training, now_epoch_s=metric_time + 901)["health"] == "critical_stall"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_coverage/test_task010_training_health.py -q
```

Expected: collection fails because `task010_training_health` does not exist.

- [ ] **Step 3: Implement progress and health classification**

```python
TARGET_UPDATE = 1000
DEGRADED_TPS = 10.0
SUSPECTED_STALL_S = 300.0
CRITICAL_STALL_S = 900.0

def parse_checkpoint_update(path: Path) -> int:
    return int(Path(path).stem.removeprefix("update_"))

def read_complete_jsonl(path: Path) -> list[dict]:
    data = path.read_bytes()
    complete = data if data.endswith(b"\n") else data.rsplit(b"\n", 1)[0] + b"\n"
    return [json.loads(line) for line in complete.splitlines() if line.strip()]

def remaining_updates(checkpoint_update: int, target_update: int = TARGET_UPDATE) -> int:
    if checkpoint_update < 0 or checkpoint_update > target_update:
        raise ValueError("checkpoint update is outside the formal training range")
    return target_update - checkpoint_update
```

`progress_snapshot` uses `read_complete_jsonl`, the newest parsed checkpoint, the last ten positive TPS values,
and the two fixed age thresholds to return the fields defined in the design spec.

Malformed trailing JSONL fragments are ignored only when the final line lacks its newline terminator; malformed complete lines raise `ValueError`.

- [ ] **Step 4: Write failing completion-audit tests**

Tests must create real `torch.save` checkpoints and prove:

```python
assert audit_completion(complete_dir, expected)["complete"] is True
assert audit_completion(update_437_dir, expected)["error_type"] == "incomplete_zero_exit"
assert audit_completion(nan_dir, expected)["error_type"] == "non_finite_metric"
assert audit_completion(wrong_identity_dir, expected)["error_type"] == "identity_mismatch"
```

- [ ] **Step 5: Implement strict completion audit**

Load the final checkpoint on CPU and compare `current_update`, `git_commit`, `config_hash`, `visual_dependence_config_sha256`, and `experiment_metadata.visual_condition`. Read the first `runner_initialized` event to compare its seed. Require every metric row to have `all_finite=true` and finite numeric values.

- [ ] **Step 6: Run health tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_coverage/test_task010_training_health.py -q

git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime/task010_training_health.py
git add -f tests/stomach_coverage/test_task010_training_health.py
git commit -m "feat: audit task010 training health"
```

Expected: all new tests pass.

### Task 2: Implement Training-Only Three-Seed State Machine

**Files:**
- Create: `scripts/stomach_coverage/task010_blind_training_supervisor_v2.py`
- Create: `tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py`
- Create: `tests/fixtures/task010_blind_training_fake_child.py`

**Interfaces:**
- Consumes: frozen TASK-010 base and visual-dependence configs, dependency audit, and `train_task010.py`.
- Produces: CLI commands `start`, `status`, `watch`, `continue`, and `diagnose`; durable `manifest.json`, `status.json`, and `events.jsonl`.

- [ ] **Step 1: Write failing stage-order and command tests**

```python
def test_stage_order_contains_only_three_training_seeds():
    assert supervisor.SEEDS == (991001, 991002, 991003)
    assert supervisor.stage_names() == (
        "train_blind_seed_991001",
        "train_blind_seed_991002",
        "train_blind_seed_991003",
    )

def test_training_command_is_frozen():
    command = supervisor.training_command(manifest, run_dir, seed=991001, resume=None)
    assert command_value(command, "--visual-condition") == "blind"
    assert command_value(command, "--max-updates") == "1000"
    assert command_value(command, "--save-interval") == "50"
    assert "validate_task010_checkpoint.py" not in " ".join(command)
```

- [ ] **Step 2: Verify RED, then implement constants, parser, and command construction**

Run the focused supervisor test and confirm import failure. Implement fixed seeds and a command builder that uses `train_task010.py`, `--backend isaac`, `--device cuda:0`, `--validation disabled`, and a seed-specific output directory.

- [ ] **Step 3: Write failing manifest and detached-start tests**

Tests must assert that start records Git branch/HEAD/worktree hash, config and dependency identities, package/GPU identity, exact training contract, and returns while the fake worker remains alive. A second active start must fail.

- [ ] **Step 4: Implement atomic persistence and detached start**

Use temporary files plus `flush`, `fsync`, and `os.replace`; use an append-only event stream; spawn `_worker` with `start_new_session=True`; create `latest` and `latest_run_path.txt` only after manifest and status exist.

- [ ] **Step 5: Write failing sequential execution tests**

The fake child records start/end events and creates valid metrics/checkpoints. Assert that the maximum concurrent fake children is one, seeds start in exact order, each completed seed is skipped on the next worker pass, and the final state is `completed_training` with no validation command.

- [ ] **Step 6: Implement file-locked worker and strict advancement**

Use a non-blocking `fcntl` lock. Before every seed, re-check the frozen Git/config identity. After the child exits, call `audit_completion`; only a passing audit permits the next seed.

- [ ] **Step 7: Run focused tests and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py -q

git add scripts/stomach_coverage/task010_blind_training_supervisor_v2.py
git add -f tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py \
  tests/fixtures/task010_blind_training_fake_child.py
git commit -m "feat: supervise task010 blind seeds"
```

### Task 3: Implement Failure Classification, Health Monitoring, and Resume

**Files:**
- Modify: `scripts/stomach_coverage/task010_blind_training_supervisor_v2.py`
- Modify: `tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py`

**Interfaces:**
- Consumes: Task 1 health snapshots and completion audits.
- Produces: explicit `paused_on_error` evidence and audited continuation from the latest complete checkpoint.

- [ ] **Step 1: Write failing failure-classification tests**

Cover zero exit at update 437, non-zero exit, non-finite metric, missing/empty/corrupt checkpoint, identity mismatch, and dead coordinator. Each test asserts the exact `error_type`, observed update, expected update, latest checkpoint, and that the next seed remains queued.

- [ ] **Step 2: Implement structured pause records**

Persist `error_type`, `error_summary`, `child_exit_code`, expected/observed update, latest checkpoint identity, and last five console lines. Never synthesize a successful state from exit code alone.

- [ ] **Step 3: Write failing health-monitor tests**

Use a fake child that slowly appends metrics. Assert status transitions through `starting`, `healthy`, `degraded_performance`, `suspected_stall`, and `critical_stall` without killing the live child.

- [ ] **Step 4: Implement periodic progress and ETA updates**

Every five seconds derive progress from durable metrics/checkpoints. ETA uses the median TPS of the latest ten complete updates and the fixed 768 transitions per update; if no reliable TPS exists, return `null` rather than inventing a duration.

- [ ] **Step 5: Write failing continuation tests**

```python
def test_continue_from_update_0400_requests_exactly_600_updates(tmp_path):
    run_dir = make_paused_run(tmp_path, seed=991002, checkpoint_update=400)
    command = supervisor.resume_command(run_dir)
    assert command_value(command, "--max-updates") == "600"
    assert command_value(command, "--resume-checkpoint").endswith("update_0400.pt")

def test_continue_refuses_changed_git_head(tmp_path, monkeypatch):
    run_dir = make_paused_run(tmp_path, seed=991002, checkpoint_update=400)
    monkeypatch.setattr(supervisor, "git_identity", lambda: {"commit": "changed"})
    with pytest.raises(RuntimeError, match="identity"):
        supervisor.prepare_continue(run_dir)

def test_continue_refuses_active_child(tmp_path, monkeypatch):
    run_dir = make_paused_run(tmp_path, seed=991002, checkpoint_update=400)
    monkeypatch.setattr(supervisor, "pid_alive", lambda pid: True)
    with pytest.raises(RuntimeError, match="alive"):
        supervisor.prepare_continue(run_dir)

def test_continue_never_restarts_completed_seed(tmp_path):
    run_dir = make_paused_run(tmp_path, seed=991003, checkpoint_update=400)
    state = json.loads((run_dir / "status.json").read_text())
    assert state["stages"]["train_blind_seed_991001"]["state"] == "completed"
    assert state["stages"]["train_blind_seed_991002"]["state"] == "completed"
```

- [ ] **Step 6: Implement continuation and read-only diagnostics**

`continue` validates identity and the latest checkpoint before spawning a new worker. `status`, `watch`, and `diagnose` never import Isaac Lab or mutate the run directory. `watch` stopping on Ctrl-C cannot signal the worker.

- [ ] **Step 7: Run both focused suites and commit**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_coverage/test_task010_training_health.py \
  tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py -q

git add scripts/stomach_coverage/task010_blind_training_supervisor_v2.py
git add -f tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py
git commit -m "fix: harden task010 blind training lifecycle"
```

### Task 4: Document Operation and Perform Final Verification

**Files:**
- Create: `docs/TASK010_BLIND_TRAINING_SUPERVISOR_V2.md`
- Modify: `docs/PROJECT_RUN_LOG.md`

**Interfaces:**
- Consumes: final CLI help and verified behavior.
- Produces: Chinese commands for user-owned smoke/formal launch and a concise project history entry.

- [ ] **Step 1: Write Chinese operating instructions**

Document one short GPU smoke command, formal `start`, read-only `status`, `watch`, `diagnose`, and failure-only `continue`. State explicitly that Codex did not start formal seeds and that the old run is diagnostic only.

- [ ] **Step 2: Run CLI and test verification**

```bash
./run_isaaclab.sh -p scripts/stomach_coverage/task010_blind_training_supervisor_v2.py --help

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest \
  tests/stomach_coverage/test_task010_training_health.py \
  tests/stomach_coverage/test_task010_blind_training_supervisor_v2.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=source/robotarm_magnetic_lab \
  /mnt/isaac-linux/IsaacLab/_isaac_sim/python.sh -m pytest tests/stomach_coverage -q

python -m compileall -q source/robotarm_magnetic_lab/robotarm_magnetic_lab/runtime \
  scripts/stomach_coverage/task010_blind_training_supervisor_v2.py

git diff --check
```

Expected: CLI exits 0; focused and full suites have zero failures; compileall and diff check exit 0.

- [ ] **Step 3: Audit that no formal process or artifact was created**

Check process commands for the V2 supervisor and confirm the formal artifact root contains no started formal run. Record this as negative evidence in the run log.

- [ ] **Step 4: Commit documentation and final log**

```bash
git add docs/TASK010_BLIND_TRAINING_SUPERVISOR_V2.md docs/PROJECT_RUN_LOG.md
git commit -m "docs: operate task010 blind training supervisor"
```

- [ ] **Step 5: Report user-owned execution commands**

Return the complete local HEAD, test counts and exit codes, clean/known worktree status, smoke command, formal start command, and monitoring commands. Do not claim the three formal seeds have run.
