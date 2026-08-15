# Idealized Capsule Surface Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate Isaac Lab task that accepts fifteen deterministic discrete capsule-motion actions and drives the capsule along the approved stomach inner surface with ideal one-second pose tracking, while reusing the existing 1 Hz RGB and occlusion-aware coverage evaluator.

**Architecture:** Keep the delivered eleven-action magnetic executor and every existing task unchanged. Add a pure NumPy ideal-surface controller with explicit surface, contact, trajectory, mask, and state-machine boundaries, then wrap it in a one-environment Isaac Lab `ActionTerm` that writes a continuous kinematic capsule target at every physics substep. Reuse the approved P0 stomach mesh and coverage runtime only through privileged evaluator interfaces.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Isaac Lab/Isaac Sim 6.0 APIs already pinned by the repository, USD/PhysX geometry queries verified during preflight, pytest, and the delivered coverage/teleoperation infrastructure.

## Global Constraints

- Work only on `feature/TASK-002-ideal-surface-controller`, created from the exact head of `workflow/TASK-002-ideal-surface-controller`.
- Preserve all eleven `AtomicAction` IDs, magnetic templates, magnetic-force behavior, existing atomic task registrations, P0 coverage semantics, camera calibration, stomach assets, robot assets, and previous acceptance evidence.
- Register a new task named `Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0`; do not repurpose the existing atomic stomach task.
- Freeze the new action schema as `ideal_surface_v1` with exactly fifteen scalar IDs `0..14`.
- Use one environment, 240 Hz physics, one 1.0 s action per environment step, 1 Hz policy RGB, and internal target updates on every physics substep.
- Keep capsule pose, surface mesh, contact geometry, coverage, rays, and controller diagnostics out of Actor observations and out of the existing deployable magnetic-action contracts.
- Allow privileged capsule and mesh truth only inside the new ideal controller, asymmetric-Critic channel, evaluator, visualization, validation, and offline records.
- Use the approved visual luminal mesh `/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/VisualMesh/Stomach` as both the surface-navigation reference and coverage denominator unless preflight proves the live path changed.
- Stop with `needs_decision` before controller implementation if capsule collision geometry, capsule long-axis convention, camera image-up convention, mesh inward normal, kinematic pose-write API, or the initial surface contact cannot be identified unambiguously.
- Treat contact-limited and open-boundary-limited motions as `DONE` with flags. Reserve `HARD_FAILURE` for nonfinite state, lost surface, nonadjacent surface jump, or penetration above the hard threshold.
- Use test-driven development and commit after every independently passing task.
- Keep simulator logs, screenshots, videos, trajectory arrays, random-run traces, and generated meshes outside Git; report their paths, sizes, and SHA-256 hashes.
- The repository ignores `/tests/` and `docs/superpowers/`; use `git add -f` only for the explicitly named new test, spec, and plan files.

---

## Expected File Map

```text
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/types.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/config.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/geometry.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/surface_mesh.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/capsule_geometry.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/contact.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/action_mask.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/trajectory.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/controller.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_ideal_surface_stomach_env_cfg.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/ideal_surface_keyboard.py
source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
scripts/ideal_surface/inspect_ideal_surface_prerequisites.py
scripts/ideal_surface/validate_ideal_surface_geometry.py
scripts/ideal_surface/validate_ideal_surface_stomach.py
scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py
tests/ideal_surface/conftest.py
tests/ideal_surface/test_preflight_schema.py
tests/ideal_surface/test_action_contract.py
tests/ideal_surface/test_surface_mesh.py
tests/ideal_surface/test_capsule_contact.py
tests/ideal_surface/test_trajectory_controller.py
tests/ideal_surface/test_ideal_surface_task_cfg.py
tests/ideal_surface/test_ideal_surface_keyboard.py
docs/IDEAL_SURFACE_CONTROLLER.md
handoffs/reports/TASK-002-ideal-surface-controller-report.md
```

## Task 1: Run the Mandatory Linux Geometry and API Preflight

**Files:**
- Create: `scripts/ideal_surface/inspect_ideal_surface_prerequisites.py`
- Create: `tests/ideal_surface/conftest.py`
- Create: `tests/ideal_surface/test_preflight_schema.py`
- Output outside Git: `logs/ideal_surface_preflight/<timestamp>/prerequisites.json`

**Interfaces:**
- Consumes: the live stomach task, approved luminal mesh path, capsule rigid-body prim, capsule camera prim, and current Isaac Lab installation.
- Produces: one JSON object with keys `repository`, `task`, `capsule`, `camera`, `surface`, `pose_write_api`, `initial_contact`, and `gate`.

- [ ] **Step 1: Write the failing preflight-schema test**

```python
import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ideal_surface"
    / "inspect_ideal_surface_prerequisites.py"
)
SPEC = importlib.util.spec_from_file_location("ideal_surface_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

def valid_report():
    return {
        "repository": {"commit": "a" * 40, "branch": "feature/test"},
        "task": {
            "id": "Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"
        },
        "capsule": {
            "shape_class": "spherocylinder",
            "radius_m": 0.005,
            "cylinder_half_length_m": 0.0075,
            "long_axis_local": [0, 0, 1],
        },
        "camera": {
            "optical_axis_local": [0, 0, 1],
            "image_up_axis_local": [0, -1, 0],
        },
        "surface": {
            "vertex_count": 24529,
            "triangle_count": 49047,
            "geometry_sha256": (
                "67b4e06a4f5cfc3b8d51e5411942226"
                "d4bcabd3a6a937a456057e408a990ad36"
            ),
            "inward_normal_confirmed": True,
        },
        "pose_write_api": {
            "pose_method": "write_root_pose_to_sim",
            "velocity_method": "write_root_velocity_to_sim",
            "quaternion_order": "wxyz",
        },
        "initial_contact": {"valid": True, "triangle_id": 1},
        "gate": {"status": "pass", "failures": []},
    }

def test_preflight_report_requires_all_design_gates():
    report = valid_report()
    MODULE.validate_preflight_report(report)
    assert set(report) == MODULE.REQUIRED_REPORT_KEYS
    assert report["task"]["id"] == (
        "Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0"
    )
    assert report["capsule"]["radius_m"] > 0.0
    assert report["capsule"]["cylinder_half_length_m"] > 0.0
    assert report["capsule"]["long_axis_local"] in ([0, 0, 1], [0, 0, -1])
    assert report["camera"]["optical_axis_local"] == [0, 0, 1]
    assert report["surface"]["triangle_count"] == 49047
    assert report["surface"]["vertex_count"] == 24529
    assert report["pose_write_api"]["pose_method"]
    assert report["pose_write_api"]["velocity_method"]
    assert report["gate"]["status"] in {"pass", "needs_decision"}
```

- [ ] **Step 2: Run the focused test and verify that it fails because the inspector module does not exist**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_preflight_schema.py -q
```

Expected: FAIL from the missing preflight module, not from an unrelated import error.

- [ ] **Step 3: Implement the read-only inspector**

The inspector must enumerate the capsule rigid-body and collider prims, compute collision bounds in capsule-local coordinates, identify the cylindrical radius and half-length, record the local long axis, record camera optical and image-up axes, verify the approved visual mesh hash, verify mesh edge manifold statistics and normal orientation, identify the closest initial surface point, and introspect the live `RigidObject` methods used to write pose and velocity.

```python
def build_gate(report: dict) -> dict:
    failures = []
    if report["capsule"]["shape_class"] != "spherocylinder":
        failures.append("capsule collision is not an unambiguous spherocylinder")
    if not report["surface"]["inward_normal_confirmed"]:
        failures.append("stomach inward normal is ambiguous")
    if not report["pose_write_api"]["pose_method"]:
        failures.append("no verified root-pose write API")
    if not report["pose_write_api"]["velocity_method"]:
        failures.append("no verified root-velocity write API")
    if not report["initial_contact"]["valid"]:
        failures.append("initial capsule pose has no valid surface contact")
    return {
        "status": "pass" if not failures else "needs_decision",
        "failures": failures,
    }
```

- [ ] **Step 4: Run the inspector in the current delivered stomach scene**

Run:

```bash
./run_isaaclab.sh -p scripts/ideal_surface/inspect_ideal_surface_prerequisites.py \
  --task Template-Robotarm-Magnetic-Atomic-Stomach-Teleop-Lab-v0 \
  --output logs/ideal_surface_preflight
```

Expected: a versioned JSON report and `IDEAL_SURFACE_PREFLIGHT status=pass`. If status is `needs_decision`, write the required report and stop every later task.

- [ ] **Step 5: Re-run the schema test and commit the passing preflight**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_preflight_schema.py -q
git add scripts/ideal_surface/inspect_ideal_surface_prerequisites.py
git add -f tests/ideal_surface/conftest.py tests/ideal_surface/test_preflight_schema.py
git commit -m "test: inspect ideal surface controller prerequisites"
```

## Task 2: Freeze the Fifteen-Action Contract and Minimal Mask

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/types.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/config.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/action_mask.py`
- Create: `tests/ideal_surface/test_action_contract.py`

**Interfaces:**
- Produces: `IdealSurfaceAction`, `IdealSurfaceConfig`, `ControllerState`, `IdealActionStatus`, `SurfaceFlags`, `ControllerSnapshot`, `IdealActionResult`, and `compute_action_mask(flags, cfg) -> np.ndarray`.
- Consumes: no Isaac Sim imports; this task must remain pure Python and NumPy.

- [ ] **Step 1: Write the failing action-contract tests**

```python
def test_frozen_action_ids_are_contiguous_and_unique():
    assert [(item.name, int(item)) for item in IdealSurfaceAction] == [
        ("HOLD", 0),
        ("START_TILT_000", 1),
        ("START_TILT_045", 2),
        ("START_TILT_090", 3),
        ("START_TILT_135", 4),
        ("START_TILT_180", 5),
        ("START_TILT_225", 6),
        ("START_TILT_270", 7),
        ("START_TILT_315", 8),
        ("TILT_MORE", 9),
        ("RISE", 10),
        ("PRECESS_POS", 11),
        ("PRECESS_NEG", 12),
        ("ROLL_POS", 13),
        ("ROLL_NEG", 14),
    ]

def test_default_config_matches_ideal_surface_v1():
    cfg = IdealSurfaceConfig()
    assert cfg.schema_version == "ideal_surface_v1"
    assert cfg.action_duration_s == 1.0
    assert cfg.tilt_step_rad == pytest.approx(math.radians(15.0))
    assert cfg.precession_step_rad == pytest.approx(math.radians(15.0))
    assert cfg.roll_arc_length_m == pytest.approx(0.004)
    assert cfg.upright_enter_rad == pytest.approx(math.radians(5.0))
    assert cfg.upright_exit_rad == pytest.approx(math.radians(8.0))
```

- [ ] **Step 2: Write mask tests for upright, tilted, side-contact, and contact-limited states**

```python
def enabled(mask):
    return set(np.flatnonzero(np.asarray(mask, dtype=bool)).tolist())

def test_mask_is_minimal_and_state_dependent():
    cfg = IdealSurfaceConfig()
    upright = compute_action_mask(
        SurfaceFlags(upright=True, side_contact=False), cfg
    )
    assert enabled(upright) == {0, 1, 2, 3, 4, 5, 6, 7, 8}

    tilted = compute_action_mask(
        SurfaceFlags(upright=False, side_contact=False), cfg
    )
    assert enabled(tilted) == {0, 9, 10, 11, 12}

    side = compute_action_mask(
        SurfaceFlags(upright=False, side_contact=True), cfg
    )
    assert enabled(side) == {0, 9, 10, 11, 12, 13, 14}

    limited = compute_action_mask(
        SurfaceFlags(
            upright=False,
            side_contact=True,
            contact_limited=True,
        ),
        cfg,
    )
    assert enabled(limited) == {0, 10, 11, 12, 13, 14}
```

- [ ] **Step 3: Run the tests and verify the expected missing-module failure**

Run:

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_action_contract.py -q
```

Expected: FAIL because `controllers.ideal_surface` is absent.

- [ ] **Step 4: Implement the frozen types and validation**

```python
class IdealSurfaceAction(IntEnum):
    HOLD = 0
    START_TILT_000 = 1
    START_TILT_045 = 2
    START_TILT_090 = 3
    START_TILT_135 = 4
    START_TILT_180 = 5
    START_TILT_225 = 6
    START_TILT_270 = 7
    START_TILT_315 = 8
    TILT_MORE = 9
    RISE = 10
    PRECESS_POS = 11
    PRECESS_NEG = 12
    ROLL_POS = 13
    ROLL_NEG = 14

class ControllerState(str, Enum):
    READY = "READY"
    EXECUTING = "EXECUTING"
    TERMINAL_FAULT = "TERMINAL_FAULT"

class IdealActionStatus(str, Enum):
    DONE = "DONE"
    HARD_FAILURE = "HARD_FAILURE"

@dataclass(frozen=True)
class SurfaceFlags:
    upright: bool
    side_contact: bool
    contact_limited: bool = False
    boundary_limited: bool = False
    no_effect: bool = False

@dataclass(frozen=True)
class ControllerSnapshot:
    sim_time_s: float
    position_world: np.ndarray
    quaternion_for_sim: np.ndarray
    axis_world: np.ndarray
    image_up_world: np.ndarray
    surface_point_world: np.ndarray
    surface_normal_world: np.ndarray
    surface_triangle_id: int
    theta_rad: float
    phi_rad: float
    flags: SurfaceFlags

@dataclass(frozen=True)
class IdealActionResult:
    request_id: int
    action: IdealSurfaceAction
    status: IdealActionStatus
    started_at_s: float
    ended_at_s: float
    contact_limited: bool
    boundary_limited: bool
    no_effect: bool
    hard_failure_detail: str | None
    final_position_world: np.ndarray
    final_quaternion_for_sim: np.ndarray
    final_axis_world: np.ndarray
    final_tilt_rad: float
    final_azimuth_rad: float
    maximum_penetration_m: float

@dataclass(frozen=True)
class IdealSurfaceConfig:
    schema_version: str = "ideal_surface_v1"
    action_duration_s: float = 1.0
    tilt_step_rad: float = math.radians(15.0)
    precession_step_rad: float = math.radians(15.0)
    roll_arc_length_m: float = 0.004
    upright_enter_rad: float = math.radians(5.0)
    upright_exit_rad: float = math.radians(8.0)
    logical_stability_s: float = 0.1
    side_contact_separation_fraction: float = 0.25
    contact_clearance_radius_fraction: float = 0.02
    planned_penetration_radius_fraction: float = 0.01
    hard_penetration_radius_fraction: float = 0.05
    recovery_query_radius_scale: float = 2.0
```

- [ ] **Step 5: Run focused tests and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_action_contract.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface
git add -f tests/ideal_surface/test_action_contract.py
git commit -m "feat: freeze fifteen-action ideal surface contract"
```

## Task 3: Build Deterministic Surface and Capsule Geometry

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/geometry.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/surface_mesh.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/capsule_geometry.py`
- Create: `tests/ideal_surface/test_surface_mesh.py`

**Interfaces:**
- Consumes: `coverage.reference_mesh.ReferenceMesh` and preflight-confirmed capsule dimensions.
- Produces: `LocalFrame`, `SurfaceNavigationMesh.from_reference(reference, inward_sign)`, `SurfaceHit`, `Spherocylinder`, `quintic(tau)`, and quaternion/matrix conversion helpers.

- [ ] **Step 1: Write synthetic mesh tests**

The fixtures must include a two-triangle plane, a bent four-triangle strip, an open boundary, and two spatially close but topologically disconnected planes.

```python
def test_local_search_never_jumps_to_disconnected_nearby_sheet():
    mesh = disconnected_parallel_sheets(gap_m=0.001)
    hit = mesh.advance(
        triangle_id=0,
        point_world=np.array([0.25, 0.25, 0.0]),
        tangent_delta_world=np.array([0.2, 0.0, 0.0]),
        recovery_radius_m=0.02,
    )
    assert hit.component_id == mesh.component_ids[0]
    assert hit.triangle_id in {0, 1}

def test_open_edge_is_reported_as_boundary_not_surface_loss():
    mesh = unit_square_mesh()
    hit = mesh.advance(
        triangle_id=1,
        point_world=np.array([0.9, 0.5, 0.0]),
        tangent_delta_world=np.array([0.2, 0.0, 0.0]),
        recovery_radius_m=0.1,
    )
    assert hit.boundary_limited
    assert np.allclose(hit.point_world[0], 1.0)
```

- [ ] **Step 2: Write local-frame and support-geometry tests**

```python
def test_direction_bins_use_image_up_reference():
    frame = LocalFrame(
        point_world=np.zeros(3),
        normal_world=np.array([0.0, 0.0, 1.0]),
        image_up_tangent_world=np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(frame.direction(math.radians(0)), [1, 0, 0])
    np.testing.assert_allclose(frame.direction(math.radians(90)), [0, 1, 0], atol=1e-12)

def test_spherocylinder_support_height_changes_with_tilt():
    capsule = Spherocylinder(radius_m=0.005, cylinder_half_length_m=0.0075)
    upright = capsule.support_distance(np.array([0, 0, 1]), np.array([0, 0, 1]))
    side = capsule.support_distance(np.array([1, 0, 0]), np.array([0, 0, 1]))
    assert upright == pytest.approx(0.0125)
    assert side == pytest.approx(0.005)
```

- [ ] **Step 3: Run tests and verify they fail before implementation**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_surface_mesh.py -q
```

- [ ] **Step 4: Implement mesh adjacency, deterministic tie-breaking, and local projection**

```python
@dataclass(frozen=True)
class SurfaceHit:
    point_world: np.ndarray
    normal_world: np.ndarray
    triangle_id: int
    component_id: int
    boundary_limited: bool = False

class SurfaceNavigationMesh:
    @classmethod
    def from_reference(
        cls,
        reference: ReferenceMesh,
        inward_sign: int,
    ) -> "SurfaceNavigationMesh":
        triangles = np.asarray(reference.triangles, dtype=np.int64)
        vertices = np.asarray(reference.vertices_world, dtype=np.float64)
        normals = oriented_triangle_normals(vertices, triangles, inward_sign)
        adjacency, boundary_edges = build_edge_adjacency(triangles)
        component_ids = connected_components(adjacency)
        return cls(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
            adjacency=adjacency,
            boundary_edges=boundary_edges,
            component_ids=component_ids,
        )

    def advance(
        self,
        triangle_id: int,
        point_world: np.ndarray,
        tangent_delta_world: np.ndarray,
        recovery_radius_m: float,
    ) -> SurfaceHit:
        target = np.asarray(point_world) + np.asarray(tangent_delta_world)
        candidates = self.local_candidate_triangles(int(triangle_id))
        ranked = self.rank_projected_candidates(target, candidates)
        if ranked:
            return self.surface_hit_from_ranked(ranked[0])
        recovered = self.recovery_candidates(
            target,
            component_id=int(self.component_ids[int(triangle_id)]),
            radius_m=float(recovery_radius_m),
        )
        if not recovered:
            raise SurfaceLostError("no same-component surface candidate")
        return self.surface_hit_from_ranked(recovered[0])
```

The implementation must build edge-to-triangle adjacency, connected-component IDs, boundary-edge flags, consistently oriented triangle normals, closest-point-on-triangle queries, and a deterministic candidate order of distance, triangle ID, then barycentric coordinates. A recovery candidate from another connected component is forbidden even when it is spatially nearer.

- [ ] **Step 5: Implement `Spherocylinder` and exact support functions**

```python
@dataclass(frozen=True)
class Spherocylinder:
    radius_m: float
    cylinder_half_length_m: float

    def support_distance(self, axis_world, normal_world) -> float:
        u = normalized(axis_world)
        n = normalized(normal_world)
        return self.radius_m + self.cylinder_half_length_m * abs(float(u @ n))

    def effective_roll_radius(self) -> float:
        return self.radius_m
```

- [ ] **Step 6: Run tests and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_surface_mesh.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/geometry.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/surface_mesh.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/capsule_geometry.py
git add -f tests/ideal_surface/test_surface_mesh.py
git commit -m "feat: model deterministic stomach surface geometry"
```

## Task 4: Implement Contact Classification and Active Tilt Anchors

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/contact.py`
- Create: `tests/ideal_surface/test_capsule_contact.py`

**Interfaces:**
- Consumes: `SurfaceNavigationMesh`, `Spherocylinder`, capsule pose, current triangle, current tilt direction, and `IdealSurfaceConfig`.
- Produces: `ContactAssessment`, `select_active_anchor(contacts_world, center_world, tilt_direction_world, triangle_ids) -> ActiveAnchor`, and `assess_pose(mesh, capsule, pose, active_triangle, cfg) -> ContactAssessment`.

- [ ] **Step 1: Write tests that distinguish support, side contact, tilt blocking, and hard penetration**

```python
def test_any_contact_is_not_side_contact():
    assessment = assess_pose(
        plane_mesh(),
        capsule(),
        upright_pose(),
        active_triangle=0,
        cfg=IdealSurfaceConfig(),
    )
    assert assessment.support_valid
    assert not assessment.side_contact
    assert not assessment.contact_limited

def test_two_separated_barrel_samples_create_stable_side_contact():
    detector = ContactClassifier(IdealSurfaceConfig(), capsule())
    for _ in range(stability_steps(dt=1 / 240, window_s=0.1)):
        result = detector.observe(
            side_pose(),
            barrel_clearances=np.array([0.0, 0.0]),
            barrel_axial_parameters=np.array([-0.5, 0.5]),
            dt=1 / 240,
        )
    assert result.side_contact

def test_penetrating_next_pose_clips_without_hard_failure():
    result = assess_swept_target(
        current=safe_pose(),
        proposed=slightly_penetrating_pose(),
        radius_m=0.005,
        cfg=IdealSurfaceConfig(),
    )
    assert result.contact_limited
    assert not result.hard_failure
```

- [ ] **Step 2: Write deterministic active-anchor tests**

```python
def test_side_contact_anchor_is_extreme_opposite_tilt_ray():
    contacts = np.array([
        [-0.0075, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0075, 0.0, 0.0],
    ])
    anchor = select_active_anchor(
        contacts_world=contacts,
        center_world=np.zeros(3),
        tilt_direction_world=np.array([1.0, 0.0, 0.0]),
        triangle_ids=np.array([3, 2, 1]),
    )
    np.testing.assert_allclose(anchor.point_world, [-0.0075, 0.0, 0.0])
```

- [ ] **Step 3: Run the tests and verify the missing implementation fails**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_capsule_contact.py -q
```

- [ ] **Step 4: Implement geometric contact classification**

```python
@dataclass(frozen=True)
class ContactAssessment:
    support_valid: bool
    side_contact: bool
    contact_limited: bool
    boundary_limited: bool
    hard_failure: bool
    maximum_penetration_m: float
    support_point_world: np.ndarray
    active_triangle: int
```

Use capsule surface samples derived from the preflight-confirmed spherocylinder, local mesh projection, and normalized thresholds from `IdealSurfaceConfig`. PhysX force and impulse may be recorded for diagnostics but cannot be the sole classification source.

- [ ] **Step 5: Run tests and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_capsule_contact.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/contact.py
git add -f tests/ideal_surface/test_capsule_contact.py
git commit -m "feat: classify ideal capsule wall contact"
```

## Task 5: Implement Continuous Trajectories and the Minimal State Machine

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/trajectory.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/controller.py`
- Create: `tests/ideal_surface/test_trajectory_controller.py`

**Interfaces:**
- Consumes: action ID, `ControllerSnapshot`, `SurfaceNavigationMesh`, `Spherocylinder`, configuration, request ID, and physics `dt`.
- Produces: `ControllerOutput`, `IdealSurfaceController.submit(action_id, snapshot, request_id)`, `IdealSurfaceController.step(dt) -> ControllerOutput`, `action_mask()`, `acknowledge_result()`, and one terminal `IdealActionResult`.

- [ ] **Step 1: Write trajectory invariant tests**

```python
@pytest.mark.parametrize("tau, expected", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
def test_quintic_endpoints(tau, expected):
    assert quintic(tau) == pytest.approx(expected)

def test_start_tilt_maps_each_id_to_one_unique_axis():
    controller = plane_controller()
    outputs = []
    for action_id in range(1, 9):
        controller.reset(upright_snapshot())
        controller.submit(action_id, upright_snapshot(), request_id=action_id)
        outputs.append(run_to_done(controller).final_axis_world)
    rounded = {tuple(np.round(axis, 8)) for axis in outputs}
    assert len(rounded) == 8

def test_precession_keeps_tilt_and_changes_azimuth_by_fifteen_degrees():
    controller = plane_controller()
    controller.reset(tilted_snapshot(theta_deg=45, phi_deg=0))
    controller.submit(IdealSurfaceAction.PRECESS_POS, controller.snapshot, 1)
    result = run_to_done(controller)
    assert result.final_tilt_rad == pytest.approx(math.radians(45), abs=math.radians(0.2))
    assert result.final_azimuth_rad == pytest.approx(math.radians(15), abs=math.radians(0.2))
```

- [ ] **Step 2: Write fixed-boundary, clipping, roll, and reset tests**

```python
def test_contact_limited_motion_holds_until_one_second_boundary():
    controller = blocked_tilt_controller(block_at_s=0.6)
    controller.submit(IdealSurfaceAction.TILT_MORE, controller.snapshot, 4)
    outputs = [controller.step(1 / 240) for _ in range(240)]
    assert outputs[-1].result.status is IdealActionStatus.DONE
    assert outputs[-1].result.contact_limited
    assert outputs[-1].result.duration_s == pytest.approx(1.0)
    assert all_close(outputs[150].pose, outputs[-1].pose)

def test_positive_roll_obeys_right_hand_no_slip_sign():
    controller = plane_controller(side_contact=True)
    before = controller.snapshot
    controller.submit(IdealSurfaceAction.ROLL_POS, before, 9)
    result = run_to_done(controller)
    expected = -0.004 * np.cross(before.surface_normal_world, before.axis_tangent_world)
    np.testing.assert_allclose(
        result.final_position_world - before.position_world,
        expected,
        atol=1e-4,
    )
```

- [ ] **Step 3: Run tests and verify they fail**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_trajectory_controller.py -q
```

- [ ] **Step 4: Implement deterministic action target generation**

```python
def target_for_action(action, snapshot, cfg):
    if action in START_TILT_ACTIONS:
        phi = math.radians(45.0 * (int(action) - 1))
        return tilt_target(snapshot, theta_target=cfg.tilt_step_rad, phi_target=phi)
    if action is IdealSurfaceAction.TILT_MORE:
        return tilt_target(
            snapshot,
            theta_target=min(snapshot.theta_rad + cfg.tilt_step_rad, math.pi / 2),
            phi_target=snapshot.phi_rad,
        )
    if action is IdealSurfaceAction.RISE:
        target = max(snapshot.theta_rad - cfg.tilt_step_rad, 0.0)
        if target <= cfg.upright_enter_rad:
            target = 0.0
        return tilt_target(snapshot, theta_target=target, phi_target=snapshot.phi_rad)
    if action in (IdealSurfaceAction.PRECESS_POS, IdealSurfaceAction.PRECESS_NEG):
        sign = 1.0 if action is IdealSurfaceAction.PRECESS_POS else -1.0
        return precession_target(snapshot, sign * cfg.precession_step_rad)
    if action in (IdealSurfaceAction.ROLL_POS, IdealSurfaceAction.ROLL_NEG):
        sign = 1.0 if action is IdealSurfaceAction.ROLL_POS else -1.0
        return roll_target(snapshot, sign, cfg.roll_arc_length_m)
    return hold_target(snapshot)
```

- [ ] **Step 5: Implement the three-state executor**

```python
@dataclass(frozen=True)
class ControllerOutput:
    position_world: np.ndarray
    quaternion_for_sim: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray
    flags: SurfaceFlags
    result: IdealActionResult | None = None

class IdealSurfaceController:
    @property
    def ready(self) -> bool:
        return self.state is ControllerState.READY and self.last_result is None

    def submit(self, action_id: int, snapshot: ControllerSnapshot, request_id: int) -> bool:
        if self.state is not ControllerState.READY:
            return False
        action = IdealSurfaceAction(action_id)
        if not self.action_mask()[int(action)]:
            self._start_no_effect(action, request_id)
            return True
        self._begin_trajectory(action, snapshot, request_id)
        self.state = ControllerState.EXECUTING
        return True

    def step(self, dt: float) -> ControllerOutput:
        if self.state is ControllerState.TERMINAL_FAULT:
            return self._held_failure_output()
        self._elapsed_s = min(self._elapsed_s + float(dt), self.cfg.action_duration_s)
        output = self._evaluate_quintic_target()
        assessment = self.contact.assess(output.pose)
        if assessment.hard_failure:
            return self._enter_terminal_fault(assessment)
        if assessment.contact_limited or assessment.boundary_limited:
            self._latch_last_safe_pose(assessment)
        return self._finish_at_fixed_boundary(output)

    def acknowledge_result(self) -> None:
        if self.last_result is not None and self.state is not ControllerState.TERMINAL_FAULT:
            self.last_result = None
            self.state = ControllerState.READY
```

The controller must maintain logical upright hysteresis, side-contact time hysteresis, fixed one-second completion, request de-duplication, last-safe target containment, and `DONE/HARD_FAILURE` result compatibility. It must never read coverage state.

- [ ] **Step 6: Run focused tests and commit**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_trajectory_controller.py -q
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/trajectory.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface/controller.py
git add -f tests/ideal_surface/test_trajectory_controller.py
git commit -m "feat: execute continuous ideal surface actions"
```

## Task 6: Add the Isaac Lab ActionTerm and Dedicated One-Hertz Task

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py`
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_ideal_surface_stomach_env_cfg.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py`
- Create: `tests/ideal_surface/test_ideal_surface_task_cfg.py`

**Interfaces:**
- Consumes: `IdealSurfaceController`, approved live surface mesh, capsule `RigidObject`, and the preflight-verified pose/velocity write methods.
- Produces: `IdealSurfaceActionTerm`, `IdealSurfaceActionTermCfg`, `ideal_surface_hard_failure(env)`, and Gym task `Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0`.

- [ ] **Step 1: Write the failing task-configuration tests**

```python
ENV_ID = "Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0"

def test_task_is_registered_at_one_hertz_with_one_scalar_action():
    spec = gym.spec(ENV_ID)
    assert "RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg" in (
        spec.kwargs["env_cfg_entry_point"]
    )
    cfg = RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg()
    assert cfg.scene.num_envs == 1
    assert cfg.sim.dt == 1.0 / 240.0
    assert cfg.decimation == 240
    assert cfg.scene.capsule_camera.update_period == 1.0
    assert set(vars(cfg.actions)) == {"ideal_surface"}

def test_task_does_not_enable_magnetic_capsule_forcing():
    cfg = RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg()
    assert "magnetic_physics" not in vars(cfg.actions)
    assert "magnetic_collision_bridge" not in vars(cfg.events)

def test_policy_observations_contain_no_capsule_or_surface_truth():
    cfg = RobotarmMagneticIdealSurfaceStomachTeleopLabEnvCfg()
    names = policy_term_names(cfg.observations.policy)
    assert not any(
        token in name.lower()
        for name in names
        for token in ("capsule", "surface", "contact", "coverage", "pose", "ray")
    )
```

- [ ] **Step 2: Run the configuration tests and verify the missing task failure**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_ideal_surface_task_cfg.py -q
```

- [ ] **Step 3: Implement the scalar Isaac ActionTerm**

```python
class IdealSurfaceActionTerm(ActionTerm):
    @property
    def action_dim(self) -> int:
        return 1

    def process_actions(self, actions: torch.Tensor) -> None:
        action_ids = torch.round(actions).to(torch.int64)
        if self._controller.ready:
            snapshot = self._snapshot_from_live_capsule()
            self._request_id += 1
            self._controller.submit(int(action_ids[0, 0]), snapshot, self._request_id)

    def apply_actions(self) -> None:
        output = self._controller.step(float(self._env.sim.cfg.dt))
        pose = torch.as_tensor(
            np.concatenate((output.position_world, output.quaternion_for_sim)),
            device=self._env.device,
            dtype=torch.float32,
        ).reshape(1, 7)
        velocity = torch.as_tensor(
            np.concatenate((output.linear_velocity_world, output.angular_velocity_world)),
            device=self._env.device,
            dtype=torch.float32,
        ).reshape(1, 6)
        self.capsule.write_root_pose_to_sim(pose)
        self.capsule.write_root_velocity_to_sim(velocity)

    def acknowledge_result(self) -> None:
        self._controller.acknowledge_result()
```

Use exactly the method names and quaternion ordering proven by Task 1; `quaternion_for_sim` is constructed in that verified ordering. Enable kinematic behavior only for the new task's capsule prim. The action term must verify one environment, load the approved surface, initialize the active triangle from the live reset pose, and expose `action_mask`, `last_result`, and `acknowledge_result` without exposing the raw geometry through observations.

- [ ] **Step 4: Implement the dedicated task configuration**

The new task must inherit the existing stomach scene and camera, replace the action group with only `ideal_surface`, replace the interval events with reset-only events, set `decimation=240`, keep `sim.dt=1/240`, and terminate only on timeout or `ideal_surface_hard_failure`. It must not change the existing atomic task.

- [ ] **Step 5: Run configuration, registration, and one-step smoke tests**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_ideal_surface_task_cfg.py -q
./run_isaaclab.sh -p -c "import gymnasium as gym; import robotarm_magnetic_lab.tasks; print(gym.spec('Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0'))"
./run_isaaclab.sh -p scripts/zero_agent.py \
  --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
  --num_envs 1 --max_steps 1 --viz kit
```

Expected: environment action-space shape `(1,)`, submitted batched tensor shape `(1, 1)`, one simulated second, no magnetic action term, finite capsule pose, and no termination.

- [ ] **Step 6: Commit the integration**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/__init__.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/robotarm_magnetic_ideal_surface_stomach_env_cfg.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/__init__.py
git add -f tests/ideal_surface/test_ideal_surface_task_cfg.py
git commit -m "feat: register ideal surface stomach task"
```

## Task 7: Add the Fifteen-Action Keyboard and Coverage Launcher

**Files:**
- Create: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/ideal_surface_keyboard.py`
- Modify: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py`
- Create: `scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py`
- Create: `tests/ideal_surface/test_ideal_surface_keyboard.py`

**Interfaces:**
- Consumes: existing `SessionController` and `P0CoverageRuntime`, new action term, and Kit keyboard events.
- Produces: `IdealSurfaceKeyboard.key_event(key, is_down)` and an interactive/manual launcher with coverage view and records.

- [ ] **Step 1: Write the complete key-map test**

```python
def test_fifteen_action_key_map_and_repeat_suppression():
    expected = {
        "SPACE": 0,
        "NUMPAD8": 1,
        "NUMPAD9": 2,
        "NUMPAD6": 3,
        "NUMPAD3": 4,
        "NUMPAD2": 5,
        "NUMPAD1": 6,
        "NUMPAD4": 7,
        "NUMPAD7": 8,
        "W": 9,
        "S": 10,
        "D": 11,
        "A": 12,
        "E": 13,
        "Q": 14,
    }
    keyboard = IdealSurfaceKeyboard()
    for key, action_id in expected.items():
        command = keyboard.key_event(key, True)
        assert command.action_id == action_id
        assert keyboard.key_event(key, True) is None
        assert keyboard.key_event(key, False) is None
```

The keyboard must normalize `NUMPAD_8`, `NUMPAD8`, `KP8`, and `8` to the same compass input, while preserving Backspace reset, F12 snapshot, and Escape exit.

- [ ] **Step 2: Run the test and verify the missing keyboard failure**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_ideal_surface_keyboard.py -q
```

- [ ] **Step 3: Implement the keyboard adapter without changing `atomic_keyboard.py`**

Reuse `CommandKind` and `KeyCommand` from the existing module. Preserve key-down edge handling and OS-repeat suppression.

- [ ] **Step 4: Implement the interactive launcher**

The launcher must make the new task, obtain `ideal_surface` from the action manager, pass its mask to `SessionController`, execute exactly one `env.step` per accepted action, acknowledge exactly one terminal result, call `P0CoverageRuntime.maybe_update()` once per new 1 Hz frame, and keep all capsule/surface truth inside the controller and evaluator objects.

```python
action_tensor = torch.full(
    env.action_space.shape,
    accepted_action_id,
    device=env.unwrapped.device,
    dtype=torch.float32,
)
_, _, terminated, truncated, _ = env.step(action_tensor)
result = term.last_result
completion = session.acknowledge(result.status.value, evaluator.sim_time_s)
evaluator.append_action_event(
    completion,
    "result",
    ideal_surface_result=result.to_dict(),
    schema_version="ideal_surface_v1",
)
evaluator.maybe_update()
term.acknowledge_result()
```

- [ ] **Step 5: Run pure keyboard tests and a rendered startup/exit smoke**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface/test_ideal_surface_keyboard.py -q
./run_isaaclab.sh -p scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py \
  --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
  --num_envs 1 --viz kit --max_idle_updates 2
```

Expected: the simulation and isolated coverage view start, display the fifteen-key help text, finalize evidence, and exit zero.

- [ ] **Step 6: Commit**

```bash
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/ideal_surface_keyboard.py
git add source/robotarm_magnetic_lab/robotarm_magnetic_lab/teleop/__init__.py
git add scripts/ideal_surface/teleop_ideal_surface_stomach_coverage.py
git add -f tests/ideal_surface/test_ideal_surface_keyboard.py
git commit -m "feat: add ideal surface keyboard coverage launcher"
```

## Task 8: Validate Geometry, All Actions, and Long-Sequence Stability

**Files:**
- Create: `scripts/ideal_surface/validate_ideal_surface_geometry.py`
- Create: `scripts/ideal_surface/validate_ideal_surface_stomach.py`
- Modify only if required by a demonstrated logging gap: `source/robotarm_magnetic_lab/robotarm_magnetic_lab/coverage/simulator_runtime.py`

**Interfaces:**
- Consumes: pure controller, live ideal-surface task, coverage runtime, and the preflight-approved initial pose.
- Produces: deterministic console summaries and external JSON/NumPy evidence for plane fixtures, all fifteen actions, and a 1,000-action valid random sequence.

- [ ] **Step 1: Implement the pure geometry validator**

```python
def validate_plane_controller() -> dict:
    checks = {
        "eight_unique_start_tilts": check_eight_start_axes(),
        "tilt_increment_deg": check_tilt_increment(expected=15.0, tolerance=0.2),
        "rise_increment_deg": check_rise_increment(expected=15.0, tolerance=0.2),
        "precession_increment_deg": check_precession(expected=15.0, tolerance=0.2),
        "roll_arc_length_m": check_roll(expected=0.004, tolerance=0.0001),
        "upright_residual_range_deg": check_upright_residuals(0.0, 5.0),
        "contact_limited_done": check_normal_saturation(),
        "boundary_limited_done": check_open_boundary(),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    return checks
```

- [ ] **Step 2: Run the pure geometry validator**

```bash
./run_isaaclab.sh -p scripts/ideal_surface/validate_ideal_surface_geometry.py
```

Expected: `IDEAL_SURFACE_GEOMETRY_PASS` with all checks true.

- [ ] **Step 3: Implement the live all-action scenario**

The validator must reset before each of the eight `START_TILT` directions, verify the final tilt and direction, then execute a valid sequence that reaches stable side contact and exercises `TILT_MORE`, `RISE`, both precession actions, both roll actions, and `HOLD`. It must compare action records, action mask, final pose, contact flags, and coverage frame IDs.

- [ ] **Step 4: Implement the deterministic 1,000-action valid random run**

At each boundary, sample uniformly only from the current mask using a recorded seed. Record position, quaternion, active triangle, component ID, maximum penetration, flags, result, and coverage. Fail immediately on nonfinite pose, nonadjacent component jump, unexpected `HARD_FAILURE`, duplicate request, duplicate frame update, or penetration above `0.05 r_eff`.

```python
rng = np.random.default_rng(seed)
for request_id in range(1, 1001):
    mask = np.asarray(term.action_mask(), dtype=bool)
    valid = np.flatnonzero(mask)
    action_id = int(rng.choice(valid))
    result = execute_one_boundary(action_id, request_id)
    assert result.status.value == "DONE"
    assert np.isfinite(result.final_position_world).all()
    assert result.maximum_penetration_m <= cfg.hard_penetration_radius_fraction * radius_m
```

- [ ] **Step 5: Run live validation**

```bash
./run_isaaclab.sh -p scripts/ideal_surface/validate_ideal_surface_stomach.py \
  --task Template-Robotarm-Magnetic-Ideal-Surface-Stomach-Teleop-Lab-v0 \
  --seed 42 --random_actions 1000 --output logs/ideal_surface_validation
```

Required results: every action ID observed, eight initial directions accepted within the upright tolerance, no unexplained hard failure, no nonadjacent jump, no hard penetration, one result per request, and one coverage update per unique 1 Hz frame. Final coverage is informational and has no pass threshold in this controller task.

- [ ] **Step 6: Commit validators**

```bash
git add scripts/ideal_surface/validate_ideal_surface_geometry.py
git add scripts/ideal_surface/validate_ideal_surface_stomach.py
git commit -m "test: validate ideal surface controller in stomach"
```

## Task 9: Run Regression, Document Operation, and Deliver Evidence

**Files:**
- Create: `docs/IDEAL_SURFACE_CONTROLLER.md`
- Create: `handoffs/reports/TASK-002-ideal-surface-controller-report.md`
- Modify: `.gitignore` only if newly generated ideal-surface artifacts are not already ignored.

**Interfaces:**
- Consumes: all task outputs and external artifact inventories.
- Produces: operator documentation, final Linux report, and a pushed unmerged feature branch.

- [ ] **Step 1: Write operator documentation**

Document the task ID, `ideal_surface_v1` action table, keypad compass, W/S tilt, D/A precession, E/Q roll, Space hold, upright hysteresis, side-contact requirement, normal saturation flags, reset/snapshot/exit keys, CLI commands, record locations, and the fact that this controller is a privileged simulation oracle rather than a real magnetic controller.

- [ ] **Step 2: Run all new pure and configuration tests**

```bash
./run_isaaclab.sh -p -m pytest tests/ideal_surface -q
```

Expected: every new test passes with zero failures.

- [ ] **Step 3: Re-run delivered P0 and stage-one regressions**

```bash
./run_isaaclab.sh -p -m pytest \
  tests/coverage \
  tests/action_layer/test_atomic_protocol.py \
  tests/action_layer/test_executor.py \
  tests/action_layer/test_safety.py \
  tests/action_layer/test_atomic_stomach_teleop_cfg.py \
  tests/action_layer/test_atomic_keyboard_protocol.py -q

./run_isaaclab.sh -p scripts/action_layer/validate_coverage_geometry.py --check all
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_stomach_teleop.py \
  --num_envs 1 --coverage_samples 5
./run_isaaclab.sh -p scripts/action_layer/validate_atomic_table.py \
  --num_envs 1 --max_steps_per_action 60 --viz kit
./run_isaaclab.sh -p scripts/zero_agent.py \
  --task Template-Robotarm-Magnetic-Table-Lab-v0 \
  --num_envs 1 --max_steps 5 --viz kit
```

Expected: no regression in the frozen eleven-action layer, P0 coverage geometry, stomach atomic task, table acceptance, or legacy 9D task.

- [ ] **Step 4: Run repository hygiene checks**

```bash
git diff --check
git status --short
python -m compileall \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/controllers/ideal_surface \
  source/robotarm_magnetic_lab/robotarm_magnetic_lab/tasks/manager_based/robotarm_magnetic_lab/mdp/ideal_surface_action.py
```

Expected: no whitespace errors, no generated artifacts staged, and no Python compilation errors.

- [ ] **Step 5: Write the required Linux report**

The report status must be one of `complete`, `partial`, `needs_decision`, or `blocked`. It must include planning/base commit, feature head, branch, every executed command and observed result, action-schema version, capsule dimensions, mesh hash, pose-write API, all deviations, unverified claims, and every external artifact path with byte size and SHA-256.

- [ ] **Step 6: Commit documentation and report**

```bash
git add docs/IDEAL_SURFACE_CONTROLLER.md
git add handoffs/reports/TASK-002-ideal-surface-controller-report.md
git commit -m "docs: report ideal surface controller acceptance"
```

- [ ] **Step 7: Push the feature branch without merging**

```bash
git push -u origin feature/TASK-002-ideal-surface-controller
```

Windows remains responsible for reviewing the implementation diff and evidence. Linux must not merge the branch into `main` or into the Windows planning branch.
