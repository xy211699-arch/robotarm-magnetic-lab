"""Isaac Lab COM-wrench adapter for TASK-005's eleven discrete actions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..controllers.eleven_action import (
    ActionTelemetry,
    CapsuleState,
    ElevenActionController,
    ElevenActionId,
    FlatSurfaceQuery,
    Lifecycle,
    StomachSurfaceQuery,
    dynamic_profile_sha256,
    load_dynamic_profile,
)
from ..controllers.eleven_action.contact_history import ContactSample
from ..controllers.eleven_action.geometry import capsule_axis_world


@dataclass(frozen=True)
class RawContactRecord:
    position_world: np.ndarray
    normal_world: np.ndarray
    impulse_n_s: float | None


class ElevenActionRequestDecoder:
    """Decode one scalar; -1 means this environment step has no new request."""

    def decode(self, action) -> ElevenActionId | None:
        value = float(np.asarray(action, dtype=np.float64).reshape(1)[0])
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("eleven-action request must be a finite integer")
        code = int(value)
        if code == -1:
            return None
        if not 0 <= code <= 10:
            raise ValueError("eleven-action request must be -1 or an ID from 0 through 10")
        return ElevenActionId(code)


class RequestGate:
    """One-slot immediate gate: executing requests are discarded, never queued."""

    def __init__(self) -> None:
        self._pending: ElevenActionId | None = None
        self.discarded_request_count = 0

    def offer(self, action: ElevenActionId, *, ready: bool) -> ElevenActionId | None:
        if not ready or self._pending is not None:
            self.discarded_request_count += 1
            return None
        self._pending = ElevenActionId(int(action))
        return self._pending

    def take(self) -> ElevenActionId | None:
        result, self._pending = self._pending, None
        return result

    def reset(self) -> None:
        self._pending = None
        self.discarded_request_count = 0


def contact_records_for_capsule(
    headers,
    contact_data,
    *,
    capsule_prim_path: str,
    path_resolver: Callable[[int], str],
) -> tuple[RawContactRecord, ...]:
    """Copy all PhysX contact points whose collider belongs to the capsule."""
    records: list[RawContactRecord] = []
    root = str(capsule_prim_path).rstrip("/")
    for header in headers:
        collider0 = str(path_resolver(int(header.collider0)))
        collider1 = str(path_resolver(int(header.collider1)))
        if not (collider0 == root or collider0.startswith(root + "/") or collider1 == root or collider1.startswith(root + "/")):
            continue
        offset = int(header.contact_data_offset)
        for contact in contact_data[offset : offset + int(header.num_contact_data)]:
            impulse_vector = np.asarray(contact.impulse, dtype=np.float64).reshape(3)
            records.append(
                RawContactRecord(
                    np.asarray(contact.position, dtype=np.float64).reshape(3).copy(),
                    np.asarray(contact.normal, dtype=np.float64).reshape(3).copy(),
                    float(np.linalg.norm(impulse_vector)),
                )
            )
    return tuple(records)


class ElevenActionTerm(ActionTerm):
    """Read real dynamics and apply only copied world-frame COM wrench tensors."""

    cfg: "ElevenActionTermCfg"

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("ElevenActionTerm requires num_envs=1")
        self.capsule = env.scene[cfg.asset_name]
        self._physics_dt_s = float(env.physics_dt)
        if not math.isclose(self._physics_dt_s, 1.0 / 240.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("ElevenActionTerm requires exactly 1/240 s physics dt")
        self.profile = load_dynamic_profile()
        self._verify_dynamic_invariants(env)
        mass = float(self.capsule.data.body_mass.torch[0, 0].item())
        if not math.isfinite(mass) or abs(mass - self.profile.capsule_mass_kg) > 1.0e-6:
            raise RuntimeError("live capsule mass does not match the TASK-005 profile")
        self.controller = ElevenActionController(self.profile, self._make_surface_query(cfg))
        self.decoder = ElevenActionRequestDecoder()
        self.request_gate = RequestGate()
        self._raw_actions = torch.full((1, 1), -1.0, device=env.device)
        self._processed_actions = torch.full_like(self._raw_actions, -1.0)
        self._force_world = torch.zeros((1, 3), device=env.device)
        self._torque_world = torch.zeros((1, 3), device=env.device)
        self._telemetry: ActionTelemetry | None = None
        self._substep_telemetry: deque[ActionTelemetry] = deque(maxlen=32768)
        self._pending_contact_records: deque[RawContactRecord] = deque()
        self._physics_substep = 0
        self._capsule_prim_path = str(self.capsule.root_view.prim_paths[0])
        self._contact_subscription = self._subscribe_contact_reports()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def ready(self) -> bool:
        return self.controller.ready

    @property
    def last_result(self):
        return self.controller.last_result

    @property
    def telemetry(self) -> ActionTelemetry | None:
        return self._telemetry

    @property
    def substep_telemetry(self) -> tuple[ActionTelemetry, ...]:
        return tuple(self._substep_telemetry)

    @property
    def profile_sha256(self) -> str:
        return dynamic_profile_sha256()

    @property
    def discarded_request_count(self) -> int:
        return self.request_gate.discarded_request_count + self.controller.discarded_request_count

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        action = self.decoder.decode(actions[0].detach().cpu().numpy())
        self._processed_actions.fill_(-1.0 if action is None else float(int(action)))
        if action is not None:
            self.request_gate.offer(action, ready=self.controller.ready)

    def apply_actions(self) -> None:
        state = self._read_state()
        self._drain_contacts(state)
        pending = self.request_gate.take()
        if pending is not None:
            self.controller.submit(pending, state, physics_substep=self._physics_substep)
        output = self.controller.step(state, physics_substep=self._physics_substep)
        self._telemetry = output.telemetry
        self._substep_telemetry.append(output.telemetry)
        # Explicit copies prevent either NumPy or PhysX from retaining mutable views.
        force = output.wrench.force_world_n.copy()
        torque = output.wrench.torque_world_nm.copy()
        self._force_world[0] = torch.as_tensor(force, device=self._env.device, dtype=self._force_world.dtype)
        self._torque_world[0] = torch.as_tensor(torque, device=self._env.device, dtype=self._torque_world.dtype)
        self.capsule.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._force_world[:, None, :],
            torques=self._torque_world[:, None, :],
            positions=None,
            body_ids=None,
            env_ids=None,
            is_global=True,
        )
        self._physics_substep += 1

    def reset(self, env_ids=None) -> None:
        self._raw_actions.fill_(-1.0)
        self._processed_actions.fill_(-1.0)
        self._force_world.zero_()
        self._torque_world.zero_()
        self._telemetry = None
        self._substep_telemetry.clear()
        self._pending_contact_records.clear()
        self._physics_substep = 0
        self.request_gate.reset()
        self.controller.reset()
        self.capsule.permanent_wrench_composer.reset(env_ids=env_ids)

    def _read_state(self) -> CapsuleState:
        com_pose = self.capsule.data.root_com_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        link_pose = self.capsule.data.root_link_pose_w.torch[0].detach().cpu().numpy().astype(np.float64)
        velocity = self.capsule.data.root_com_vel_w.torch[0].detach().cpu().numpy().astype(np.float64)
        quaternion_xyzw = link_pose[3:7]
        return CapsuleState(com_pose[:3], quaternion_xyzw[[3, 0, 1, 2]], velocity[:3], velocity[3:6])

    def _drain_contacts(self, state: CapsuleState) -> None:
        axis = capsule_axis_world(state)
        while self._pending_contact_records:
            record = self._pending_contact_records.popleft()
            hit = self.controller.surface_query.query(record.position_world)
            sigma = float((record.position_world - state.position_world_m) @ axis)
            self.controller.observe_contact(
                ContactSample(
                    physics_substep=self._physics_substep,
                    point_world=record.position_world,
                    normal_world=hit.normal_world,
                    axial_coordinate_m=sigma,
                    impulse_n_s=record.impulse_n_s,
                    cylinder_half_length_m=self.profile.capsule_cylinder_half_length_m,
                )
            )

    def _subscribe_contact_reports(self):
        from omni.physics.core import get_physics_simulation_interface
        from pxr import PhysicsSchemaTools

        def resolve(path_id: int) -> str:
            return str(PhysicsSchemaTools.intToSdfPath(path_id))

        def callback(headers, contact_data, _friction_anchors) -> None:
            records = contact_records_for_capsule(
                headers,
                contact_data,
                capsule_prim_path=self._capsule_prim_path,
                path_resolver=resolve,
            )
            self._pending_contact_records.extend(records)

        return get_physics_simulation_interface().subscribe_physics_contact_report_events(callback)

    def _make_surface_query(self, cfg):
        if cfg.surface_kind == "flat":
            return FlatSurfaceQuery.regular_plane(
                half_extent_m=cfg.flat_half_extent_m,
                cells_per_side=cfg.flat_cells_per_side,
            )
        if cfg.surface_kind != "stomach":
            raise ValueError("surface_kind must be 'flat' or 'stomach'")
        from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage

        reference = reference_from_stage(cfg.surface_prim_path)
        if reference.geometry_sha256 != cfg.expected_surface_geometry_sha256:
            raise RuntimeError("approved stomach surface geometry digest changed")
        return StomachSurfaceQuery.from_reference(reference, inward_sign=cfg.inward_normal_sign)

    def _verify_dynamic_invariants(self, env) -> None:
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.capsule.root_view.prim_paths[0])
        rigid = UsdPhysics.RigidBodyAPI(prim)
        physx = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not prim.IsValid() or not rigid:
            raise RuntimeError("capsule non-kinematic rigid body is unavailable")
        if rigid.GetKinematicEnabledAttr() and bool(rigid.GetKinematicEnabledAttr().Get()):
            raise RuntimeError("TASK-005 capsule must remain non-kinematic")
        if physx.GetDisableGravityAttr() and bool(physx.GetDisableGravityAttr().Get()):
            raise RuntimeError("TASK-005 capsule gravity must remain enabled")
        # The source USD intentionally remains unchanged.  Reuse TASK-003/004's
        # accepted task-local session authoring so this isolated dynamic task
        # has body CCD whenever the selected PhysX backend supports it.
        ccd = physx.GetEnableCCDAttr()
        if not ccd:
            ccd = physx.CreateEnableCCDAttr()
        ccd.Set(True)
        if not bool(ccd.Get()):
            raise RuntimeError("TASK-005 capsule body CCD could not be enabled")
        if not bool(getattr(env.cfg.sim.physics, "enable_ccd", False)):
            raise RuntimeError("TASK-005 scene CCD must already be enabled")


@configclass
class ElevenActionTermCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ElevenActionTerm
    asset_name: str = "capsule"
    surface_kind: str = "flat"
    flat_half_extent_m: float = 1.0
    flat_cells_per_side: int = 8
    surface_prim_path: str = (
        "/World/envs/env_0/Stomach/ConvertedSource/Environment/Stomach/VisualMesh/Stomach"
    )
    expected_surface_geometry_sha256: str = (
        "85ddd3e79438509364245c87be9a9564d1bf9ca29afb2c922fc013b2f7561d09"
    )
    inward_normal_sign: int = -1


def eleven_action_fault(env, term_name: str = "eleven_action") -> torch.Tensor:
    term = env.action_manager.get_term(term_name)
    failed = term.controller.lifecycle is Lifecycle.FAULTED
    return torch.full((env.num_envs,), failed, dtype=torch.bool, device=env.device)
