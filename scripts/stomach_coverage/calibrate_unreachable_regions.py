#!/usr/bin/env python3
"""Interactively author frozen geodesic unreachable regions on the stomach wall."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))

from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "configs/task009b/unreachable_region_v1.json",
)
parser.add_argument(
    "--log_root",
    type=Path,
    default=ROOT / "logs/task009b_unreachable_calibration",
)
parser.add_argument("--operator", default=os.environ.get("USER", "unknown"))
parser.add_argument("--initial_radius_mm", type=int, choices=tuple(range(10, 81, 5)), default=20)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this calibrator only accepts {TASK_ID}")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.ui
import omni.usd
from pxr import Gf, UsdGeom, Vt

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.entry_surface_region import nearest_surface_point
from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage
from robotarm_magnetic_lab.coverage.unreachable_region import (
    UNREACHABLE_RADII_M,
    UnreachableSeed,
    build_unreachable_mask,
    save_and_reload_unreachable,
    unreachable_region_record,
)
from robotarm_magnetic_lab.teleop.atomic_keyboard import normalize_key


CONTROLS = (
    "W/S=+Y/-Y A/D=-X/+X Q/E=+Z/-Z; Shift=fine; G=add seed; "
    "Tab=select seed; [ ]=radius; Backspace=delete; C=clear; F=freeze/save; Esc=exit"
)
FROZEN_SELECTION_REASON = "operator_confirmed_physical_or_anatomical_unreachable_surface"


def _tensor(value):
    return getattr(value, "torch", value)


class SessionLog:
    def __init__(self, root: Path) -> None:
        self.directory = Path(root) / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"

    def emit(self, event: str, **values) -> None:
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"TASK009B_UNREACHABLE_{event} {json.dumps(values, sort_keys=True)}", flush=True)


class Keyboard:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.presses: deque[str] = deque()
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(self._device, self._on_event)

    def _on_event(self, event, *_args):
        raw = event.input
        key = normalize_key(getattr(raw, "name", raw))
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key not in self.down:
                self.presses.append(key)
            self.down.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.down.discard(key)
        return True

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None


class Panel:
    def __init__(self) -> None:
        self.window = omni.ui.Window("TASK-009B Unreachable Surface Mask", width=900, height=245)
        with self.window.frame:
            with omni.ui.VStack(spacing=5):
                self.cursor = omni.ui.Label("", height=30)
                self.selection = omni.ui.Label("No seed selected.", height=42, word_wrap=True)
                self.summary = omni.ui.Label("No excluded surface.", height=42, word_wrap=True)
                omni.ui.Label(CONTROLS, height=55, word_wrap=True)
                self.status = omni.ui.Label(
                    "Only anatomical/physical unreachable surfaces may be excluded.",
                    height=45,
                    word_wrap=True,
                )

    def update(self, cursor: np.ndarray, seeds, selected, mask) -> None:
        self.cursor.text = f"Cursor world m: {np.round(cursor, 5).tolist()} (physics paused)"
        if selected is None:
            self.selection.text = f"Seeds: {len(seeds)}; selected: none"
        else:
            seed = seeds[selected]
            self.selection.text = (
                f"Seeds: {len(seeds)}; selected={selected + 1}; face={seed.triangle_index}; "
                f"radius={1000.0 * seed.radius_m:.0f} mm; point={np.round(seed.point_world_m, 5).tolist()}"
            )
        if mask is None:
            self.summary.text = "Excluded union: none"
        else:
            self.summary.text = (
                f"Excluded faces={len(mask.excluded_triangle_indices)} "
                f"area={mask.excluded_area_m2:.7f} m2 "
                f"fraction={100.0 * mask.excluded_area_fraction:.3f}% "
                f"reachable area={mask.reachable_area_m2:.7f} m2"
            )


class DebugGeometry:
    ROOT = "/World/TASK009BUnreachableCalibration"

    @staticmethod
    def _constant(gprim, name: str) -> None:
        primvar = UsdGeom.PrimvarsAPI(gprim.GetPrim()).GetPrimvar(name)
        if not primvar or not primvar.IsDefined():
            raise RuntimeError(f"missing USD primvar {gprim.GetPath()}.{name}")
        primvar.SetInterpolation(UsdGeom.Tokens.constant)

    def __init__(self) -> None:
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, self.ROOT)
        root.GetPrim().SetMetadata("comment", "visual-only unreachable surface calibration")
        self.surface = UsdGeom.Mesh.Define(stage, self.ROOT + "/ExcludedSurface")
        self.surface.CreatePointsAttr([])
        self.surface.CreateFaceVertexCountsAttr([])
        self.surface.CreateFaceVertexIndicesAttr([])
        self.surface.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.02, 0.02)])
        self.surface.CreateDisplayOpacityAttr([0.82])
        self._constant(self.surface, "displayColor")
        self._constant(self.surface, "displayOpacity")
        self.surface.CreateDoubleSidedAttr(True)
        self.cursor = UsdGeom.Points.Define(stage, self.ROOT + "/Cursor")
        self.cursor.CreatePointsAttr([])
        self.cursor.CreateWidthsAttr([0.008])
        self.cursor.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self.cursor.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.85, 0.0)])
        self._constant(self.cursor, "displayColor")
        self.seeds = UsdGeom.Points.Define(stage, self.ROOT + "/Seeds")
        self.seeds.CreatePointsAttr([])
        self.seeds.CreateWidthsAttr([0.006])
        self.seeds.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        self.seeds.CreateDisplayColorAttr([Gf.Vec3f(0.0, 1.0, 0.2)])
        self._constant(self.seeds, "displayColor")

    def update_cursor(self, point: np.ndarray) -> None:
        self.cursor.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(np.asarray(point, dtype=np.float32).reshape(1, 3))
        )

    def update(self, reference, mask, seeds) -> None:
        self.seeds.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(
                np.asarray([seed.point_world_m for seed in seeds], dtype=np.float32).reshape(-1, 3)
            )
        )
        if mask is None:
            triangle_points = np.empty((0, 3, 3), dtype=np.float64)
        else:
            triangle_points = reference.vertices_world[
                reference.triangles[mask.excluded_triangle_indices]
            ]
        if len(triangle_points):
            normals = np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            lengths = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
            offsets = 0.00035 * normals / lengths
            doubled = np.concatenate(
                [triangle_points + offsets[:, None, :], triangle_points - offsets[:, None, :]], axis=0
            )
            points = doubled.reshape(-1, 3)
        else:
            points = np.empty((0, 3), dtype=np.float64)
        self.surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        self.surface.GetFaceVertexCountsAttr().Set([3] * (len(points) // 3))
        self.surface.GetFaceVertexIndicesAttr().Set(list(range(len(points))))


def _left_bracket(key: str) -> bool:
    return key in ("[", "LEFT_BRACKET", "BRACKET_LEFT")


def _right_bracket(key: str) -> bool:
    return key in ("]", "RIGHT_BRACKET", "BRACKET_RIGHT")


def _escape(key: str) -> bool:
    return key in ("ESC", "ESCAPE")


def main() -> int:
    if args_cli.output.exists() and not args_cli.overwrite:
        raise FileExistsError(f"output already exists; inspect it or pass --overwrite: {args_cli.output}")
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    session = SessionLog(args_cli.log_root)
    env = keyboard = None
    session.emit(
        "STARTED",
        output=str(args_cli.output.resolve()),
        operator=args_cli.operator,
        reason=FROZEN_SELECTION_REASON,
        controls=CONTROLS,
    )
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            base = env.unwrapped
            base.sim.physics_manager.pause()
            paused_time = float(base.sim.physics_manager.get_simulation_time())
            reference = reference_from_stage()
            capsule_pose = _tensor(base.scene["capsule"].data.root_pose_w)[0]
            cursor = capsule_pose[:3].detach().cpu().numpy().astype(np.float64)
            seeds: list[UnreachableSeed] = []
            selected: int | None = None
            mask = None
            radius_index = UNREACHABLE_RADII_M.index(float(args_cli.initial_radius_mm) / 1000.0)
            keyboard = Keyboard()
            panel = Panel()
            geometry = DebugGeometry()
            geometry.update_cursor(cursor)
            geometry.update(reference, mask, seeds)
            panel.update(cursor, seeds, selected, mask)
            last_wall = time.monotonic()
            last_report = 0.0
            running = True
            while simulation_app.is_running() and running:
                base.sim.render()
                now = time.monotonic()
                dt = min(max(now - last_wall, 0.0), 0.05)
                last_wall = now
                speed = 0.002 if any("SHIFT" in key for key in keyboard.down) else 0.010
                direction = np.zeros(3, dtype=np.float64)
                direction[0] += float("D" in keyboard.down) - float("A" in keyboard.down)
                direction[1] += float("W" in keyboard.down) - float("S" in keyboard.down)
                direction[2] += float("Q" in keyboard.down) - float("E" in keyboard.down)
                if np.any(direction):
                    cursor += speed * dt * direction / max(float(np.linalg.norm(direction)), 1.0)
                    geometry.update_cursor(cursor)
                    panel.update(cursor, seeds, selected, mask)

                while keyboard.presses:
                    key = keyboard.presses.popleft()
                    if _escape(key):
                        session.emit("EXITED", status="needs_input", seed_count=len(seeds))
                        running = False
                        break
                    if key == "G":
                        closest = nearest_surface_point(reference, cursor)
                        seeds.append(
                            UnreachableSeed(
                                closest.triangle_index,
                                closest.point_world_m.copy(),
                                UNREACHABLE_RADII_M[radius_index],
                            )
                        )
                        selected = len(seeds) - 1
                        mask, _ = build_unreachable_mask(reference, seeds)
                        geometry.update(reference, mask, seeds)
                        panel.status.text = "Seed added. Red surface is excluded; inspect both sides."
                        session.emit(
                            "SEED_ADDED",
                            seed_index=selected,
                            triangle_index=closest.triangle_index,
                            surface_point_world_m=closest.point_world_m.tolist(),
                            cursor_to_surface_distance_m=closest.distance_m,
                            radius_m=UNREACHABLE_RADII_M[radius_index],
                        )
                    elif key == "TAB" and seeds:
                        selected = 0 if selected is None else (selected + 1) % len(seeds)
                        radius_index = UNREACHABLE_RADII_M.index(seeds[selected].radius_m)
                    elif (_left_bracket(key) or _right_bracket(key)) and selected is not None:
                        radius_index += 1 if _right_bracket(key) else -1
                        radius_index = min(max(radius_index, 0), len(UNREACHABLE_RADII_M) - 1)
                        old = seeds[selected]
                        seeds[selected] = UnreachableSeed(
                            old.triangle_index, old.point_world_m, UNREACHABLE_RADII_M[radius_index]
                        )
                        mask, _ = build_unreachable_mask(reference, seeds)
                        geometry.update(reference, mask, seeds)
                        session.emit(
                            "RADIUS_CHANGED",
                            seed_index=selected,
                            radius_m=UNREACHABLE_RADII_M[radius_index],
                        )
                    elif key == "BACKSPACE" and selected is not None:
                        removed = seeds.pop(selected)
                        selected = None if not seeds else min(selected, len(seeds) - 1)
                        mask = None if not seeds else build_unreachable_mask(reference, seeds)[0]
                        geometry.update(reference, mask, seeds)
                        session.emit("SEED_REMOVED", triangle_index=removed.triangle_index)
                    elif key == "C":
                        seeds.clear()
                        selected = None
                        mask = None
                        geometry.update(reference, mask, seeds)
                        session.emit("CLEARED")
                    elif key == "F":
                        if not seeds:
                            panel.status.text = "Cannot save: add at least one seed with G."
                            continue
                        record = unreachable_region_record(
                            reference=reference,
                            seeds=seeds,
                            reason=FROZEN_SELECTION_REASON,
                            operator=args_cli.operator,
                        )
                        save_and_reload_unreachable(args_cli.output, record)
                        session.emit(
                            "SAVED",
                            status="complete",
                            path=str(args_cli.output.resolve()),
                            config_sha256=record["config_sha256"],
                            stomach_geometry_sha256=record["stomach_geometry_sha256"],
                            seed_count=len(seeds),
                            excluded_triangle_count=record["excluded_triangle_count"],
                            excluded_area_m2=record["excluded_area_m2"],
                            excluded_area_fraction=record["excluded_area_fraction"],
                            reachable_area_m2=record["reachable_area_m2"],
                            reload_validated=True,
                        )
                        panel.status.text = "Frozen unreachable mask saved and reloaded."
                        running = False
                        break
                    panel.update(cursor, seeds, selected, mask)

                if float(base.sim.physics_manager.get_simulation_time()) != paused_time:
                    raise RuntimeError("physics time advanced during unreachable-region calibration")
                if now - last_report >= 1.0:
                    last_report = now
                    session.emit(
                        "STATE",
                        cursor_world_m=cursor.tolist(),
                        seed_count=len(seeds),
                        selected_seed=selected,
                        excluded_area_fraction=(None if mask is None else mask.excluded_area_fraction),
                        physics_paused=True,
                    )
            return 0
        finally:
            if keyboard is not None:
                keyboard.close()
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
