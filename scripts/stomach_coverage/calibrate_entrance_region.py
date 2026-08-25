#!/usr/bin/env python3
"""Interactively calibrate TASK-009B's world-axis-aligned stomach entrance box.

This gate intentionally does not step the environment or command the capsule.
It only reads the composed stomach surface, selects triangles by exact
triangle/AABB intersection, visualizes the selection, and saves a pending
operator-confirmation artifact.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source" / "robotarm_magnetic_lab"))
HEADLESS = "--headless" in sys.argv
if HEADLESS:
    sys.argv.remove("--headless")
    os.environ["HEADLESS"] = "1"

from isaaclab.app import AppLauncher


TASK_ID = "Template-Robotarm-Magnetic-Parameterized-Force-Stomach-Coverage-Lab-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID)
parser.add_argument("--center", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--size", nargs=3, type=float, default=None, metavar=("SX", "SY", "SZ"))
parser.add_argument("--translation_step_m", type=float, default=0.002)
parser.add_argument("--size_step_m", type=float, default=0.002)
parser.add_argument(
    "--smoke_updates",
    type=int,
    default=0,
    help="Exit after this many Kit updates; zero keeps the interactive session open.",
)
parser.add_argument(
    "--config_out",
    type=Path,
    default=ROOT / "configs" / "stomach_entrance_region_v1.json",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=[] if HEADLESS else ["kit"])
args_cli = parser.parse_args()
if args_cli.task != TASK_ID:
    parser.error(f"this calibrator only accepts {TASK_ID}")
if args_cli.translation_step_m <= 0.0 or args_cli.size_step_m <= 0.0:
    parser.error("calibration steps must be positive")
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import carb.input
import gymnasium as gym
import numpy as np
import omni.appwindow
import omni.ui
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt

import robotarm_magnetic_lab.tasks  # noqa: F401
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from robotarm_magnetic_lab.coverage.entrance_region import (
    entrance_region_record,
    save_entrance_region,
    select_entrance_triangles,
)
from robotarm_magnetic_lab.coverage.simulator_runtime import reference_from_stage
from robotarm_magnetic_lab.tasks.manager_based.robotarm_magnetic_lab.robotarm_magnetic_stomach_env_cfg import (
    STOMACH_ASSET_USD_PATH,
)


INSTRUCTIONS = (
    "X/Y/Z: select world axis | A/D: center -/+ | Q/E: size -/+ | "
    "1/2/3: step 0.5/2/5 mm | S: save | R: reset | Esc: exit"
)


class Keyboard:
    def __init__(self) -> None:
        self.events: deque[str] = deque()
        self._input = carb.input.acquire_input_interface()
        self._device = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._device, self._on_event
        )

    def _on_event(self, event, *_args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.events.append(str(event.input.name).upper())
        return True

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._device, self._subscription)
            self._subscription = None


class CalibrationPanel:
    def __init__(self) -> None:
        self.window = omni.ui.Window("TASK-009B Entrance Calibration", width=760, height=205)
        with self.window.frame:
            with omni.ui.VStack(spacing=5):
                omni.ui.Label("TASK-009B stomach entrance AABB", height=28)
                self.parameters = omni.ui.Label("", height=42, word_wrap=True)
                self.selection = omni.ui.Label("", height=42, word_wrap=True)
                omni.ui.Label(INSTRUCTIONS, height=48, word_wrap=True)
                self.status = omni.ui.Label(
                    "Not saved. Saving does not confirm Gate 2; reply to Codex after visual inspection.",
                    height=32,
                    word_wrap=True,
                )

    def update(self, center, size, axis, translation_step, size_step, selection) -> None:
        self.parameters.text = (
            f"axis={axis} center_m={np.round(center, 6).tolist()} "
            f"size_m={np.round(size, 6).tolist()} "
            f"move_step={translation_step:.4f} size_step={size_step:.4f}"
        )
        self.selection.text = (
            f"triangles={selection.triangle_count} area={selection.area_m2:.9f} m^2 "
            f"connected_components={selection.connected_components}"
        )
        if selection.connected_components > 1:
            self.status.text = "WARNING: multiple disconnected stomach regions selected; adjust the box."

    def saved(self, path: Path, config_hash: str) -> None:
        self.status.text = (
            f"Saved pending-confirmation config: {path} hash={config_hash[:16]}... "
            "Reply to Codex only after checking the highlighted surface."
        )


class EntranceDebugGeometry:
    ROOT = "/World/TASK009BEntranceCalibration"

    def __init__(self, vertices_world: np.ndarray) -> None:
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, self.ROOT)
        root.GetPrim().SetMetadata(
            "comment", "TASK-009B visual calibration only; no physics schemas are applied"
        )
        self.vertices = np.asarray(vertices_world, dtype=np.float64)
        self.box = UsdGeom.Mesh.Define(stage, self.ROOT + "/Box")
        self.box.CreateFaceVertexCountsAttr([4] * 6)
        self.box.CreateFaceVertexIndicesAttr(
            [
                0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1,
                1, 5, 6, 2, 2, 6, 7, 3, 4, 0, 3, 7,
            ]
        )
        self.box.CreateDisplayColorAttr([Gf.Vec3f(0.0, 0.65, 1.0)])
        self.box.CreateDisplayOpacityAttr([0.18])
        self.box.SetDisplayColorInterpolation(UsdGeom.Tokens.constant)
        self.box.SetDisplayOpacityInterpolation(UsdGeom.Tokens.constant)
        self.box.CreateDoubleSidedAttr(True)

        self.highlight = UsdGeom.Mesh.Define(stage, self.ROOT + "/SelectedTriangles")
        self.highlight.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.16, 0.0)])
        self.highlight.CreateDisplayOpacityAttr([0.82])
        self.highlight.SetDisplayColorInterpolation(UsdGeom.Tokens.constant)
        self.highlight.SetDisplayOpacityInterpolation(UsdGeom.Tokens.constant)
        self.highlight.CreateDoubleSidedAttr(True)

        self.axes = UsdGeom.BasisCurves.Define(stage, self.ROOT + "/WorldAxes")
        self.axes.CreateTypeAttr(UsdGeom.Tokens.linear)
        self.axes.CreateCurveVertexCountsAttr([2, 2, 2])
        self.axes.CreateDisplayColorAttr(
            [Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0.45, 1)]
        )
        self.axes.SetDisplayColorInterpolation(UsdGeom.Tokens.uniform)
        self.axes.CreateWidthsAttr([0.0015])
        self.axes.SetWidthsInterpolation(UsdGeom.Tokens.constant)

    @staticmethod
    def _box_points(center: np.ndarray, size: np.ndarray) -> np.ndarray:
        half = 0.5 * size
        signs = np.asarray(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        return center + signs * half

    def update(self, center: np.ndarray, size: np.ndarray, reference, selection) -> None:
        self.box.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(self._box_points(center, size).astype(np.float32))
        )
        selected_faces = reference.triangles[selection.triangle_indices]
        triangle_points = reference.vertices_world[selected_faces]
        if len(triangle_points):
            normals = np.cross(
                triangle_points[:, 1] - triangle_points[:, 0],
                triangle_points[:, 2] - triangle_points[:, 0],
            )
            normal_lengths = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-15)
            offsets = 0.00035 * normals / normal_lengths
            # Draw a visual-only copy on each side of the thin stomach shell,
            # avoiding depth fighting regardless of authored winding.
            selected_points = np.concatenate(
                [triangle_points + offsets[:, None, :], triangle_points - offsets[:, None, :]],
                axis=0,
            ).reshape(-1, 3)
        else:
            selected_points = np.empty((0, 3), dtype=np.float64)
        self.highlight.GetPointsAttr().Set(
            Vt.Vec3fArray.FromNumpy(selected_points.astype(np.float32))
        )
        self.highlight.GetFaceVertexCountsAttr().Set([3] * (2 * len(selected_faces)))
        self.highlight.GetFaceVertexIndicesAttr().Set(list(range(len(selected_points))))
        extent = max(float(np.max(size)), 0.02) * 0.8
        axis_points = np.asarray(
            [
                center, center + [extent, 0, 0],
                center, center + [0, extent, 0],
                center, center + [0, 0, extent],
            ],
            dtype=np.float32,
        )
        self.axes.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(axis_points))


def _default_box(reference) -> tuple[np.ndarray, np.ndarray]:
    lower = reference.vertices_world.min(axis=0)
    upper = reference.vertices_world.max(axis=0)
    span = upper - lower
    size = np.maximum(0.20 * span, np.asarray([0.025, 0.025, 0.025]))
    center = 0.5 * (lower + upper)
    center[1] = upper[1] - 0.5 * size[1]
    return center, size


def _summary(center, size, axis, move_step, size_step, selection) -> str:
    warning = " WARNING=DISCONNECTED" if selection.connected_components > 1 else ""
    return (
        "TASK009B_ENTRANCE "
        f"axis={'XYZ'[axis]} center_m={np.round(center, 6).tolist()} "
        f"size_m={np.round(size, 6).tolist()} move_step_m={move_step:.6f} "
        f"size_step_m={size_step:.6f} triangles={selection.triangle_count} "
        f"area_m2={selection.area_m2:.9f} components={selection.connected_components}{warning}"
    )


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cpu", num_envs=1, use_fabric=True)
    cfg.sim.device = "cpu"
    env = keyboard = panel = None
    with launch_simulation(cfg, args_cli):
        try:
            env = gym.make(args_cli.task, cfg=cfg)
            env.reset()
            reference = reference_from_stage()
            default_center, default_size = _default_box(reference)
            center = np.asarray(args_cli.center or default_center, dtype=np.float64)
            size = np.asarray(args_cli.size or default_size, dtype=np.float64)
            initial_center = center.copy()
            initial_size = size.copy()
            axis = 0
            move_step = float(args_cli.translation_step_m)
            size_step = float(args_cli.size_step_m)
            keyboard = Keyboard()
            panel = CalibrationPanel()
            geometry = EntranceDebugGeometry(reference.vertices_world)
            selection = select_entrance_triangles(reference, center, size)

            def refresh() -> None:
                nonlocal selection
                selection = select_entrance_triangles(reference, center, size)
                geometry.update(center, size, reference, selection)
                panel.update(center, size, "XYZ"[axis], move_step, size_step, selection)
                print(_summary(center, size, axis, move_step, size_step, selection), flush=True)

            refresh()
            print("TASK009B_ENTRANCE_CONTROLS " + INSTRUCTIONS, flush=True)
            update_count = 0
            while simulation_app.is_running():
                simulation_app.update()
                update_count += 1
                if args_cli.smoke_updates > 0 and update_count >= args_cli.smoke_updates:
                    print(
                        f"TASK009B_ENTRANCE_SMOKE_PASS updates={update_count} "
                        f"triangles={selection.triangle_count} components={selection.connected_components}",
                        flush=True,
                    )
                    return 0
                changed = False
                while keyboard.events:
                    key = keyboard.events.popleft()
                    if key in ("X", "Y", "Z"):
                        axis = "XYZ".index(key)
                        changed = True
                    elif key == "A":
                        center[axis] -= move_step
                        changed = True
                    elif key == "D":
                        center[axis] += move_step
                        changed = True
                    elif key == "Q":
                        size[axis] = max(0.0005, size[axis] - size_step)
                        changed = True
                    elif key == "E":
                        size[axis] += size_step
                        changed = True
                    elif key in ("1", "2", "3"):
                        move_step = size_step = {"1": 0.0005, "2": 0.002, "3": 0.005}[key]
                        changed = True
                    elif key == "R":
                        center[:] = initial_center
                        size[:] = initial_size
                        changed = True
                    elif key == "S":
                        record = entrance_region_record(
                            reference,
                            center,
                            size,
                            selection,
                            stomach_asset_identifier=STOMACH_ASSET_USD_PATH,
                            operator_confirmation="pending",
                        )
                        save_entrance_region(args_cli.config_out, record)
                        save_line = (
                            "TASK009B_ENTRANCE_SAVED "
                            f"path={args_cli.config_out.resolve()} "
                            f"asset={record['stomach_asset_identifier']} "
                            f"geometry_sha256={record['stomach_geometry_sha256']} "
                            f"center_m={record['center_world_m']} size_m={record['size_world_m']} "
                            f"triangles={record['selected_triangle_count']} "
                            f"area_m2={record['selected_area_m2']:.9f} "
                            f"components={record['connected_component_count']} "
                            f"config_sha256={record['config_sha256']} confirmation=pending"
                        )
                        args_cli.config_out.with_suffix(".save_summary.txt").write_text(
                            save_line + "\n", encoding="utf-8"
                        )
                        print(save_line, flush=True)
                        panel.saved(args_cli.config_out.resolve(), record["config_sha256"])
                    elif key in ("ESCAPE", "ESC"):
                        return 0
                if changed:
                    refresh()
        finally:
            if keyboard is not None:
                keyboard.close()
            if env is not None:
                env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
