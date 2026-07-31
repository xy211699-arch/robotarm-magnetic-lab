# Stomach contact-physics acceptance tests

These tests use the stomach-only Isaac Lab task and the real passive capsule.
Each entry point opens the Kit viewport by default, writes per-step contact
telemetry, and produces an objective PASS/FAIL summary. ContactSensor markers
are hidden by default because Isaac Lab's standard red marker has a 20 mm
radius and can completely hide the real 13 mm diameter capsule. Add
`--contact_debug` to show a reduced 2 mm marker.

Run from the project root:

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
./run_isaaclab.sh -p scripts/contact_physics/test_01_no_magnet_resting.py
./run_isaaclab.sh -p scripts/contact_physics/test_02_drop_heights.py
./run_isaaclab.sh -p scripts/contact_physics/test_03_incline_slide.py
./run_isaaclab.sh -p scripts/contact_physics/test_04_magnetic_attraction.py
./run_isaaclab.sh -p scripts/contact_physics/test_05_wall_roll_turn.py
./run_isaaclab.sh -p scripts/contact_physics/test_06_ball_pose_contact.py
./run_isaaclab.sh -p scripts/contact_physics/test_07_multi_start.py
```

Add `--capsule_camera_view` to inspect a circular 30 Hz engineering preview.
In the stomach task this preview does not change the recorded/policy camera,
which remains 1 Hz and is the only camera included in observations. Add
`--visualizer none --no-realtime` for automated execution. This Isaac Lab 3.0
launcher does not expose a `--headless` flag. The project launcher adds the
required Kit geometry-streaming workaround automatically.

Results are written to:

```text
logs/contact_physics/<scenario>/<timestamp>/
├── telemetry.jsonl
└── summary.json
```

Important visualization cues:

- the capsule is passive; only the three Ball joints are commanded in magnetic
  scenarios;
- the visible capsule is the actual 13 mm diameter, 25 mm long dynamic rigid
  body; no test replaces it with a sphere;
- `--contact_debug` adds a small red contact marker. The marker is visual only:
  it has no mass, collision, force or measurement role;
- when `--contact_debug` is enabled, the small red marker should appear at
  impact/wall support;
- no-magnet tests must report exactly zero magnetic wrench;
- the capsule must not cross the stomach wall, launch, or acquire non-finite
  velocity;
- the final console line is `CONTACT_TEST_RESULT ... PASS|FAIL`.

`friction_estimate_N` is reconstructed from `m*a - gravity - magnetic -
normal_contact` at the 20 Hz environment rate. It is a diagnostic estimate,
not a replacement for a force/torque transducer calibration.

## Scenario implementation

| Script | Magnetic field | Ball command | Physical procedure |
|---|---:|---:|---|
| `test_01_no_magnet_resting.py` | Off | None | Reset the real capsule on the lower stomach wall and observe support, drift and settling for 8 s. |
| `test_02_drop_heights.py` | Off | None | Release the capsule from 5, 15 and 30 mm above the same wall region; check impact, settling and tunneling. |
| `test_03_incline_slide.py` | Off | None | Release it 20 mm above each of three offset stomach regions and allow gravity/contact/friction alone to produce sliding. |
| `test_04_magnetic_attraction.py` | Off reference, then on | Ball held at reset pose | Compare a one-step gravity reference with a 50 mm magnetic-attraction release and verify wall capture. |
| `test_05_wall_roll_turn.py` | On | Smooth three-axis sequence | Tilt the external magnet, then vary roll/yaw components so magnetic force and torque passively roll and turn the capsule along the wall. |
| `test_06_ball_pose_contact.py` | On | Three smooth absolute poses | Move through three Ball poses while checking that the passive capsule responds without losing wall contact. |
| `test_07_multi_start.py` | Off | None | Repeat free contact from five nearby stomach points and check finite, bounded, retained contact. |

The Ball inputs are normalized **absolute joint-position offsets**, not angular
velocities and not incremental commands. Action elements 6–8 map to
`ballxj/ballyj/ballzj`; a value of `1.0` means `+pi/2 rad` from that joint's
reset angle and `-1.0` means `-pi/2 rad`. Smoothstep/sine profiles prevent
instantaneous target jumps. These actuators rotate the robot-mounted magnet.
The capsule is never actively rotated: its orientation changes only through
the analytical magnetic torque, gravity, stomach contact, friction and passive
linear/angular drag.
