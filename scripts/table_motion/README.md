# Flat-table magnetic motion acceptance

This directory validates the simplified, Z-up table benchmark. The capsule is
always a passive rigid body. Robot-arm and Ball commands create the magnetic
field; capsule pose is recorded only for offline acceptance.

Run a visual test from the project root:

```bash
./run_isaaclab.sh -p scripts/table_motion/test_01_baseline.py
./run_isaaclab.sh -p scripts/table_motion/test_02_axial_field_scan.py
./run_isaaclab.sh -p scripts/table_motion/test_03_tilt_azimuth.py
./run_isaaclab.sh -p scripts/table_motion/test_04_upright_to_side.py
./run_isaaclab.sh -p scripts/table_motion/test_05_long_axis_roll.py
./run_isaaclab.sh -p scripts/table_motion/test_06_composite_motion.py
```

Optional switches:

```bash
--capsule_camera_view
--contact_debug
--visualizer none --no-realtime
```

Each run writes `telemetry.jsonl` and `summary.json` below
`logs/table_motion/<scenario>/<timestamp>/`.

Test 06 is the continuous compound sequence: initial side rest, side-to-upright,
30-degree tilt and one full azimuth revolution, return through upright to side,
then passive rolling approximately 100 mm along world +X. It does not reset the
capsule between phases. The composite-only Ball action envelope spans +/-pi so
the finite-field inverse can cover both magnetic hemispheres. During the final
roll, j6 remains at its collision-selected reset angle while j1..j5 produce the
field-gradient translation.

The axial internal magnet can align the capsule long axis through
`torque = m × B`, but a uniform field cannot directly create torque about that
axis. Test 05 therefore uses a small robot-arm translation to create lateral
field-gradient force. Table friction converts translation into passive rolling
while Ball maintains the axial field.

Final nominal bench values are static/dynamic friction `0.55/0.48`, capsule
linear/angular damping `0.10/0.50`, zero gastric-fluid angular drag, and a
`0.040 N` force safety cap used only by the long-axis-roll scenario. These are
simulation bring-up values and must be replaced by measured parameters before
quantitative sim-to-real claims.

See `docs/TABLE_MOTION_ACCEPTANCE.md` for acceptance metrics and visual checks.
