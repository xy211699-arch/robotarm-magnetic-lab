# Friction and compliant-contact calibration

This workflow calibrates the passive capsule/stomach contact model without
editing the validated base USD assets.

## What is calibrated

- Both capsule and stomach static/dynamic friction coefficients.
- Capsule-side PhysX compliant-contact stiffness and damping.
- Restitution stays at zero.

Each candidate runs in a fresh Isaac process through temporary USDA override
layers. This avoids stale PhysX material state and preserves the base scene.

## 1. Run a numerical baseline

```bash
cd /mnt/isaac-linux/robotarm_magnetic_lab
python3 scripts/calibration/contact/run_sweep.py \
  --scenario both \
  --max_candidates 1
```

The complete built-in five-point sensitivity sweep is:

```bash
python3 scripts/calibration/contact/run_sweep.py --scenario both
```

Results are written below `logs/contact_calibration/<timestamp>/`. These logs
show numerical stability and parameter sensitivity; they are **not** physical
calibration by themselves.

## 2. Measure the physical system

Use the same 5, 15 and 30 mm drop heights on a representative wet-tissue
phantom. Record peak normal force and impact time with a force sensor/high-speed
camera. For friction, use measured incline angles or reproduce the three
sliding regions and record planar displacement over seven seconds.

For a conventional incline coupon:

```text
mu_static ~= tan(theta_at_first_slip)
mu_dynamic = (g*sin(theta) - acceleration) / (g*cos(theta))
```

Enter measured means and an error scale/standard deviation in
`configs/calibration/contact_targets.json`. Use at least five repeats per
condition. Tissue batch, wetting, capsule shell finish and temperature must be
recorded because they materially affect friction.

## 3. Fit the sweep

```bash
python3 scripts/calibration/contact/fit_parameters.py \
  logs/contact_calibration/<timestamp>/sweep_index.json
```

The fitter minimizes normalized RMSE across every populated target and writes
`fit_result.json`. Blank targets deliberately stop fitting instead of inventing
a “calibrated” answer.

## Acceptance

- All base contact checks still pass.
- Candidate ordering is repeatable across at least three simulation runs.
- The chosen candidate predicts held-out physical repeats within their stated
  measurement uncertainty.
- No tunnelling, capsule launch or non-finite state occurs at 240 Hz physics.

This remains a static rigid triangle-mesh stomach model. The fitted compliant
contact approximates local normal softness but does not reproduce distributed
tissue deformation, viscoelasticity, folds moving under load, or fluid films.
