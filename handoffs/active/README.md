# Active task contracts

The currently authorized Linux implementation task is
`TASK-009D0-vectorized-training-infrastructure.md`.

Linux must fetch the Windows planning branch
`workflow/TASK-009D0-vectorized-training-infrastructure`, record its exact HEAD, confirm
`7c4c5a18780b980ad3882ce75f1d64733fc3080d` and
`f8eb6b825aa8e5765b3db52532b169a9d299066e` are ancestors, create
`feature/TASK-009D0-vectorized-training-infrastructure` from the planning HEAD, and execute
`docs/superpowers/plans/2026-08-28-task009d0-vectorized-training-infrastructure.md` in task and Gate order.

TASK-009D0 adds a separate non-destructive multi-environment task, exact batched GPU coverage,
synchronous 120-second episodes, isolation tests, throughput selection, and long-soak evidence.
The accepted TASK-009C single-environment task remains the regression reference.

TASK-009D0 does not authorize CNN, GRU, Actor/Critic models, PPO, VLM, reward shaping,
disturbance ranges, controller recalibration, ROI changes, pose-library regeneration, or USD edits.

Linux execution status (2026-08-28): `partial`. Gate 1 and Gate 2 passed; Gate 3 failed the frozen
two-environment 1 micrometre local-trajectory tolerance after the required HOLD stabilization.
The user explicitly accepted this environment difference and authorized Gates 4--6 to continue as
a manual waiver; Gate 3 remains failed and its threshold is unchanged. See
`handoffs/reports/TASK-009D0-vectorized-training-infrastructure-report.md`.
