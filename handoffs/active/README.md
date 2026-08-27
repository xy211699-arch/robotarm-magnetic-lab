# Active task contracts

The currently authorized Linux implementation task is `TASK-009C-synchronous-random-baseline-preexperiment.md`.

Linux must fetch the Windows planning branch `workflow/TASK-009C-synchronous-random-baselines`, record its exact head, create `feature/TASK-009C-synchronous-random-baselines` from that planning head, and execute `docs/design/2026-08-27-task009c-synchronous-random-baseline-preexperiment-plan.md` manually in gate order.

TASK-009C starts from the completed TASK-009B implementation at `64dd2ff33951cb780f938a81c91c22dde8764c93`. It adds selected-pose reset ordering, a single-environment synchronous episode runner, seven random baselines, one HOLD diagnostic, a 37-episode preexperiment, and aligned coverage summaries.

TASK-009C does not authorize VLM, CNN, GRU, Actor, Critic, PPO, rewards, multi-environment training, controller recalibration, coverage-ROI changes, pose-library regeneration, or simulation-asset changes.
