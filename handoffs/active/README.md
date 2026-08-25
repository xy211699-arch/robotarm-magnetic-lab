# Active task contracts

The currently authorized Linux task is `TASK-009A-stage1-controller-baseline-audit.md` on Windows planning branch `workflow/TASK-009A-stage1-controller-baseline-audit`.

TASK-009A publishes and audits the already completed Linux-side 10 Hz parameterized six-mode force controller. It does not authorize Linux to reconstruct that controller from the older TASK-008 one-second macro branch, and it does not yet authorize area-coverage changes, VLM, GRU, PPO, rewards, or random-policy experiments.

Linux must fetch the Windows planning branch, record its exact head, read the active contract and `docs/design/2026-08-25-vlm-gastric-coverage-research-contract-v1.md`, then create `feature/TASK-009A-stage1-controller-baseline-audit` from the exact current high-frequency controller commit in the Linux workspace. The required return report is `handoffs/reports/TASK-009A-stage1-controller-baseline-audit-report.md`.

The older TASK-008 contract and report remain historical evidence only. Its 1 Hz one-second macros, 50 mm vertex coverage, wait-force-wait timing, and double-ended UP couple are not the current research contract.
