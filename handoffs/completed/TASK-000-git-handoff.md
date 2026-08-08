# TASK-000: Git handoff round trip

Status: `complete`. Accepted remote response commit: `24a1469d3b797d75b4d7e8e3079a69c26e85fc83`.

## Objective

Prove that Linux Codex can receive an approved task contract from the Windows planning branch and return a structured evidence report through the shared GitHub Fork.

## Required base

Fetch `origin` and start from `workflow/TASK-000-git-handoff`. Record the exact base commit before making any change. Create and work only on `workflow/TASK-000-linux-response`.

## Authorized change

Create exactly one report at `handoffs/reports/TASK-000-git-handoff-report.md`. Do not modify source code, simulation assets, task or training configuration, existing validation results, or any other tracked file.

## Required observations

The report must contain the status, Linux distribution and kernel, repository path, Isaac Lab project path, current branch, base commit, HEAD immediately before the report commit, concise repository status, configured remote names with repository identities but no credentials, and whether Linux can fetch and push to the shared Fork. The final report commit cannot contain its own hash; Windows verifies that hash independently from the pushed branch.

The report must also record every command executed, its exit code or concise result, deviations from this contract, unverified claims, and paths to any external artifacts. Do not run simulation or training for TASK-000.

## Delivery

Commit only the report with message `docs: report TASK-000 Linux Git handoff`. Push `workflow/TASK-000-linux-response` to `origin`. Use one of the exact status values defined in `handoffs/reports/README.md`.
