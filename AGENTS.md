# Cross-machine Codex workflow

## Authority and roles

Approved specifications and active task contracts in `handoffs/active/` are authoritative. Windows Codex is the planning and acceptance side. Linux Codex is the implementation and evidence side. Implementation must occur on an isolated task branch and must not be pushed directly to `main`.

If an ambiguity would change architecture, safety behavior, interfaces, training semantics, or acceptance criteria, stop implementation and return a report with status `needs_decision`. Do not silently choose a design-changing interpretation.

## Evidence requirements

Every Linux report must include the base commit, head commit, current branch, commands executed, observed results, deviations from the task contract, unverified claims, and artifact paths. Report only directly observed results as verified. Large logs, datasets, recordings, checkpoints, and generated assets must remain outside normal Git history; report their paths and hashes instead.

## Safety and scope

Preserve existing user changes and repository history. Do not modify simulation source, assets, training configuration, or validated results unless the active task explicitly authorizes those paths. Never use capsule ground truth in deployable observations, action completion decisions, or device hard-failure decisions unless a later approved contract explicitly changes that boundary.
