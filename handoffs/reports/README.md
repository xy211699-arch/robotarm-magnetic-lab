# Handoff report contract

Every report must use exactly one status: `complete`, `partial`, `needs_decision`, or `blocked`.

`complete` means every authorized requirement was executed and verified. `partial` means useful authorized work was completed but one or more requirements remain. `needs_decision` means a design-changing ambiguity requires Windows-side approval. `blocked` means no further authorized progress is possible because of an external dependency or access failure.

Every report must include task ID, status, base commit, HEAD immediately before the report commit, branch, environment identity, commands and exit codes, observed results, deviations, unverified claims, and artifact paths. The final report commit hash is verified from the pushed remote branch rather than embedded recursively in its own content. Credentials, private keys, access tokens, large logs, datasets, recordings, generated assets, and checkpoints must never be committed.
