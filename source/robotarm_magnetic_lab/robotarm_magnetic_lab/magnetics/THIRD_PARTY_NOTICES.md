# TASK-007 magnetic-model provenance

The repository-local `config.py`, `field_models.py`, and `resources/default.json`
were migrated from the local `robotarm.magnetic_sim` extension on 2026-08-20.
Only package resource lookup and explicit dependency discovery were changed.

| Source | Original SHA-256 |
|---|---|
| `robotarm/magnetic_sim/config.py` | `5d32740c62a75e06b7b876ed16f0043378ad45b72317b1f99637466b7f71ee07` |
| `robotarm/magnetic_sim/magnetics/field_models.py` | `be2f4d4af8db2e3a04552add61cbbc84d89e2348c08864a0c9cc3e6283265965` |
| `data/config/default.json` | `e38563d558f6945f3041458060965ce6cd4b7044eacce573318c0f0fdcd319a6` |

The analytical implementation uses Magpylib 5.2.3, Copyright (c) 2022,
Silicon Austria Labs and Magpylib Developers, under the BSD 3-Clause License.
The full license text is available in the Magpylib distribution and at
<https://github.com/magpylib/magpylib/blob/5.2.3/LICENSE>.

TASK-007 does not copy Magpylib itself. The runtime requires version 5.2.3 as a
Python dependency. `ROBOTARM_MAGPYLIB_VENDOR` may point at a verified local
installation during migration regression; no legacy path is embedded in source.
