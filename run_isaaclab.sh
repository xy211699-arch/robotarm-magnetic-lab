#!/usr/bin/env bash
#
# Run this project with Isaac Sim's bundled Python even when the caller's shell
# has Conda base (or another virtual environment) activated.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_LAUNCHER="/mnt/isaac-linux/IsaacLab/isaaclab.sh"
ARGS=("$@")

cd "${PROJECT_DIR}"

# Kit 110's experimental geometry streaming can conflict with the Fabric
# render delegate and leave referenced dynamic meshes at stale transforms.
# Apply the workaround to project Python entry points, but not to helper
# invocations such as ``-p -m pip``.
if [[ "${ARGS[0]:-}" == "-p" && "${ARGS[1]:-}" == *.py ]]; then
    HAS_KIT_ARGS=false
    for ARG in "${ARGS[@]}"; do
        if [[ "${ARG}" == --kit_args* ]]; then
            HAS_KIT_ARGS=true
            break
        fi
    done
    if [[ "${HAS_KIT_ARGS}" == false ]]; then
        ARGS+=("--kit_args=--/UJITSO/enabled=false --/UJITSO/geometry=false")
    fi
fi

exec env \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u CONDA_PROMPT_MODIFIER \
    -u VIRTUAL_ENV \
    -u PYTHONEXE \
    -u PYTHONHOME \
    -u LD_LIBRARY_PATH \
    -u LD_PRELOAD \
    PYTHONPATH="${PROJECT_DIR}/source/robotarm_magnetic_lab" \
    "${ISAACLAB_LAUNCHER}" "${ARGS[@]}"
