#!/usr/bin/env bash
# Delete the embedded Qdrant store (all collections). Stop the API first:
# embedded Qdrant locks data/qdrant/ while a process has it open.
set -euo pipefail

QDRANT_DIR="./data/qdrant"

if [ -d "${QDRANT_DIR}" ]; then
    echo "==> Removing existing ${QDRANT_DIR}..."
    rm -rf "${QDRANT_DIR}"
fi

mkdir -p "${QDRANT_DIR}"
touch "${QDRANT_DIR}/.gitkeep"

echo "[OK] Qdrant store reset at ${QDRANT_DIR}"
