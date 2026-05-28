#!/usr/bin/env bash
set -euo pipefail

CHROMA_DIR="./data/chroma"

if [ -d "${CHROMA_DIR}" ]; then
    echo "==> Removing existing ${CHROMA_DIR}..."
    rm -rf "${CHROMA_DIR}"
fi

mkdir -p "${CHROMA_DIR}"
touch "${CHROMA_DIR}/.gitkeep"

echo "[OK] Chroma DB reset at ${CHROMA_DIR}"
