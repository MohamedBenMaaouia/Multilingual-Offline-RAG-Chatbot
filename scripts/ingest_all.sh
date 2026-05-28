#!/usr/bin/env bash
# Wipe the Chroma collection and rebuild it from data/raw/.
#
# Per ARCHITECTURE.md §3 — there is one canonical ingestion path
# (src/ingest.py). This script just provides the file-system reset
# alongside it so `make ingest` is a single command from a clean slate.
set -euo pipefail

cat <<'BANNER'
================================================================
  RAG POC — Full Ingest
  Wipes ./data/chroma/ then runs python -m src.ingest
================================================================
BANNER

CHROMA_DIR="./data/chroma"

echo "[1/2] Removing existing ChromaDB at ${CHROMA_DIR}/ ..."
rm -rf "${CHROMA_DIR}"
mkdir -p "${CHROMA_DIR}"
touch "${CHROMA_DIR}/.gitkeep"

echo "[2/2] Running ingestion (this may take several minutes on CPU)..."
python -m src.ingest

echo "Ingest complete."
