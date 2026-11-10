#!/usr/bin/env bash
# Rebuild the Qdrant collection from data/raw/.
#
# Per ARCHITECTURE.md §3 — there is one canonical ingestion path
# (src/ingest.py), which drops and recreates the collection itself
# (wipe_first=True). Stop the API first when Qdrant runs embedded.
set -euo pipefail

cat <<'BANNER'
================================================================
  RAG POC — Full Ingest
  Drops the collection, then runs python -m src.ingest
================================================================
BANNER

echo "Running ingestion (this may take several minutes on CPU)..."
python -m src.ingest

echo "Ingest complete."
