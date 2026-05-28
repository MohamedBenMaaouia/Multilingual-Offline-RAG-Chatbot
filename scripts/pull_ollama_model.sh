#!/usr/bin/env bash
set -euo pipefail

MODEL="mistral:7b-instruct"

echo "==> Pulling Ollama model: ${MODEL}"
echo "    (this can take a few minutes the first time — ~4 GB)"

ollama pull "${MODEL}"

echo "==> Verifying model is present in 'ollama list'..."
if ollama list | grep -q "${MODEL}"; then
    echo "[OK] ${MODEL} is available locally."
else
    echo "[FAIL] ${MODEL} not found in 'ollama list' after pull." >&2
    exit 1
fi
