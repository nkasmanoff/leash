#!/usr/bin/env bash
# Start the Leash OpenAI shim and print OpenCode launch instructions.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export LEASH_URL="${LEASH_URL:-https://nkasmanoff--leash-leash-chat-dev.modal.run}"
export LEASH_CLAMP="${LEASH_CLAMP:-false}"
export LEASH_THINKING="${LEASH_THINKING:-true}"
export LEASH_FAKE_TOOLS="${LEASH_FAKE_TOOLS:-true}"
export LEASH_SHIM_HOST="${LEASH_SHIM_HOST:-127.0.0.1}"
export LEASH_SHIM_PORT="${LEASH_SHIM_PORT:-8787}"

echo "== Leash harness =="
echo "Leash backend:  $LEASH_URL"
echo "Shim (OpenAI):  http://${LEASH_SHIM_HOST}:${LEASH_SHIM_PORT}/v1"
echo "OpenCode model: leash/qwen3-32b"
echo "Capping:        LEASH_CLAMP=$LEASH_CLAMP"
echo "Thinking:       LEASH_THINKING=$LEASH_THINKING"
echo "Fake tools:     LEASH_FAKE_TOOLS=$LEASH_FAKE_TOOLS"
echo
echo "In another terminal:"
echo "  cd $ROOT"
echo "  opencode"
echo
echo "Inside OpenCode:"
echo "  /models          # pick leash/qwen3-32b if needed"
echo "  Ask it to edit a file in this repo"
echo
echo "Projection traces land in traces/harness/"
echo "Harness dashboard:  cd dashboard && npm run dev  →  Agent tab"
echo "  (requires ./scripts/run_harness.sh for /api/agent on :8787)"
echo

exec python scripts/openai_shim.py
