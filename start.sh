#!/usr/bin/env bash
# Launch SGLang's OpenAI-compatible server for WhiteRabbitNeo V3 on a single H200.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-WhiteRabbitNeo/WhiteRabbitNeo-V3-7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-whiterabbitneo-v3}"
PORT="${PORT:-30000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
TP_SIZE="${TP_SIZE:-1}"

args=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --context-length "$CONTEXT_LENGTH"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --tp "$TP_SIZE"
)

# Require an API key only when one is provided.
if [ -n "${API_KEY:-}" ]; then
  args+=(--api-key "$API_KEY")
fi

exec python3 -m sglang.launch_server "${args[@]}"
