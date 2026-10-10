#!/usr/bin/env bash
# Compare LLMs on the eval sets under one config profile (default: the Pi budget).
#   bash scripts/benchmark_models.sh "qwen3:8b:think gemma2:9b"      # on the Mac or the Pi
#   CONFIG=config/default.yaml REPEAT=1 bash scripts/benchmark_models.sh "qwen2.5:7b"
# Append ":think" to a model to enable reasoning mode (qwen3), e.g. "qwen3:4b:think".
# EASY=0 skips the easy set (faster screening).
set -euo pipefail

MODELS="${1:-qwen3:8b:think}"
CONFIG="${CONFIG:-config/pi.yaml}"
REPEAT="${REPEAT:-3}"
EASY="${EASY:-1}"
PY=.venv/bin/python

for spec in $MODELS; do
  model="${spec%:think}"
  think=false
  [ "$spec" != "$model" ] && think=true
  echo "################ $model (think=$think, $CONFIG, hard set x$REPEAT)"
  LLM_MODEL="$model" LLM_THINK="$think" $PY scripts/eval_llm.py --config "$CONFIG" \
    --cases tests/fixtures/hard_cases.json --repeat "$REPEAT" 2>&1 | grep -v Warning
  if [ "$EASY" = "1" ]; then
    echo "---- easy set"
    LLM_MODEL="$model" LLM_THINK="$think" $PY scripts/eval_llm.py --config "$CONFIG" 2>&1 \
      | grep -E "^(Retrieval|LLM|Tokens|Answers|By topic)"
  fi
  echo
done
