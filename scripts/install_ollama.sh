#!/usr/bin/env bash
# Arch-aware Ollama install + model pull for mtg-judge.
set -euo pipefail

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  echo "✓ x86_64 detected";;
  aarch64) echo "✓ ARM64 detected (Raspberry Pi 5)";;
  armv7l)  echo "✓ ARMv7 detected";;
  *) echo "✗ Unsupported arch: $ARCH"; exit 1;;
esac

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

LLM_MODEL="${LLM_MODEL:-llama3.1:8b}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"

echo ""
echo "Pulling models (this may take a while on first run)..."
echo "  LLM: $LLM_MODEL"
echo "  Embeddings: $EMBED_MODEL"
ollama pull "$LLM_MODEL"
ollama pull "$EMBED_MODEL"

echo ""
echo "✓ Ollama ready."
echo "  Run 'ollama list' to verify."
