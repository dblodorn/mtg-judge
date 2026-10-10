#!/usr/bin/env bash
# OS/arch-aware Ollama install + model pull for mtg-judge.
#   LLM_MODEL=qwen2.5:7b bash scripts/install_ollama.sh   # pull a different model
set -euo pipefail

OS=$(uname -s)
ARCH=$(uname -m)
case "$OS/$ARCH" in
  Darwin/arm64|Darwin/x86_64) echo "✓ macOS ($ARCH) detected";;
  Linux/x86_64)  echo "✓ Linux x86_64 detected";;
  Linux/aarch64) echo "✓ Linux ARM64 detected (Raspberry Pi 5)";;
  *) echo "✗ Unsupported platform: $OS/$ARCH"; exit 1;;
esac

if command -v ollama >/dev/null 2>&1; then
  echo "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
elif [ "$OS" = "Darwin" ]; then
  if command -v brew >/dev/null 2>&1; then
    echo "Installing Ollama via Homebrew..."
    brew install ollama
  else
    echo "✗ Install Ollama from https://ollama.com/download (or install Homebrew), then re-run."
    exit 1
  fi
else
  echo "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Make sure the server is up before pulling
if ! curl -sf http://localhost:11434/api/version >/dev/null; then
  if [ "$OS" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    brew services start ollama
  else
    # Linux installer registers a systemd service; fall back to a background server
    sudo systemctl start ollama 2>/dev/null || (nohup ollama serve >/tmp/ollama.log 2>&1 &)
  fi
  for _ in $(seq 1 30); do
    curl -sf http://localhost:11434/api/version >/dev/null && break
    sleep 1
  done
fi

LLM_MODEL="${LLM_MODEL:-qwen3:8b}"
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
