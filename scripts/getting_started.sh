#!/usr/bin/env bash
# MTG Judge — getting started. Safe to re-run: every step skips what's already done.
#
#   bash scripts/getting_started.sh            # Mac or Linux desktop
#   bash scripts/getting_started.sh --pi       # Raspberry Pi 5 (uses config/pi.yaml)
#   bash scripts/getting_started.sh --build    # build the index locally instead of downloading it
#
# Steps: system check → Python venv → Ollama + models → rules/card data → text smoke
# test → audio check. Ends by telling you how to start the voice judge.
set -euo pipefail
cd "$(dirname "$0")/.."

PI=0
BUILD=0
for arg in "$@"; do
  case "$arg" in
    --pi) PI=1 ;;
    --build) BUILD=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done
if [ "$PI" = 1 ]; then export MTG_JUDGE_CONFIG=config/pi.yaml; fi

bold() { printf "\n\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
die()  { printf "  \033[31m✗\033[0m %s\n" "$*"; exit 1; }

OS=$(uname -s)
ARCH=$(uname -m)
PY=.venv/bin/python

# ---------------------------------------------------------------------------
bold "1/6  System"
case "$OS/$ARCH" in
  Darwin/*)       ok "macOS ($ARCH)" ;;
  Linux/aarch64)  ok "Linux ARM64 (Raspberry Pi 5)" ;;
  Linux/x86_64)   ok "Linux x86_64" ;;
  *) die "Unsupported platform: $OS/$ARCH" ;;
esac
if [ "$OS" = Linux ]; then
  # sounddevice needs PortAudio; venv needs python3-venv
  missing=()
  ldconfig -p 2>/dev/null | grep -q libportaudio || missing+=(libportaudio2)
  python3 -c "import ensurepip" 2>/dev/null || missing+=(python3-venv)
  if [ ${#missing[@]} -gt 0 ]; then
    die "Missing system packages. Run: sudo apt install -y ${missing[*]}"
  fi
  ok "PortAudio and python3-venv present"
fi
free_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1024/1024}')
[ "$free_gb" -ge 15 ] && ok "${free_gb} GB free disk" || warn "Only ${free_gb} GB free — models + index need ~10 GB"

# ---------------------------------------------------------------------------
bold "2/6  Python environment"
PYTHON=$(command -v python3.12 || command -v python3.11 || true)
if [ -z "$PYTHON" ]; then
  if python3 -c 'import sys; exit(not ((3, 11) <= sys.version_info[:2] < (3, 13)))' 2>/dev/null; then
    PYTHON=$(command -v python3)
  elif [ "$OS" = Darwin ]; then
    die "Python 3.11 or 3.12 required. Install it with: brew install python@3.12"
  else
    die "Python 3.11 or 3.12 required (Raspberry Pi OS Bookworm ships 3.11)."
  fi
fi
ok "Using $($PYTHON --version) at $PYTHON"
if [ -x "$PY" ] && $PY -c "import moonshine_voice, chromadb, bm25s, ollama" 2>/dev/null; then
  ok ".venv already set up"
else
  [ -d .venv ] || "$PYTHON" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e ".[dev]"
  ok "Installed dependencies into .venv"
fi

# ---------------------------------------------------------------------------
bold "3/6  Ollama + models"
LLM_MODEL=$($PY -c "import sys; sys.path.insert(0, '.'); from src.config import load_config; print(load_config()['llm_model'])")
EMBED_MODEL=$($PY -c "import sys; sys.path.insert(0, '.'); from src.config import load_config; print(load_config()['embed_model'])")
# Capture the list once: `ollama list | grep -q` trips pipefail when grep exits early
models=$(command -v ollama >/dev/null && curl -sf http://localhost:11434/api/version >/dev/null && ollama list || true)
if grep -q "^${LLM_MODEL}" <<<"$models" && grep -q "^${EMBED_MODEL}" <<<"$models"; then
  ok "Ollama running with $LLM_MODEL and $EMBED_MODEL"
else
  LLM_MODEL="$LLM_MODEL" EMBED_MODEL="$EMBED_MODEL" bash scripts/install_ollama.sh
fi

# ---------------------------------------------------------------------------
bold "4/6  Rules + card data (index)"
INDEX=$($PY -c "import sys; sys.path.insert(0, '.'); from src.config import load_config; print(load_config()['index_path'])")
ARTIFACTS="${MTG_JUDGE_ARTIFACTS:-$HOME/Dropbox/mtg-judge-artifacts}"
if [ -d "$INDEX/chroma" ] && [ -f "$INDEX/parents.jsonl" ]; then
  ok "Index present in $INDEX"
elif [ "$BUILD" = 0 ] && { [ -f "$ARTIFACTS/latest.json" ] || [[ "$ARTIFACTS" == http* ]]; }; then
  echo "  Downloading the published index from $ARTIFACTS ..."
  $PY scripts/fetch_data.py --source "$ARTIFACTS"
  $PY scripts/parse_corpus.py >/dev/null   # rebuild chunks/parents from the snapshot
  ok "Installed published index"
else
  warn "No published index found — building from scratch (fetch + embed ~110k chunks)."
  warn "Takes ~30 min on a Mac; hours on a Pi (better: build on a Mac, then 'make sync-pi' or 'make publish')."
  $PY scripts/fetch_corpus.py
  $PY scripts/parse_corpus.py
  $PY scripts/build_index.py
  ok "Index built"
fi

# ---------------------------------------------------------------------------
bold "5/6  Smoke test (text only, no audio)"
answer=$($PY -m src.app --format modern --ask "Does protection from red stop Lightning Bolt?" 2>/dev/null | grep "⚖️" || true)
if [ -n "$answer" ]; then
  ok "Judge answered:"
  echo "     ${answer#*⚖️  }"
else
  die "No answer — run: $PY -m src.app --ask \"Does protection from red stop Lightning Bolt?\" to see the error"
fi

# ---------------------------------------------------------------------------
bold "6/6  Audio"
$PY - <<'EOF'
import sounddevice as sd
d = sd.query_devices()
i, o = sd.default.device
print(f"  ✓ Microphone: {d[i]['name']}" if i is not None and i >= 0 else "  ! No default microphone found")
print(f"  ✓ Speaker:    {d[o]['name']}" if o is not None and o >= 0 else "  ! No default speaker found")
EOF
echo "  To use other devices, set input_device / output_device in config/default.yaml"
echo "  (index or name from: $PY -c 'import sounddevice; print(sounddevice.query_devices())')"
if [ "$OS" = Darwin ]; then
  echo "  macOS will ask your terminal app for microphone access the first time — click Allow."
  echo "  (System Settings → Privacy & Security → Microphone if you missed it.)"
fi

# ---------------------------------------------------------------------------
bold "Ready. Start the voice judge:"
if [ "$PI" = 1 ]; then echo "  MTG_JUDGE_CONFIG=config/pi.yaml make run"; else echo "  make run"; fi
cat <<'EOF'

  1. It asks "Which format are we playing?" — say e.g. "Modern" or "Commander".
  2. Ask questions starting with the wake word:
       "Judge, does deathtouch work with trample?"
       "Judge, what happens to Urza's Saga under Blood Moon?"
  3. Change format any time: "Judge, switch to Legacy."
  4. Ctrl+C to stop.

  It thinks before answering (qwen3:8b reasoning) — expect ~10–20 s on a Mac.
  Text-only:  make ask Q="your question"     Keep data current:  make update
EOF
