# MTG Judge Voice Assistant

A fully-offline voice assistant that answers Magic: The Gathering rules questions like a judge. Runs on a Raspberry Pi 5.

## Quick Start

```bash
git clone https://github.com/dblodorn/mtg-judge.git
cd mtg-judge
make setup
make run
```

Speak a rules question. Get a spoken answer with rule citations. No cloud calls — everything runs locally.

## Architecture

```
Mic → Moonshine STT → Query Router → Hybrid RAG → Ollama LLM → Moonshine TTS → Speaker
                                       ↑
              MTG CR + MTR + JAR + IPG + Scryfall oracle (36k+ chunks)
```

### RAG Pipeline

1. **Query Router** — detects rule numbers, card names, or tournament keywords → routes to appropriate sources
2. **Hybrid Retrieval**:
   - **BM25 sparse** — exact-match for rule numbers and card names
   - **Dense** (nomic-embed-text via Ollama → ChromaDB) — semantic similarity
   - **RRF fusion** — combines both rankings
3. **Parent Document Expansion** — retrieves sub-rule (e.g. 702.16a), returns full section (702.16) for context
4. **MMR Diversification** — avoids returning 5 near-identical sub-rules
5. **LLM** — Ollama (Llama 3.1 8B) with structured output: `{answer, confidence, citations}`
6. **TTS** — Moonshine speaks the ruling with confidence prefix ("I'm fairly sure...")

## Requirements

- **Raspberry Pi 5** (8GB recommended, 4GB with `qwen2.5:3b` model)
- USB microphone
- Speaker (3.5mm or USB)
- microSD card (64GB+, A2-rated)

## Configuration

All knobs in `config/default.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | `llama3.1:8b` | Ollama LLM model |
| `embed_model` | `nomic-embed-text` | Ollama embedding model |
| `k` | `8` | Top-k chunks after MMR |
| `rrf_k` | `60` | RRF fusion constant |
| `mmr_lambda` | `0.5` | 0=max diversity, 1=max relevance |
| `tts_voice` | `amy` | Moonshine/Piper voice |
| `moonshine_model_arch` | `null` (auto) | tiny on Pi, medium on desktop |

### Pi 5 with 4GB RAM

Use a smaller model:
```bash
LLM_MODEL=qwen2.5:3b make setup-ollama
```

## Corpus Sources

| Source | What | License |
|--------|------|---------|
| MTG Comprehensive Rules | Official game rules | Wizards of the Coast |
| Magic Tournament Rules (MTR) | Tournament procedures | Wizards of the Coast |
| Judging at Regular (JAR) | Regular REL guidance | Magic Judges |
| Infraction Procedure Guide (IPG) | Competitive REL penalties | Magic Judges |
| Scryfall Oracle | Card text + metadata | Scryfall (CC BY) |

Refresh weekly: `make corpus && make index`

## Evaluation

```bash
make eval
```

Runs 20+ known MTG questions through the retrieval pipeline and checks:
- Expected sources are retrieved (CR, Scryfall, IPG, etc.)
- Expected rule sections appear (e.g. 702.16 for protection)
- Expected keywords are present in retrieved text

## Troubleshooting

**No microphone detected:**
```bash
arecord -l  # list recording devices
```

**Ollama too slow on Pi:**
```bash
LLM_MODEL=qwen2.5:3b bash scripts/install_ollama.sh
```

**ChromaDB index corrupted:**
```bash
rm -rf index/ && make index
```

**Scryfall data stale:**
```bash
make corpus && make index
```

**Verify zero cloud calls:**
```bash
sudo tcpdump -i any -n 'tcp and not port 22' &
make run
# All traffic should be localhost only
```

## Development

```bash
# Run tests (unit only, no Ollama needed)
make test

# Run integration tests (requires built index)
.venv/bin/pytest tests/test_evaluation.py -v -m integration
```

## License

MIT (code) — See corpus sources for their respective licenses.
