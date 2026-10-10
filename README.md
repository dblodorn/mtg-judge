# MTG Judge Voice Assistant

A fully-offline voice assistant that answers Magic: The Gathering rules questions like a judge. Runs on a Raspberry Pi 5.

## Quick Start (Mac or Linux desktop)

```bash
git clone https://github.com/dblodorn/mtg-judge.git
cd mtg-judge
make setup      # venv (Python 3.11/3.12) + Ollama + models + corpus + index
make ask Q="Does protection from red stop Lightning Bolt?"   # text-only smoke test
make run        # voice mode
```

At startup the judge asks **"Which format are we playing?"**. Answer "Modern", "EDH", "Legacy" and so on. Say "Judge, switch to Pioneer" any time to change it, or set `format:` in the config (or `--format modern`) to skip the question. The format is used for legality questions: each named card's legality in that format goes into the context.

Start each question with the wake word: **"Judge, does deathtouch work with trample?"**. If you only say "Judge", it answers "Yes?" and treats the next thing you say as the question. Anything not addressed to the judge is ignored, so ordinary table talk doesn't trigger it. The mic is muted while the judge speaks. Set `wake_word: null` in the config to answer everything.

## Raspberry Pi 5 (8GB)

Build the index on the desktop, because embedding ~38k chunks on the Pi takes hours. Then copy it over:

```bash
# On the Mac
make index
make sync-pi PI=pi@raspberrypi.local     # rsyncs code + index/ to ~/mtg-judge

# On the Pi
cd ~/mtg-judge
make setup-venv
make setup-ollama                        # qwen3:8b + the same embed model as the Mac index
MTG_JUDGE_CONFIG=config/pi.yaml make run

# Autostart on boot
sudo cp deploy/mtg-judge.service /etc/systemd/system/
sudo systemctl enable --now mtg-judge
```

`config/pi.yaml` keeps the same model and retrieval as the desktop. It only shrinks the LLM context window to 4k (memory) and uses lighter STT/TTS. Expect answers to take a while on the Pi: reasoning adds about 500 output tokens per question. Use `LLM_MODEL=gemma2:9b LLM_THINK=false` if you want speed over accuracy.

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
5. **LLM** — Ollama `qwen3:8b` with thinking, on both desktop and Pi (most accurate in [docs/benchmarks.md](docs/benchmarks.md); `gemma2:9b` is the fast alternative), with schema-constrained output: `{answer, confidence, citations}`. Citations not found in the retrieved context are dropped, and confidence is downgraded.
6. **TTS** — Moonshine speaks the ruling with confidence prefix ("I'm fairly sure...")

Exact rule numbers ("what does 702.16a say") and named cards are looked up directly and pinned ahead of search results.

## Requirements

- **Raspberry Pi 5** (8GB)
- USB microphone
- Speaker (3.5mm or USB)
- microSD card (64GB+, A2-rated)

## Keeping data current

```bash
make update      # new CR, new sets + rulings, MTR/IPG/JAR, and the metagame; prints what changed
```

Only new or changed cards get embedded, so it's quick even on the Pi. Then restart the judge. On the Pi, `deploy/mtg-judge-update.timer` runs this weekly (Sunday 04:00) and restarts the service:

```bash
sudo cp deploy/mtg-judge-update.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now mtg-judge-update.timer
```

At startup the judge prints the data versions it loaded (`📖 Rules 20260925 · cards as of …`).

### Metagame

```bash
make metagame FORMAT=modern   # last 6 months of MTGO Challenges/Showcases/Qualifiers → archetypes + tiers
```

Decklists come from official MTGO results on mtgo.com, fetched at one request per second. Archetypes are found by clustering decks by their nonland mainboards, and tiers are set by metagame share (Tier 1 ≥5%, Tier 2 ≥2%). The output goes to `corpus/metagame/<format>/`: `report.md`, `report.json`, and `cards.json`, the cards played in Tier 1–2 decks. mtgdecks.net isn't used because its robots.txt disallows AI crawlers.

## Configuration

`config/default.yaml` is the desktop profile. `config/pi.yaml` only overrides what differs on the Pi. Select a profile with `MTG_JUDGE_CONFIG=…` or `--config`. The env vars `LLM_MODEL`, `EMBED_MODEL` and `OLLAMA_HOST` override either profile.

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | `qwen3:8b` | Ollama LLM model (benchmarks: `docs/benchmarks.md`) |
| `think` / `max_tokens` | `true` / `3072` | Reasoning on, capped so it can't run away |
| `embed_model` | `nomic-embed-text` | Must match the model the index was built with |
| `wake_word` | `judge` | `null` = answer every utterance |
| `format` | `null` | Game format; `null` = ask by voice at startup |
| `metagame_formats` | `[modern]` | Formats `make update` refreshes the metagame for |
| `k` | `4` | Max chunks sent to the LLM |
| `max_context_chars` | `3000` | Total context budget (more scored no better) |
| `rrf_k` | `60` | RRF fusion constant |
| `mmr_lambda` | `0.5` | 0=max diversity, 1=max relevance |
| `tts_voice` | `piper_en_US-amy-medium` | See `moonshine_voice.list_tts_voices("en_us")` |
| `moonshine_model_arch` | `null` | STT model; `null` = library default (Pi: `small-streaming`) |
| `input_device` / `output_device` | `null` | Audio device index or name |

## Corpus Sources

| Source | What | License |
|--------|------|---------|
| MTG Comprehensive Rules | Official game rules | Wizards of the Coast |
| Magic Tournament Rules (MTR) | Tournament procedures (official text via blogs.magicjudges.org) | Wizards of the Coast |
| Judging at Regular (JAR) | Regular REL guidance | Magic Judges |
| Infraction Procedure Guide (IPG) | Competitive REL penalties | Magic Judges |
| Scryfall Oracle | Card text + metadata (paper cards) | Scryfall (CC BY) |

Refresh weekly: `make corpus && make index`

## Evaluation

```bash
make eval        # pytest retrieval checks (needs Ollama + built index)
make eval-llm    # score every case, retrieval + LLM, with per-question latency
MTG_JUDGE_CONFIG=config/pi.yaml make eval-llm   # try the Pi model/budgets on the desktop
make eval-hard   # 11 tricky interactions (Urza's Saga, layers…), graded on the answer itself
bash scripts/benchmark_models.sh "qwen3:8b:think gemma2:9b"   # compare models
```

The eval runs 20 known MTG questions and checks that the expected sources, rule sections, cards and keywords are retrieved. It also checks that the LLM cites an expected section. The hard set grades the answers themselves against the current CR and official rulings. Model results are in [docs/benchmarks.md](docs/benchmarks.md).

## Troubleshooting

**No microphone detected:**
```bash
arecord -l  # list recording devices
```

**Ollama too slow on Pi:** make sure you're running the Pi profile (`MTG_JUDGE_CONFIG=config/pi.yaml`). Then lower `max_context_chars`, or try a smaller model (`LLM_MODEL=qwen2.5:1.5b`).

**Judge answers its own voice or table talk:** keep `wake_word` set. The mic is muted while TTS plays.

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
make test        # unit tests only, no Ollama needed
make lint        # ruff
make eval        # integration tests (requires Ollama + built index)
```

## License

MIT (code) — See corpus sources for their respective licenses.
