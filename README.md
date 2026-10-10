# MTG Judge Voice Assistant

An offline voice assistant that answers Magic: The Gathering rules questions like a judge. Ask it out loud ("Judge, what happens to Urza's Saga under Blood Moon?") and it looks up the current Comprehensive Rules, tournament documents and official card rulings, then answers with citations. Everything runs locally: speech recognition and voice (Moonshine), retrieval, and the LLM (Ollama). It runs on a Mac and is built to run on a Raspberry Pi 5.

## Status (October 2026)

| | |
|---|---|
| **Works on the Mac** | Full voice loop tested live: format prompt, wake word, multi-sentence questions, thinking cues, spoken answers, "Judge, stop" |
| **Raspberry Pi 5** | Profile, systemd units and setup script ready; **not yet run on real hardware** |
| **Accuracy** | `qwen3:8b` with reasoning: **73%** on a 38-case hard-ruling benchmark (Urza's Saga, layers, replacement effects…); about 85–90% cite the right rule on basic questions ([docs/benchmarks.md](docs/benchmarks.md)) |
| **Data** | CR 2026-09-25, 32k cards, 76k official rulings, MTR/IPG/JAR, 6 months of Modern MTGO decklists |
| **Known issues / roadmap** | [TODO.md](TODO.md): a late spoken acknowledgement in live sessions, a wrong Blood Moon + Urza's Saga answer (retrieval), voice fonts, custom sounds |

It isn't a replacement for a certified judge. Answers carry a confidence prefix ("I'm fairly sure…", "I'm not certain, but…"), so treat low-confidence rulings as a starting point.

## Getting started

```bash
git clone https://github.com/dblodorn/mtg-judge.git
cd mtg-judge
make quickstart        # or: bash scripts/getting_started.sh   (add --pi on a Raspberry Pi)
make run               # voice judge
```

`make quickstart` is safe to re-run; it skips whatever's already done:
1. Checks the system. Python 3.11/3.12 is required; on Linux it also needs PortAudio and python3-venv.
2. Creates `.venv` and installs the dependencies.
3. Installs Ollama and pulls `qwen3:8b` and `nomic-embed-text`.
4. Gets the rules and card index. It downloads a published snapshot if `MTG_JUDGE_ARTIFACTS` or `~/Dropbox/mtg-judge-artifacts` has one; otherwise it builds the index, which takes about 30 minutes on a Mac.
5. Answers one question as a text-only smoke test.
6. Shows which microphone and speaker it will use.

**macOS:** the first `make run` makes macOS ask your terminal app for microphone access, so click Allow. If the judge prints "Waiting for the microphone…", enable your terminal under System Settings → Privacy & Security → Microphone and restart it.

Text-only, no audio: `make ask Q="Does protection from red stop Lightning Bolt?"`

## Talking to the judge

| Say | What happens |
|---|---|
| *(at startup)* "Modern" / "EDH" / "Legacy" … | Answers "Which format are we playing?". The format is used for legality questions. Skip the question with `format:` in the config or `--format modern`. |
| "Judge, does deathtouch work with trample?" | A rules question. Pausing mid-sentence is fine: it waits for 3 seconds of quiet. |
| "Judge" … *(chime)* … "what happens if…" | A bare wake word gets a chime, and the mic stays open for your question. |
| "Judge, switch to Pioneer" | Changes the format. Add a question and it answers that too. |
| "Judge, are you there?" | "I'm here." |
| "Judge, stop" / "Judge, goodbye" | Ends the session (or press Ctrl+C). |

While it works, the judge tells you what it's looking up: *"Just a minute, looking up rulings for Blood Moon and Urza's Saga."* If reasoning runs past 12 seconds it says "Still thinking.", then chimes every 12 seconds. Expect about 10–20 seconds per answer on a Mac. Anything not addressed to "Judge" is ignored, and the mic is muted while the judge speaks, so table talk and its own voice don't trigger it.

Speech recognition is biased toward MTG vocabulary ("lore counter", "trample", …) and the current format's ~130 most-played cards. Without the bias, "Urza's Saga" came out as "ursus of".

## Raspberry Pi 5 (8GB)

Don't build the index on the Pi: embedding ~113k chunks there takes hours. Install a published snapshot, or copy one from the Mac:

```bash
# On the Pi
git clone https://github.com/dblodorn/mtg-judge.git && cd mtg-judge
sudo apt install -y libportaudio2 python3-venv
bash scripts/getting_started.sh --pi                     # fetches the index if MTG_JUDGE_ARTIFACTS points at a snapshot
#   …or, from the Mac:  make sync-pi PI=pi@raspberrypi.local
MTG_JUDGE_CONFIG=config/pi.yaml make run

# Autostart on boot, plus a weekly data update
sudo cp deploy/mtg-judge.service deploy/mtg-judge-update.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mtg-judge mtg-judge-update.timer
```

`config/pi.yaml` keeps the desktop's model and retrieval, so the Mac reproduces what the Pi will say. It only shrinks the LLM context window to 4k (to fit in 8 GB), uses the small streaming STT model and a lighter voice. Expect answers to take minutes: the model reasons for about 500 tokens per question on the Pi's CPU. For speed over accuracy, use `LLM_MODEL=gemma2:9b LLM_THINK=false` (69% vs 73% on the hard set).

## Keeping data current

```bash
make update      # new CR, new sets + rulings, MTR/IPG/JAR, and the metagame; prints what changed
```

Only new or changed chunks get embedded, so updates are quick. Restart the judge afterwards (on the Pi: `sudo systemctl restart mtg-judge`; the weekly timer does this for you). At startup the judge prints the data versions it loaded (`📖 Rules 20260925 · cards as of …`).

### Data snapshots

Large data stays out of git. `make publish` packs the built index and a raw-corpus snapshot into an immutable, versioned artifact with a sha256 manifest. `make fetch-data` installs one (`latest` by default) and verifies the checksums.

```
<root>/latest.json
<root>/<date>_cr<CR date>_cards<Scryfall date>_<content hash>/{manifest.json, index.tar.gz, corpus.tar.gz}
```

The root is `MTG_JUDGE_ARTIFACTS`, which defaults to `~/Dropbox/mtg-judge-artifacts` (synced by the Dropbox app). `fetch_data.py --source` also accepts an HTTPS base URL. Archives are deterministic, so publishing identical data does nothing. The artifacts contain Wizards' rules text verbatim, so keep the storage private. A Cloudflare R2 backend is planned ([TODO.md](TODO.md)).

### Metagame

```bash
make metagame FORMAT=modern   # last 6 months of MTGO Challenges/Showcases/Qualifiers → archetypes + tiers
```

Decklists come from official MTGO results on mtgo.com, fetched politely (one request per second, cached per event). Archetypes are found by clustering decks by their nonland mainboards. Tiers are set by metagame share (Tier 1 ≥5%, Tier 2 ≥2%). Output goes to `corpus/metagame/<format>/`: `report.md`, `report.json` and `cards.json` (the cards played in Tier 1–2 decks, which also feed speech biasing). mtgdecks.net isn't used because its robots.txt disallows AI crawlers.

## How it works

```
Mic → Moonshine STT (MTG-biased) → wake word / format / commands
    → Query Router → Hybrid retrieval → Ollama qwen3:8b (reasoning) → Moonshine TTS → Speaker
                          ↑
   CR + glossary · MTR · IPG · JAR · Scryfall cards + official rulings   (~113k chunks)
```

1. **Router:** finds rule numbers (→ CR), tournament words (→ MTR/IPG/JAR), and card names. Card names are matched on whole words, longest first, and common-word names like "Opt" or "Island" need a capital letter.
2. **Direct lookups:** exact rule numbers and named cards are pinned first. Each card comes with its 2 rulings closest to the question and its legality in the current format.
3. **Hybrid search:** dense search (nomic-embed-text → ChromaDB) and BM25 run over the rules documents, once for the question as asked and once with card names replaced by what the cards do. All rankings are fused with RRF.
4. **Parent expansion + MMR:** a matched sub-rule brings in its section, trimmed to the relevant lines. Near-duplicates are dropped.
5. **LLM:** a structured answer `{answer, confidence, citations}`. Citations missing from the retrieved context are dropped, and confidence is lowered when that happens. If reasoning runs out of budget, the model answers again without reasoning.
6. **TTS:** speaks the ruling with a confidence prefix.

Architecture notes for contributors are in [CLAUDE.md](CLAUDE.md).

## Requirements

- **Development:** a Mac (Apple Silicon) or Linux desktop, Python 3.11/3.12, about 10 GB of disk (models + index), and roughly 8 GB of free RAM for `qwen3:8b`.
- **Deployment:** a Raspberry Pi 5 (8 GB), a USB microphone, a speaker (3.5mm or USB), and a 64 GB+ A2 microSD card.

## Configuration

`config/default.yaml` is the desktop profile. `config/pi.yaml` only overrides what differs on the Pi. Select a profile with `MTG_JUDGE_CONFIG=…` or `--config`. The env vars `LLM_MODEL`, `LLM_THINK`, `EMBED_MODEL` and `OLLAMA_HOST` override either profile.

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | `qwen3:8b` | Ollama LLM model (see [docs/benchmarks.md](docs/benchmarks.md)) |
| `think` / `max_tokens` | `true` / `3072` | Reasoning on, capped so it can't run away |
| `num_ctx` | `8192` (Pi: `4096`) | LLM context window |
| `embed_model` | `nomic-embed-text` | Must match the model the index was built with |
| `wake_word` | `judge` | `null` = answer every utterance |
| `format` | `null` | Game format; `null` = ask by voice at startup |
| `question_pause` | `3.0` | Seconds of quiet that end a question |
| `thinking_cue` | `speech` | `speech`, `beep` or `off` |
| `metagame_formats` | `[modern]` | Formats `make update` refreshes the metagame for |
| `k` / `max_context_chars` | `4` / `3000` | Retrieval budget (more context scored no better) |
| `tts_voice` | `piper_en_US-amy-medium` | See `moonshine_voice.list_tts_voices("en_us")` |
| `moonshine_model_arch` | `null` | STT model; `null` = library default (Pi: `small-streaming`) |
| `input_device` / `output_device` | `null` | Audio device index or name |

## Sources

| Source | What | License |
|--------|------|---------|
| [Comprehensive Rules](https://magic.wizards.com/en/rules) | Official game rules + glossary (official .txt) | Wizards of the Coast |
| Magic Tournament Rules (MTR) | Tournament procedures (official text via blogs.magicjudges.org) | Wizards of the Coast |
| Infraction Procedure Guide (IPG) | Competitive REL penalties | Wizards of the Coast |
| Judging at Regular (JAR) | Regular REL guidance | Wizards of the Coast |
| [Scryfall](https://scryfall.com/docs/api/bulk-data) | Oracle card text, legalities, and official rulings | Scryfall (card data © Wizards) |
| [MTGO decklists](https://www.mtgo.com/decklists) | Metagame (Challenges, Showcases, Qualifiers) | Wizards of the Coast |

## Evaluation

```bash
make eval-hard   # 39 tricky interactions, graded on the answer itself (by topic)
make eval-llm    # 20 basic questions: retrieval + cited section + latency
make eval        # pytest retrieval checks (needs Ollama + built index)
bash scripts/benchmark_models.sh "qwen3:8b:think gemma2:9b"   # compare models (works on the Pi too)
```

Every expected answer in `tests/fixtures/hard_cases.json` is checked against the current CR or an official ruling. Each case records its source, including cases where popular articles or AI answers get it wrong (for example, Blood Moon no longer kills Urza's Saga since the June 2025 Saga rules change). 14 local models were compared in [docs/benchmarks.md](docs/benchmarks.md).

## Troubleshooting

**"Waiting for the microphone…":** on macOS, allow your terminal under System Settings → Privacy & Security → Microphone. On Linux, check `arecord -l` and `input_device`.

**Wrong mic or speaker:** list devices with `.venv/bin/python -c 'import sounddevice; print(sounddevice.query_devices())'` and set `input_device` / `output_device`.

**Slow startup after killing the judge mid-answer:** Ollama finishes the abandoned answer first; wait or restart Ollama. Stop with "Judge, stop" or Ctrl+C between answers ([TODO.md](TODO.md)).

**Too slow on the Pi:** check you're running the Pi profile, then try `LLM_MODEL=gemma2:9b LLM_THINK=false`.

**Judge answers table talk or itself:** keep `wake_word` set. The mic is muted while it speaks.

**Index missing or corrupted:** `make fetch-data` (snapshot) or `rm -rf index/ && make index`.

**Stale cards or rules:** `make update`.

**Verify zero cloud calls while running:**
```bash
sudo tcpdump -i any -n 'tcp and not port 22' &
make run          # all traffic should be localhost only
```

## Development

```bash
make test        # unit tests (no Ollama needed)
make lint        # ruff
make eval-hard   # integration benchmark (needs Ollama + index)
```

## License

MIT (code). See [Sources](#sources) for data licenses.
