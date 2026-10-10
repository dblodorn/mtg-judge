# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fully offline voice MTG rules judge. It is developed on a Mac and deployed to a Raspberry Pi 5 (8GB). All inference is local: Moonshine (STT/TTS) and Ollama (LLM and embeddings). Only `scripts/fetch_corpus.py` touches the network. Keep runtime code free of cloud calls.

## Commands

Python 3.11/3.12. The Makefile picks `python3.12`/`python3.11` from PATH (macOS system Python is 3.9). Everything runs from the repo root, because paths like `config/`, `index/` and `corpus/` are relative.

```bash
make setup                  # venv + Ollama/models + corpus + index
make test                   # unit tests (-m "not integration"); no Ollama needed
.venv/bin/pytest tests/test_retriever.py::test_mmr_returns_k_results   # single test
make lint                   # ruff
make ask Q="..."            # text-only pipeline run (no mic/TTS)
make run                    # voice loop
make eval / make eval-llm   # retrieval pytest checks / full scored eval with latency (need Ollama + index)
make eval-hard              # tests/fixtures/hard_cases.json, graded on answers (verdict / answer_any)
bash scripts/benchmark_models.sh "m1 m2"   # model comparison under config/pi.yaml; results in docs/benchmarks.md
make corpus && make index   # refetch + reparse + reindex
make sync-pi PI=user@host   # rsync code + index/ to the Pi
make update                 # refetch rules/cards/rulings + metagame, incremental reindex, "what changed" report
make metagame FORMAT=modern # MTGO decklists (mtgo.com, 6 months) → corpus/metagame/<fmt>/report.md + cards.json
```

Config profiles: `config/default.yaml` is the desktop profile. `config/pi.yaml` is layered on top of it and lists only the keys it overrides. Select a profile with `MTG_JUDGE_CONFIG` or `--config`. The env vars `LLM_MODEL`, `EMBED_MODEL` and `OLLAMA_HOST` override any profile (`src/config.py`). If you add a knob, wire it through `build_pipeline` in `src/app.py`, which the demo and `scripts/eval_llm.py` also use.

## Architecture

Pipeline: `MicTranscriber → JudgeListener (wake word) → QueryRouter → HybridRetriever → JudgeLLM → TextToSpeech`

- **`src/judge.py`**: Moonshine `TranscriptEventListener`.
  - Accepts only lines starting with the wake word ("Judge, …"). A bare "Judge" gets "Yes?" and arms the next line.
  - Hands questions to a worker thread through a size-1 queue, and drops questions that arrive while busy.
  - Mutes the mic around `tts.say()` + `tts.wait()`; `say()` is async.
  - Game format (`src/formats.py`): `start()` asks for it by voice unless `format`/`--format` is set. The next line answers without the wake word. "Judge, switch to X" changes it, and a longer utterance is also answered as a question. The format goes on `route["format"]`, so the retriever appends a `<Format> legality:` line to pinned cards, and it's passed to `llm.answer(game_format=)`.
  - Work items on the queue are `("say", text)` or `("ask", question)`.
- **Moonshine 0.1.x** uses a builder API: `MicTranscriber().language(..).model_arch(..)` then `.load()` and `.start()`. Constructor kwargs raise `TypeError`. The package is pinned in `pyproject.toml`. TTS voice IDs look like `piper_en_US-amy-medium`.
- **`src/query_router.py`**:
  - Rule-number regex → `rule_numbers`, sources `["cr"]`. Tournament keywords → `mtr/ipg/jar`. Everything else searches `RULES_SOURCES` (cr/mtr/ipg/jar). Card text and rulings are never searched, because their 110k chunks swamp the CR. Named cards are pinned instead.
  - Card names are matched on whole words, longest first, against `index/card_names.json`. A full name beats another card's face. A trailing possessive ("Saga's") is allowed. Single-word names (or faces) listed under `ambiguous`, like "Opt", "Island" or "What", match only when capitalized and not at the start of a sentence.
  - Every route carries `mentioned_cards`.
- **`src/retriever.py`**: `retrieve(query, route=...)` runs these steps:
  1. Pins direct lookups. Rule numbers bring their parent group. Each named card brings its text plus the `rulings_per_card` rulings closest to the question, chosen by stored embeddings.
  2. Searches dense (Chroma) and BM25 for the question, and again for a copy with card names replaced by the card's type line and rules text (`_expand_card_names`). It fuses all four rankings with RRF.
  3. Expands each hit to its parent document.
  4. Diversifies with MMR. Relevance is the fused RRF score, and the stored embeddings only measure redundancy.
  5. Caps the total context at `max_context_chars`.

  BM25 filters by source inside the search (`weight_mask` from `_source_of`, which is derived from the ID prefix). Post-filtering a global top-n starves the rules sources. Long parents are trimmed to the heading plus the child's `anchor` line ±2.
- **`src/llm.py`**: Ollama `chat` with `format=JudgeRuling.model_json_schema()`, an explicit `num_ctx` and an optional `think`. An Ollama `ResponseError` returns a spoken fallback instead of raising. The default model is `qwen3:8b` with `think: true` and `max_tokens: 3072` on both profiles (see `docs/benchmarks.md`); `gemma2:9b` is the fast non-reasoning alternative. `verify_citations` drops citations that don't appear in the context and downgrades confidence when it does. Keep the system prompt as a stable first message so Ollama can reuse its KV cache.
- **`src/embeddings.py`**: Ollama `/api/embed`, using nomic task prefixes (`search_document: ` / `search_query: `). If you change the prefixes or the model, the index needs a rebuild. `build_index.py` detects this through collection metadata and rebuilds.

### Corpus → index

- **`fetch_corpus.py`** writes:
  - CR as the official `.txt`, linked from magic.wizards.com/en/rules
  - MTR and IPG as one JSONL record per section page on blogs.magicjudges.org, with `.alert` annotation boxes stripped
  - JAR from a single page, split on headings
  - Scryfall `oracle_cards` and `rulings` as gzipped JSONL
- **`parse_corpus.py`** writes `chunks.jsonl` (children), `parents.jsonl` and `card_names.json`:
  - CR: a group like `702.16` is the parent, with one line per rule. A `702.16.`-style heading becomes the title, `Example:` lines fold into the preceding rule, and sub-rules like `704.5aa` exist.
  - The glossary becomes `cr:glossary:<slug>`.
  - Cards are `scryfall:<oracle_id>`. Each official ruling is a child `scryfall:<oracle_id>:r<i>` with source `ruling`. The card's parent text is the card plus a `Rulings:` block.
  - It exits if a source parses suspiciously few chunks or produces duplicate IDs.
- **`build_index.py`** writes into `index/`:
  - Chroma (collection `mtg_judge`, cosine)
  - BM25, where row i corresponds to `chunk_ids.json[i]`
  - `parents.jsonl` and `card_names.json`

  The dense step is resumable and removes stale IDs. Chroma metadata must be flat scalars.

Chunk fields (`source`, `section`, `parent_id`, `anchor`, `name`, …) drive filtering, trimming and citations. If you change them, update the parser, retriever and `format_context` together.

### Data updates and metagame

- `fetch_corpus.py` records the versions it fetched (CR date, Scryfall `updated_at`) in `corpus/versions.json`. `scripts/update.py` diffs them, writes `index/VERSION.json` (printed at startup), and reports new cards by set.
- `fetch_metagame.py` reads official MTGO results from mtgo.com monthly archives. It's polite (1 req/s, cached per event), and recent events have no decklists until Wizards publishes them. mtgdecks.net is off-limits because its robots.txt blocks AI crawlers.
- `analyze_metagame.py` clusters nonland mainboards (cosine ≥0.55, greedy plus reassignment), names clusters by colors and their most distinctive cards, and assigns tiers by share.

### Demo (`demo/`, deprioritized)

The demo is a FastAPI app plus a static frontend, with `JudgePipeline` reusing `build_pipeline` against `demo/index/`. That index has no Scryfall data and is built by `scripts/build_demo_index.py`, which reuses `build_index.py`'s functions. Known issues remain: the sync pipeline runs in async handlers, and the STT/TTS models load on every request.
