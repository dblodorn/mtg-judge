# MTG Judge

A fully-offline voice assistant that answers Magic: The Gathering rules questions like a judge.

> **Status:** In development — see plan in `.hermes/plans/`

## Quick Start

```bash
git clone https://github.com/dblodorn/mtg-judge.git
cd mtg-judge
make setup
make run
```

Speak a rules question. Get a spoken answer with rule citations.

## Architecture

```
Mic → Moonshine STT → Query Router → Hybrid RAG → Ollama LLM → Moonshine TTS → Speaker
```

All models run locally. No cloud calls. Optimized for Raspberry Pi 5.

See `config/default.yaml` for all configuration options.
