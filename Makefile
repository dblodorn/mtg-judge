.PHONY: setup setup-venv setup-ollama corpus index update metagame run ask test eval eval-llm eval-hard lint clean sync-pi

# Newest supported interpreter on PATH (project needs 3.11 or 3.12)
PYTHON ?= $(shell command -v python3.12 || command -v python3.11 || echo python3)
PY := .venv/bin/python
# Pi deploy target, e.g. `make sync-pi PI=pi@mtg-judge.local`
PI ?= pi@raspberrypi.local
PI_DIR ?= ~/mtg-judge

setup: setup-venv setup-ollama corpus index

setup-venv:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -e ".[dev]"

setup-ollama:
	bash scripts/install_ollama.sh

corpus:
	$(PY) scripts/fetch_corpus.py
	$(PY) scripts/parse_corpus.py

index:
	$(PY) scripts/build_index.py

# Refresh rules, cards (new sets), rulings, index and metagame; prints what changed
update:
	$(PY) scripts/update.py

# MTGO metagame for one format: make metagame FORMAT=legacy
FORMAT ?= modern
metagame:
	$(PY) scripts/fetch_metagame.py --format $(FORMAT)
	$(PY) scripts/analyze_metagame.py --format $(FORMAT)

run:
	$(PY) -m src.app

# Text-only smoke test, no mic/speaker: make ask Q="does trample go through protection?"
ask:
	$(PY) -m src.app --ask "$(Q)"

test:
	.venv/bin/pytest -m "not integration"

eval:
	.venv/bin/pytest tests/test_evaluation.py -v -m integration

# End-to-end (retrieval + LLM) accuracy and latency for the active profile
eval-llm:
	$(PY) scripts/eval_llm.py

# Tricky interactions (Urza's Saga, layers, replacement order) graded on the answer itself
eval-hard:
	$(PY) scripts/eval_llm.py --cases tests/fixtures/hard_cases.json

lint:
	.venv/bin/ruff check src scripts tests

# Copy code + a Mac-built index to the Pi (embedding on the Pi takes hours)
sync-pi:
	rsync -az --delete --exclude .venv --exclude corpus --exclude .git --exclude __pycache__ ./ $(PI):$(PI_DIR)/

clean:
	rm -rf __pycache__ .pytest_cache index/
