.PHONY: setup setup-venv setup-ollama corpus index run test eval clean

setup: setup-venv setup-ollama corpus index

setup-venv:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

setup-ollama:
	bash scripts/install_ollama.sh

corpus:
	.venv/bin/python scripts/fetch_corpus.py
	.venv/bin/python scripts/parse_corpus.py

index:
	.venv/bin/python scripts/build_index.py

run:
	.venv/bin/python -m src.app

test:
	.venv/bin/pytest -v

eval:
	.venv/bin/pytest tests/test_evaluation.py -v

clean:
	rm -rf __pycache__ .pytest_cache index/
