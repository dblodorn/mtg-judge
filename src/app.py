"""Main entry point — wires all components and runs the voice loop."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.config import load_config


def build_pipeline(config: dict):
    """Router + retriever + LLM — everything except audio. Shared with `--ask`."""
    from src.llm import JudgeLLM
    from src.query_router import QueryRouter
    from src.retriever import HybridRetriever, IndexMissing

    try:
        retriever = HybridRetriever(
            index_path=config["index_path"],
            k=config["k"],
            rrf_k=config["rrf_k"],
            mmr_lambda=config["mmr_lambda"],
            embed_model=config["embed_model"],
            ollama_host=config["ollama_host"],
            max_parent_chars=config["max_parent_chars"],
            max_context_chars=config["max_context_chars"],
        )
    except IndexMissing as e:
        sys.exit(f"✗ {e}")

    router = QueryRouter(card_names_path=f"{config['index_path'].rstrip('/')}/card_names.json")

    try:
        llm = JudgeLLM(
            model=config["llm_model"],
            host=config["ollama_host"],
            num_ctx=config["num_ctx"],
            think=config.get("think"),
            max_tokens=config.get("max_tokens"),
        )
    except ConnectionError as e:
        sys.exit(f"✗ Cannot reach Ollama at {config['ollama_host']} ({e}). Is `ollama serve` running?")

    return retriever, router, llm


def main():
    parser = argparse.ArgumentParser(description="MTG Judge voice assistant")
    parser.add_argument("--config", help="Config profile (default: $MTG_JUDGE_CONFIG or config/default.yaml)")
    parser.add_argument("--ask", help="Answer one typed question (no mic/TTS) and exit")
    parser.add_argument("--format", help="Game format (modern, legacy, commander…); skips the spoken prompt")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"⚙️  LLM {config['llm_model']} · embed {config['embed_model']} · k={config['k']}")
    version = Path(config["index_path"], "VERSION.json")
    if version.exists():
        v = json.loads(version.read_text())
        print(f"📖 Rules {v.get('cr', '?')} · cards as of {v.get('scryfall_cards', '?')[:10]} · built {v.get('built_at', '?')[:10]}")
    retriever, router, llm = build_pipeline(config)

    if args.ask:
        from src.judge import JudgeListener

        JudgeListener(retriever, router, llm, tts=None, wake_word=None,
                      game_format=args.format or config.get("format")).handle(args.ask)
        return

    # Import here so config/index errors show before heavy audio imports
    from moonshine_voice import MicTranscriber, TextToSpeech, string_to_model_arch

    from src.judge import JudgeListener

    # Moonshine TTS
    tts = (
        TextToSpeech()
        .language(config["tts_language"])
        .voice(config["tts_voice"])
        .output_device(config["output_device"])
        .load()
    )

    # Moonshine STT
    mic = MicTranscriber().language(config["language"]).device(config["input_device"])
    if config["moonshine_model_arch"]:
        mic = mic.model_arch(string_to_model_arch(config["moonshine_model_arch"]))

    listener = JudgeListener(
        retriever=retriever,
        router=router,
        llm=llm,
        tts=tts,
        mic=mic,
        wake_word=config["wake_word"],
        game_format=args.format or config.get("format"),
    )
    mic.add_listener(listener)
    mic.load()

    mic.start()
    wake = f'Say "{config["wake_word"].title()}, …" to' if config["wake_word"] else "Just"
    print(f"🎤 MTG Judge listening — {wake} ask a rules question. Ctrl+C to quit.")
    listener.start()  # asks "Which format are we playing?" unless one is configured
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        mic.close()
        tts.close()
        print("\n👋 Judge stopped.")


if __name__ == "__main__":
    main()
