"""Main entry point — wires all components and runs the voice loop."""

import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, ".")


def load_config(path: str = "config/default.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def main():
    config = load_config()

    # Import here so config errors show before heavy imports
    from moonshine_voice import (
        MicTranscriber,
        TextToSpeech,
        get_model_for_language,
    )
    from src.retriever import HybridRetriever
    from src.query_router import QueryRouter
    from src.llm import JudgeLLM
    from src.judge import JudgeListener

    # Moonshine STT
    model_path, model_arch = get_model_for_language(
        config["language"], config["moonshine_model_arch"]
    )
    mic = MicTranscriber(model_path=model_path, model_arch=model_arch)

    # RAG
    retriever = HybridRetriever(
        index_path=config["index_path"],
        k=config["k"],
        rrf_k=config["rrf_k"],
        mmr_lambda=config["mmr_lambda"],
    )

    # Query router
    router = QueryRouter()

    # LLM
    llm = JudgeLLM(model=config["llm_model"])

    # Moonshine TTS
    tts = TextToSpeech(language=config["tts_language"], voice=config["tts_voice"])

    # Wire the judge
    listener = JudgeListener(
        retriever=retriever,
        router=router,
        llm=llm,
        tts=tts,
        k=config["k"],
    )
    mic.add_listener(listener)

    # Run
    mic.start()
    print("🎤 MTG Judge listening — ask a rules question. Ctrl+C to quit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        print("\n👋 Judge stopped.")


if __name__ == "__main__":
    main()
