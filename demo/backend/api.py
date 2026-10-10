"""API routes for the MTG Judge demo."""

import io
import json
import sys

sys.path.insert(0, ".")

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from demo.backend.server import limiter

router = APIRouter()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    confidence: str
    citations: list[str]
    query_type: str
    retrieved_chunks: list[dict]


@router.get("/api/health")
async def health():
    """Health check endpoint (no rate limit)."""
    return {"status": "ok"}


@router.post("/api/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, req: AskRequest):
    """Text Q&A: ask a rules question, get a ruling with citations."""
    from demo.backend.pipeline import JudgePipeline

    result = JudgePipeline.get().ask(req.question)
    return result


@router.post("/api/transcribe")
@limiter.limit("20/minute")
async def transcribe(request: Request, audio: UploadFile = File(...)):
    """Accept audio file, return transcribed text.

    Falls back to a simple message if Moonshine isn't available.
    """
    try:
        from moonshine_voice import Transcriber, get_model_for_language
        import numpy as np

        audio_bytes = await audio.read()
        # Decode WAV → numpy float32
        import wave
        wav_io = io.BytesIO(audio_bytes)
        with wave.open(wav_io, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        model_path, model_arch = get_model_for_language("en")
        transcriber = Transcriber(model_path, model_arch)
        text = transcriber.transcribe(audio_array)
        return {"text": text}
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={"error": "moonshine_not_available", "detail": "STT not installed in this demo."},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "transcription_failed", "detail": str(e)},
        )


@router.post("/api/speak")
@limiter.limit("20/minute")
async def speak(request: Request, text: bytes = File(...)):
    """Accept text, return WAV audio.

    Falls back gracefully if TTS isn't available.
    """
    try:
        from moonshine_voice import TextToSpeech

        from src.config import load_config

        config = load_config()
        tts = TextToSpeech().language(config["tts_language"]).voice(config["tts_voice"]).load()
        text_str = text.decode("utf-8")
        wav_bytes = tts.synthesize(text_str)
        return Response(content=wav_bytes, media_type="audio/wav")
    except ImportError:
        return JSONResponse(
            status_code=501,
            content={"error": "tts_not_available", "detail": "TTS not installed in this demo."},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "tts_failed", "detail": str(e)},
        )
