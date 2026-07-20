"""FastAPI server for the MTG Judge demo."""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="MTG Judge API", version="0.1.0")
app.state.limiter = limiter

# CORS — allow GitHub Pages origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only; restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "detail": f"Rate limit exceeded. Try again in a minute.",
            "retry_after": 60,
        },
        headers={"Retry-After": "60"},
    )


# Import and include routes after app setup
from demo.backend.api import router

app.include_router(router)


# Serve frontend static files if they exist (for local dev)
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


@app.on_event("startup")
async def startup():
    """Pre-warm the pipeline on startup."""
    print("🚀 Starting MTG Judge API...")
    from demo.backend.pipeline import JudgePipeline

    JudgePipeline.get()
    print("✅ Pipeline ready.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
