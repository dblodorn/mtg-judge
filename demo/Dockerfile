# --- Stage 1: Build the demo index ---
FROM python:3.11-slim AS index-builder
WORKDIR /app

# Install system deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source + scripts
COPY src/ src/
COPY scripts/ scripts/

# Copy corpus (must be pre-fetched, gitignored but present in build context)
COPY corpus/ corpus/

# Build the demo index (CR+MTR+JAR+IPG, no Scryfall)
# Requires Ollama running — we'll use a separate build step
# For now, just copy if pre-built
RUN if [ -d demo/index ]; then echo "Index exists"; fi

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY src/ src/
COPY demo/ demo/
COPY config/ config/
COPY scripts/ scripts/

# Copy pre-built index (built outside Docker, copied in)
# Or build in container — see docker-compose for the index build step
COPY demo/index/ demo/index/

# Pull Ollama models at build time (so container starts fast)
# Use a smaller model for the demo by default
ARG LLM_MODEL=llama3.1:8b
ARG EMBED_MODEL=nomic-embed-text
RUN /usr/local/bin/ollama serve & \
    sleep 3 && \
    ollama pull ${EMBED_MODEL} && \
    ollama pull ${LLM_MODEL} && \
    kill %1 2>/dev/null; true

ENV LLM_MODEL=${LLM_MODEL}
ENV EMBED_MODEL=${EMBED_MODEL}
ENV OLLAMA_HOST=http://localhost:11434

EXPOSE 8000

# Start script: Ollama + uvicorn
RUN echo '#!/bin/bash\nset -e\nollama serve &\nsleep 2\nexec uvicorn demo.backend.server:app --host 0.0.0.0 --port 8000' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]
