# MTG Judge Demo

A browser-based voice demo of the MTG Judge. Fully offline — all AI runs in the container.

## Live Demo

Frontend: https://dblodorn.github.io/mtg-judge/

To connect to a live backend, enter the backend URL in the config bar (e.g. `http://your-droplet-ip:8000`).

## Run Locally (Docker)

```bash
# 1. Build the index (requires Ollama running locally)
make corpus
python scripts/build_demo_index.py

# 2. Build and run the Docker image
docker build -f demo/Dockerfile -t mtg-judge .
docker run -p 8000:8000 mtg-judge

# 3. Open the demo
open http://localhost:8000
```

## Run Locally (dev mode, no Docker)

```bash
# 1. Start Ollama
ollama serve &

# 2. Build the demo index
make corpus
python scripts/build_demo_index.py

# 3. Start the API
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn demo.backend.server:app --reload --port 8000

# 4. Open http://localhost:8000
```

## Deploy to DigitalOcean

### App Platform (1-click via CLI)
```bash
doctl apps create --spec app-spec.yml
```

### Droplet with Docker
```bash
# Create a 4GB+ droplet, then:
docker run -d -p 80:8000 --restart unless-stopped ghcr.io/dblodorn/mtg-judge:latest
```

## API Endpoints

| Endpoint | Method | Rate Limit | Description |
|----------|--------|------------|-------------|
| `/api/health` | GET | unlimited | Health check |
| `/api/ask` | POST | 10/min | Text Q&A → `{answer, confidence, citations}` |
| `/api/transcribe` | POST | 20/min | Audio file → transcribed text |
| `/api/speak` | POST | 20/min | Text → WAV audio |

### Example

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Does protection from red stop lightning bolt?"}'
```

```json
{
  "answer": "Yes. Protection from red prevents all damage from red sources (CR 702.16e). Lightning Bolt is red, so its damage is prevented.",
  "confidence": "high",
  "citations": ["CR 702.16e"],
  "query_type": "card_interaction",
  "retrieved_chunks": [...]
}
```

## Rate Limiting

IP-based rate limiting protects the CPU-intensive LLM inference:
- 10 questions/minute per IP
- 429 response includes `Retry-After` header
- Frontend shows a friendly "try again" message

For production, add Cloudflare in front for DDoS protection and CDN-level rate limits.

## Requirements

- **Docker:** 4GB+ RAM (Llama 3.1 8B needs ~5GB)
- **Dev mode:** Python 3.11 + Ollama installed locally
- **Corpus:** CR + MTR + JAR + IPG (~2400 chunks, pre-built in image)
