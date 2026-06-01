# LlamaQueue

A lightweight FastAPI sidecar for `llama.cpp` (`llama-server`) that provides request queuing and unified metrics.

## Purpose

LlamaQueue sits in front of a `llama-server` instance to solve two main issues:
1.  **Sequential Processing:** It ensures that only one inference request (Chat Completions, Completions, etc.) reaches the backend at a time. This prevents crashes and interrupted generations when the server needs to swap models dynamically.
2.  **Unified Monitoring:** It intercepts the `/metrics` endpoint to aggregate Prometheus data from all available models into a single, label-injected stream.

## Basic Usage

LlamaQueue behaves like a transparent wrapper for the `llama-server` API.

### Inference (Queued)
Requests to these endpoints are queued and processed one-by-one:
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /completion`
- `POST /infill`

If the queue is full (default: 100), LlamaQueue returns `503 Service Unavailable`.

### Administrative (Bypass)
These endpoints bypass the queue and return immediately:
- `GET /v1/models`
- `GET /health`
- `GET /metrics`
- All `OPTIONS` (CORS) requests

### Metrics
Scraping `GET /metrics` returns aggregated Prometheus metrics for all models currently managed by the backend. Each metric is injected with a `model="<model_id>"` label for easy filtering in dashboards.

The custom metric `llamaqueue:requests_deferred` tracks the number of clients currently waiting in the queue.

## Quick Start

### Docker Compose (recommended)

The typical deployment pairs LlamaQueue with a `llama-server` container. LlamaQueue is the only exposed service; `llama-server` stays internal to the stack.

```yaml
services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server
    volumes:
      - ./models:/models
    command: >
      -m /models/your-model.gguf
      --host 0.0.0.0
      --port 8080
      --ctx-size 4096
    restart: unless-stopped

  llamaqueue:
    image: ghcr.io/stixes/llamaqueue:1
    environment:
      LLAMA_URL: http://llama-server:8080
    ports:
      - "8000:8000"
    depends_on:
      - llama-server
    restart: unless-stopped
```

Start it with:

```bash
docker compose up -d
```

Your OpenAI-compatible endpoint is then available at `http://localhost:8000`.

### Run directly

```bash
docker run -d \
  -e LLAMA_URL=http://your-llama-server:8080 \
  -p 8000:8000 \
  ghcr.io/stixes/llamaqueue:1
```

## Configuration

The following environment variables can be used to tune the proxy:
- `LLAMA_URL`: URL of the backend `llama-server` (default: `http://llama-server:8080`).
- `MAX_QUEUE_SIZE`: Maximum number of requests allowed to wait in the queue (default: `100`).
- `MAX_BODY_SIZE`: Maximum allowed size for request bodies in bytes (default: `50MB`).

## AI Attribution

This project was designed and implemented with the assistance of [Claude](https://anthropic.com/claude) (Anthropic) via [OpenCode](https://opencode.ai). The code, architecture decisions, and documentation were produced through human-AI collaboration.

If you find a bug or have a concern about the implementation, please [open an issue](https://github.com/stixes/llamaqueue/issues).
