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

## Configuration

The following environment variables can be used to tune the proxy:
- `LLAMA_URL`: URL of the backend `llama-server` (default: `http://llama-server:8080`).
- `MAX_QUEUE_SIZE`: Maximum number of requests allowed to wait in the queue (default: `100`).
- `MAX_BODY_SIZE`: Maximum allowed size for request bodies in bytes (default: `50MB`).
