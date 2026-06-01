# Project Context: Llama-Server Inference Proxy

## Overview
This project is a lightweight FastAPI-based reverse proxy acting as a sidecar to a `llama.cpp` (`llama-server`) Docker container. 
Its primary purpose is to serialize (queue) incoming LLM inference requests to prevent the `llama-server` from crashing or aborting generations during dynamic model swapping (e.g., swapping between Gemma and Qwen models).

## Tech Stack
- Python 3.11+
- FastAPI & Uvicorn
- HTTPX (Async HTTP proxying)
- Docker & Docker Compose

## Core Architecture & Mechanisms
1. **The Inference Queue (Mutex Lock):** - Uses `asyncio.Lock()` to process inference calls (e.g., `chat/completions`) strictly sequentially.
   - **CRITICAL:** The lock is acquired and held *inside* the streaming generator. It is only released when the backend `llama-server` has finished streaming the final chunk, or if the client disconnects.
2. **Bypass Routing:** - Fast/administrative endpoints (`/metrics`, `/health`, `/v1/models`, CORS preflight `OPTIONS`) bypass the queue completely to ensure UI and monitoring tools remain responsive.
3. **Metrics Interception & Aggregation:**
   - The `/metrics` endpoint is intercepted to provide a unified view of all models.
   - It fetches metrics for all models listed in `/v1/models` concurrently.
   - **Non-Autoloading:** Uses `autoload=false` to ensure that scraping metrics does not trigger a reload of idle/offloaded models.
   - **Label Injection:** Dynamically injects `model="<model_id>"` labels into the Prometheus output and deduplicates metadata headers.
   - **Queue Tracking:** Overwrites/Appends the internal FastAPI queue count (`queued_requests`) to the `llamacpp:requests_deferred` metric.
4. **Security & Resource Management:**
   - **Queue Depth Limit:** Enforces `MAX_QUEUE_SIZE` (default 100) with 503 responses when full.
   - **Body Size Limit:** Enforces `MAX_BODY_SIZE` (default 50MB) to prevent memory DoS.
   - **Path Sanitization:** Rejects directory traversal attempts (`..`).
   - **Timeouts:** Applies `ADMIN_TIMEOUT` (10s) to non-inference calls to maintain responsiveness.
5. **Client Disconnect Handling:** - Uses `request.is_disconnected()` to drop dead requests *before* and *during* the lock, preventing the backend from loading models for clients that have already timed out.
6. **Log Filtering:** - A custom `logging.Filter` suppresses `/metrics` spam from Uvicorn's access logs.

## Rules for AI Assistant
When assisting with modifications to this codebase, adhere strictly to the following rules:
- **Preserve the Lock:** Never alter the `asyncio.Lock` logic in a way that allows concurrent inference requests to reach the backend.
- **Maintain State Accuracy:** If modifying the queue mechanism, ensure the global `queued_requests` counter is safely decremented in a `finally` block to prevent metric drift.
- **Proxy Transparency:** Always forward raw bytes, query parameters, and headers (sanitizing host/connection) seamlessly unless explicitly instructed to mutate them.
- **Zero Bloat:** Keep the architecture as a minimal, single-file (`proxy.py`) sidecar. Do not suggest external dependencies (Redis, RabbitMQ, Celery) for queueing.
- **Shared Client:** Always use the `app.state.client` (HTTPX) initialized in the lifespan handler to ensure connection pooling.
