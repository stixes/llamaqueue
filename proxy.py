import asyncio
import httpx
import json
import os
import logging
import re
import threading
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, Response
from contextlib import asynccontextmanager

# Configuration
LLAMA_URL = os.getenv("LLAMA_URL", "http://llama-server:8080")
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))
MAX_BODY_SIZE = int(os.getenv("MAX_BODY_SIZE", str(1024 * 1024 * 50))) # 50MB
MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "4"))
ADMIN_TIMEOUT = 10.0

# Whitelist: ONLY these paths are subject to the inference queue.
QUEUE_PATHS = {
    "chat/completions",
    "v1/chat/completions",
    "v1/completions",
    "completion",
    "infill"
}

# Logging configuration to suppress noisy logs from metrics scraping
class MetricsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/metrics" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(MetricsFilter())

# Global state for the inference queue
inference_cv = threading.Condition()
active_model: str | None = None
active_inference_count = 0
queued_requests = 0

# Model validation cache
known_models_cache: set = set()
model_cache_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize a shared HTTPX client for connection pooling and reuse
    async with httpx.AsyncClient(timeout=None) as client:
        app.state.client = client
        yield

class HeaderAwareStreamingResponse(StreamingResponse):
    """StreamingResponse that captures backend headers from the first yield
    before sending the http.response.start ASGI message."""

    async def __call__(self, scope, receive, send):
        ait = self.body_iterator.__aiter__()
        try:
            first = await ait.__anext__()
        except StopAsyncIteration:
            first = b""

        if isinstance(first, dict):
            for k, v in first.items():
                self.headers[k] = v
            try:
                chunk = await ait.__anext__()
            except StopAsyncIteration:
                chunk = b""
        else:
            chunk = first

        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        })
        if chunk:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
        async for c in ait:
            await send({"type": "http.response.body", "body": c, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

app = FastAPI(lifespan=lifespan)

def get_forward_headers(request: Request) -> dict:
    """Prepare headers for forwarding, removing hop-by-hop and host headers."""
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("connection", None)
    return headers

def extract_model_from_body(body: bytes) -> str | None:
    """Extract the model identifier from a JSON request body."""
    try:
        data = json.loads(body)
        return data.get("model")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

async def refresh_model_cache(client: httpx.AsyncClient) -> set:
    """Fetch available models from the backend and update the cache."""
    global known_models_cache
    try:
        resp = await client.get(f"{LLAMA_URL}/v1/models", timeout=ADMIN_TIMEOUT)
        resp.raise_for_status()
        known_models_cache = {m["id"] for m in resp.json().get("data", [])}
    except Exception:
        pass
    return known_models_cache

async def resolve_and_validate_model(request: Request, body: bytes) -> str | None:
    """Extract model from body and validate against known models.
    Returns the model_id if valid, None if invalid (caller should reject)."""
    model_id = extract_model_from_body(body)
    if model_id is None:
        return None

    client = request.app.state.client

    async with model_cache_lock:
        if model_id not in known_models_cache:
            await refresh_model_cache(client)

    if model_id not in known_models_cache:
        return None

    return model_id

async def acquire_inference_slot(request: Request, model_id: str) -> str | None:
    """Try to acquire an inference slot for the given model.
    Blocks (offloading to thread executor) until a slot is available or
    the client disconnects.
    Returns model_id if acquired, None if disconnected while waiting."""
    global active_model, active_inference_count
    loop = asyncio.get_running_loop()

    def _try_once() -> bool:
        """Single pass: return True if acquired, False if must wait.
        Held for at most 0.5s so the async loop can periodically check
        client disconnect."""
        global active_model, active_inference_count
        with inference_cv:
            if active_model is None:
                active_model = model_id
                active_inference_count = 1
                return True
            if active_model == model_id and active_inference_count < MAX_CONCURRENT_INFERENCE:
                active_inference_count += 1
                return True
            inference_cv.wait(timeout=0.5)
            return False

    while True:
        if await request.is_disconnected():
            return None
        acquired = await loop.run_in_executor(None, _try_once)
        if acquired:
            return model_id

async def release_inference_slot():
    """Release an inference slot and wake waiting requests."""
    global active_model, active_inference_count
    loop = asyncio.get_running_loop()

    def _release():
        global active_model, active_inference_count
        with inference_cv:
            active_inference_count -= 1
            if active_inference_count == 0:
                active_model = None
            inference_cv.notify_all()

    await loop.run_in_executor(None, _release)

async def stream_backend(method: str, url: str, content: bytes, request: Request, model_id: str):
    """Generator for streaming backend responses while holding an inference slot.
    Yields a dict of headers first, then raw byte chunks."""
    global queued_requests
    queued_requests += 1
    in_queue = True

    try:
        result = await acquire_inference_slot(request, model_id)
        queued_requests -= 1
        in_queue = False

        if result is None:
            yield {}
            return

        if await request.is_disconnected():
            await release_inference_slot()
            yield {}
            return

        client = request.app.state.client
        headers = get_forward_headers(request)

        try:
            async with client.stream(method, url, content=content, headers=headers, timeout=None) as response:
                fwd = {k: v for k, v in response.headers.items()
                       if k.lower() not in {"transfer-encoding", "connection", "content-length"}}
                yield fwd

                async for chunk in response.aiter_bytes():
                    if await request.is_disconnected():
                        break
                    yield chunk
        except (httpx.ReadError, httpx.RemoteProtocolError) as e:
            yield {"content-type": "application/json"}
            yield b'{"error":{"message":"Backend connection lost during streaming","type":"read_error"}}'
    finally:
        if in_queue:
            queued_requests -= 1
        else:
            await release_inference_slot()

async def handle_metrics(request: Request):
    """Intercept metrics endpoint to aggregate results from all models and inject labels."""
    client = request.app.state.client

    # 1. Fetch available models to know which metrics to scrape
    try:
        models_resp = await client.get(f"{LLAMA_URL}/v1/models", timeout=ADMIN_TIMEOUT)
        models_resp.raise_for_status()
        model_ids = [m["id"] for m in models_resp.json().get("data", [])]
    except Exception as e:
        return Response(content=f"# Error fetching models: {e}", status_code=502)

    # 2. Fetch metrics for all models concurrently
    async def fetch_model_metrics(model_id):
        try:
            # autoload=false prevents triggering model loads during metric scraping
            res = await client.get(f"{LLAMA_URL}/metrics?model={model_id}&autoload=false", timeout=ADMIN_TIMEOUT)
            if res.status_code != 200:
                return model_id, ""
            return model_id, res.text
        except httpx.HTTPError:
            return model_id, ""

    metrics_results = await asyncio.gather(*(fetch_model_metrics(mid) for mid in model_ids))

    # 3. Process and aggregate metrics with injected model labels
    output_lines = [
        "# HELP llamaqueue:requests_deferred Number of requests waiting in the LlamaQueue",
        "# TYPE llamaqueue:requests_deferred gauge",
        f'llamaqueue:requests_deferred {queued_requests}'
    ]
    seen_headers = set()

    for model_id, raw_text in metrics_results:
        # Simple escape for model_id to prevent label injection
        escaped_model_id = model_id.replace('"', '\\"')

        for line in raw_text.splitlines():
            if not line or line.isspace():
                continue

            if line.startswith("#"):
                if line not in seen_headers:
                    output_lines.append(line)
                    seen_headers.add(line)
                continue

            # Inject model="<id>" label into Prometheus metrics
            match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*?\})?\s+(.*)$', line)
            if match:
                name, labels, value = match.groups()
                if labels:
                    new_labels = labels[:-1] + f',model="{escaped_model_id}"}}'
                else:
                    new_labels = f'{{model="{escaped_model_id}"}}'
                output_lines.append(f"{name}{new_labels} {value}")

    return Response(content="\n".join(output_lines) + "\n", media_type="text/plain")

async def handle_current_model(request: Request):
    """Query llama-server for the currently loaded model via /v1/models."""
    client = request.app.state.client
    try:
        resp = await client.get(f"{LLAMA_URL}/v1/models", timeout=ADMIN_TIMEOUT)
        if resp.status_code == 200:
            for model in resp.json().get("data", []):
                if model.get("status", {}).get("value") == "loaded":
                    return Response(
                        content=json.dumps({"model": model["id"]}),
                        media_type="application/json",
                    )
    except Exception:
        pass
    return Response(
        content=json.dumps({"model": None}),
        media_type="application/json",
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    # Security: Normalize path and prevent directory traversal (SSRF-lite)
    if ".." in path or path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid path")

    method = request.method
    query = request.url.query
    url = f"{LLAMA_URL}/{path}?{query}" if query else f"{LLAMA_URL}/{path}"
    clean_path = path.strip("/")

    # 0. Current model status: return proxy's tracked active model
    if method == "GET" and clean_path == "v1/model":
        return await handle_current_model(request)

    # 1. Inference calls: Sent to the queue
    if method != "OPTIONS" and clean_path in QUEUE_PATHS:
        # Security: Rate limit the queue depth to prevent DoS
        if queued_requests >= MAX_QUEUE_SIZE:
            return Response(content="Service Busy: Queue Full", status_code=503)

        # Security: Enforce max body size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE:
            raise HTTPException(status_code=413, detail="Request entity too large")

        body = await request.body()
        model_id = await resolve_and_validate_model(request, body)
        if model_id is None:
            raise HTTPException(status_code=400, detail="Unknown or missing model")

        return HeaderAwareStreamingResponse(stream_backend(method, url, body, request, model_id))

    # 2. Metrics interception: Aggregate and inject model labels
    if method == "GET" and clean_path == "metrics":
        return await handle_metrics(request)

    # 3. All other calls (models, health, props): Pass through directly without locking
    client = request.app.state.client
    headers = get_forward_headers(request)

    try:
        # Check body size for non-inference calls too
        body = await request.body()
        if len(body) > MAX_BODY_SIZE:
             raise HTTPException(status_code=413, detail="Request entity too large")

        resp = await client.request(method, url, content=body, headers=headers, timeout=ADMIN_TIMEOUT)

        # Also refresh model cache when /v1/models is queried
        if method == "GET" and clean_path == "v1/models":
            try:
                data = resp.json()
                async with model_cache_lock:
                    known_models_cache.clear()
                    known_models_cache.update(m["id"] for m in data.get("data", []))
            except Exception:
                pass

        # Filter response headers to avoid proxy-related conflicts
        response_headers = {k: v for k, v in resp.headers.items()
                            if k.lower() not in {"transfer-encoding", "connection"}}
        return Response(content=resp.content, status_code=resp.status_code, headers=response_headers)
    except httpx.HTTPError as e:
        return Response(content=f"Proxy error: {e}", status_code=502)
