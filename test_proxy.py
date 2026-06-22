"""
Acceptance tests for the LlamaQueue proxy.

Uses FastAPI TestClient with respx to mock the backend llama-server.
"""

import os

os.environ["LLAMA_URL"] = "http://mock-llama:8080"
os.environ["MAX_QUEUE_SIZE"] = "100"
os.environ["MAX_CONCURRENT_INFERENCE"] = "4"

import asyncio
import json
import threading

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

import proxy
from proxy import app


@pytest.fixture(autouse=True)
def reset_globals():
    proxy.queued_requests = 0
    proxy.active_model = None
    proxy.active_inference_count = 0
    proxy.MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "4"))
    proxy.known_models_cache.clear()
    yield


@pytest.fixture
def mock_llama():
    with respx.mock(assert_all_called=False) as respx_mock:
        respx_mock.get("http://mock-llama:8080/v1/models").respond(
            json={"data": [{"id": "gemma"}, {"id": "qwen"}]}
        )
        yield respx_mock


@pytest.fixture
def client(mock_llama):
    with TestClient(app) as c:
        yield c


def make_chat_request(model="gemma"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }


# ─── Basic tests (TestClient, synchronous) ────────────────────────────────────

def test_chat_completion_streams(client, mock_llama):
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200,
        content=b"data: Hello\n\ndata: world\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert "data: Hello" in response.text
    assert "data: world" in response.text
    assert "data: [DONE]" in response.text


def test_chat_completion_forwards_headers(client, mock_llama):
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200,
        content=b"data: test\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream", "x-request-id": "abc-123"},
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.headers.get("content-type") == "text/event-stream"
    assert response.headers.get("x-request-id") == "abc-123"


def test_bypass_health_check(client, mock_llama):
    mock_llama.get("http://mock-llama:8080/health").respond(
        status_code=200, json={"status": "ok"}
    )
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_bypass_models_endpoint(client, mock_llama):
    mock_llama.get("http://mock-llama:8080/v1/models").respond(
        json={"data": [{"id": "gemma"}, {"id": "qwen"}]}
    )
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 2


def test_metrics_aggregation(client, mock_llama):
    mock_llama.get(
        "http://mock-llama:8080/metrics",
        params={"model": "gemma", "autoload": "false"},
    ).respond(
        status_code=200,
        text="llamacpp:requests_deferred 0\nllamacpp:tokens_per_second 42.0\n",
    )
    mock_llama.get(
        "http://mock-llama:8080/metrics",
        params={"model": "qwen", "autoload": "false"},
    ).respond(status_code=200, text="llamacpp:tokens_per_second 10.0\n")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert 'model="gemma"' in response.text
    assert 'tokens_per_second{model="gemma"} 42.0' in response.text


def test_queue_size_limit(client, mock_llama):
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200, content=b"data: done\n\n"
    )
    proxy.queued_requests = proxy.MAX_QUEUE_SIZE
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 503
    assert "Queue Full" in response.text


def test_options_bypasses_queue(client, mock_llama):
    mock_llama.options("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200, headers={"access-control-allow-origin": "*"},
    )
    response = client.options("/v1/chat/completions")
    assert response.status_code == 200


def test_metrics_include_queue_depth(client, mock_llama):
    mock_llama.get(
        "http://mock-llama:8080/metrics",
        params={"model": "gemma", "autoload": "false"},
    ).respond(status_code=200, text="")
    mock_llama.get(
        "http://mock-llama:8080/metrics",
        params={"model": "qwen", "autoload": "false"},
    ).respond(status_code=200, text="")
    proxy.queued_requests = 5
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llamaqueue:requests_deferred 5" in response.text


def test_backend_read_error_returns_error_json(client, mock_llama):
    from httpx import ReadError

    mock_llama.post("http://mock-llama:8080/v1/chat/completions").mock(
        side_effect=ReadError("Connection lost")
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert "Backend connection lost during streaming" in response.text
    assert "read_error" in response.text


def test_backend_remote_protocol_error_returns_error_json(client, mock_llama):
    from httpx import RemoteProtocolError

    mock_llama.post("http://mock-llama:8080/v1/chat/completions").mock(
        side_effect=RemoteProtocolError("Bad protocol")
    )
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gemma", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert "Backend connection lost during streaming" in response.text


# ─── Model validation tests ───────────────────────────────────────────────────

def test_unknown_model_rejected(client, mock_llama):
    response = client.post(
        "/v1/chat/completions",
        json=make_chat_request("nonexistent-model-xyz"),
    )
    assert response.status_code == 400
    assert "Unknown or missing model" in response.text


def test_missing_model_field_rejected(client, mock_llama):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert "Unknown or missing model" in response.text


def test_model_cache_refreshed_from_models_endpoint(client, mock_llama):
    mock_llama.get("http://mock-llama:8080/v1/models").respond(
        json={"data": [{"id": "gemma"}, {"id": "qwen"}, {"id": "new-model"}]}
    )
    proxy.known_models_cache.clear()
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200, content=b"data: ok\n\n"
    )
    resp = client.post("/v1/chat/completions", json=make_chat_request("new-model"))
    assert resp.status_code == 200


def test_model_cache_fallback_fetch(client, mock_llama):
    proxy.known_models_cache.clear()
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200, content=b"data: ok\n\n"
    )
    resp = client.post("/v1/chat/completions", json=make_chat_request("gemma"))
    assert resp.status_code == 200
    assert "gemma" in proxy.known_models_cache


def test_slot_released_after_stream_completes(client, mock_llama):
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200, content=b"data: done\n\n"
    )
    resp = client.post("/v1/chat/completions", json=make_chat_request("gemma"))
    assert resp.status_code == 200
    assert proxy.active_model is None
    assert proxy.active_inference_count == 0


def test_slot_released_after_error(client, mock_llama):
    from httpx import ReadError

    mock_llama.post("http://mock-llama:8080/v1/chat/completions").mock(
        side_effect=ReadError("boom")
    )
    resp = client.post("/v1/chat/completions", json=make_chat_request("gemma"))
    assert resp.status_code == 200
    assert proxy.active_model is None
    assert proxy.active_inference_count == 0


# ─── Concurrent inference tests ───────────────────────────────────────────────


class ThrottleStream:
    """Async iterable yielding one chunk then blocking on a threading.Event.

    Uses run_in_executor to bridge the async iterator (running on
    whichever event loop TestClient uses in its thread) with
    threading.Event for cross-thread coordination."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.started.is_set():
            self.started.set()
            return b"data: start\n\n"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.release.wait)
        raise StopAsyncIteration


def test_same_model_parallel(client, mock_llama):
    """Two requests for the same model acquire slots concurrently."""
    stream_a = ThrottleStream()
    stream_b = ThrottleStream()
    assigned = []

    def handler(request):
        if not assigned:
            assigned.append("a")
            return Response(status_code=200, headers={"content-type": "text/event-stream"}, stream=stream_a)
        assigned.append("b")
        return Response(status_code=200, headers={"content-type": "text/event-stream"}, stream=stream_b)

    mock_llama.post("http://mock-llama:8080/v1/chat/completions").mock(side_effect=handler)

    def post():
        client.post("/v1/chat/completions", json=make_chat_request("gemma"))

    t_a = threading.Thread(target=post, daemon=True)
    t_a.start()
    assert stream_a.started.wait(timeout=5)

    t_b = threading.Thread(target=post, daemon=True)
    t_b.start()
    assert stream_b.started.wait(timeout=5)

    stream_a.release.set()
    stream_b.release.set()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert len(assigned) == 2


def test_different_model_queues_serial(client, mock_llama):
    """Request for model B waits until all model A requests finish."""
    stream_a = ThrottleStream()
    stream_b = ThrottleStream()
    backend_calls = []

    def handler(request):
        body = json.loads(request.content)
        model = body.get("model")
        backend_calls.append(model)
        if model == "gemma":
            return Response(status_code=200, headers={"content-type": "text/event-stream"}, stream=stream_a)
        return Response(status_code=200, headers={"content-type": "text/event-stream"}, stream=stream_b)

    mock_llama.post("http://mock-llama:8080/v1/chat/completions").mock(side_effect=handler)

    def post_gemma():
        client.post("/v1/chat/completions", json=make_chat_request("gemma"))

    t = threading.Thread(target=post_gemma, daemon=True)
    t.start()
    assert stream_a.started.wait(timeout=5)

    def post_qwen():
        client.post("/v1/chat/completions", json=make_chat_request("qwen"))

    t2 = threading.Thread(target=post_qwen, daemon=True)
    t2.start()

    # Qwen should NOT have started yet
    assert not stream_b.started.wait(timeout=0.5)

    # Release gemma → qwen gets the slot
    stream_a.release.set()
    assert stream_b.started.wait(timeout=5)
    stream_b.release.set()
    t.join(timeout=5)
    t2.join(timeout=5)

    assert backend_calls == ["gemma", "qwen"]


def test_max_concurrent_enforced(client, mock_llama):
    """When MAX_CONCURRENT_INFERENCE slots are fully used, same-model
    requests must wait in the condition queue."""
    cap = 2
    proxy.MAX_CONCURRENT_INFERENCE = cap

    streams = [ThrottleStream() for _ in range(cap + 1)]
    assigned = []

    def handler(request):
        assigned.append(len(assigned))
        return Response(status_code=200, headers={"content-type": "text/event-stream"}, stream=streams[len(assigned) - 1])

    mock_llama.post("http://mock-llama:8080/v1/chat/completions").mock(side_effect=handler)

    def post():
        client.post("/v1/chat/completions", json=make_chat_request("gemma"))

    threads = []
    for i in range(cap):
        t = threading.Thread(target=post, daemon=True)
        t.start()
        threads.append(t)
        assert streams[i].started.wait(timeout=5)

    overflow_done = threading.Event()

    def post_overflow():
        client.post("/v1/chat/completions", json=make_chat_request("gemma"))
        overflow_done.set()

    t3 = threading.Thread(target=post_overflow, daemon=True)
    t3.start()

    # Overflow should wait
    assert not overflow_done.wait(timeout=0.5)

    # Release one slot
    streams[0].release.set()
    # Overflow acquires the freed slot and starts
    assert streams[cap].started.wait(timeout=5)
    # Release overflow
    streams[cap].release.set()
    assert overflow_done.wait(timeout=5)

    for s in streams[1:cap]:
        s.release.set()
    for t in threads[1:]:
        t.join(timeout=5)


# ─── Current model endpoint tests ─────────────────────────────────────────────

def test_current_model_loaded(client, mock_llama):
    mock_llama.get("http://mock-llama:8080/metrics").respond(
        status_code=200,
        text='llamacpp:loaded_model{model="gemma"} 1\nllamacpp:tokens_per_second 42.0\n',
    )
    response = client.get("/v1/model")
    assert response.status_code == 200
    assert response.json() == {"model": "gemma"}


def test_current_model_unloaded(client, mock_llama):
    mock_llama.get("http://mock-llama:8080/metrics").respond(
        status_code=200,
        text='llamacpp:tokens_per_second 0.0\n',
    )
    response = client.get("/v1/model")
    assert response.status_code == 200
    assert response.json() == {"model": None}


def test_current_model_backend_unreachable(client, mock_llama):
    mock_llama.get("http://mock-llama:8080/metrics").mock(
        side_effect=Exception("connection refused")
    )
    response = client.get("/v1/model")
    assert response.status_code == 200
    assert response.json() == {"model": None}
