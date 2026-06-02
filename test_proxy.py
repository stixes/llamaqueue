"""
Acceptance tests for the LlamaQueue proxy.

Uses FastAPI TestClient with respx to mock the backend llama-server.
"""

import os

# Must set before importing proxy
os.environ["LLAMA_URL"] = "http://mock-llama:8080"
os.environ["MAX_QUEUE_SIZE"] = "100"

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

import proxy
from proxy import app


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state between tests."""
    proxy.queued_requests = 0
    yield


@pytest.fixture
def mock_llama():
    """Mock the backend llama-server with respx."""
    with respx.mock(assert_all_called=False) as respx_mock:
        # Default: return a single test model
        respx_mock.get("http://mock-llama:8080/v1/models").respond(
            json={"data": [{"id": "test-model"}]}
        )
        yield respx_mock


@pytest.fixture
def client(mock_llama):
    """Create a TestClient that runs the app lifespan (initializes app.state.client)."""
    with TestClient(app) as c:
        yield c


def test_chat_completion_streams(client, mock_llama):
    """Acceptance test: POST /v1/chat/completions streams back SSE tokens."""
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200,
        content=b"data: Hello\n\ndata: world\n\ndata: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert "data: Hello" in response.text
    assert "data: world" in response.text
    assert "data: [DONE]" in response.text


def test_chat_completion_forwards_headers(client, mock_llama):
    """Verify response headers from backend are forwarded."""
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200,
        content=b"data: test\n\ndata: [DONE]\n\n",
        headers={
            "content-type": "text/event-stream",
            "x-request-id": "abc-123",
        },
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.headers.get("content-type") == "text/event-stream"
    assert response.headers.get("x-request-id") == "abc-123"


def test_bypass_health_check(client, mock_llama):
    """Health endpoint bypasses queue and returns backend response directly."""
    mock_llama.get("http://mock-llama:8080/health").respond(
        status_code=200, json={"status": "ok"}
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_bypass_models_endpoint(client, mock_llama):
    """Models endpoint bypasses queue."""
    mock_llama.get("http://mock-llama:8080/v1/models").respond(
        json={"data": [{"id": "gemma"}, {"id": "qwen"}]}
    )

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert len(response.json()["data"]) == 2


def test_metrics_aggregation(client, mock_llama):
    """Metrics endpoint aggregates and injects model labels."""
    mock_llama.get(
        "http://mock-llama:8080/metrics",
        params={"model": "test-model", "autoload": "false"},
    ).respond(
        status_code=200,
        text="llamacpp:requests_deferred 0\nllamacpp:tokens_per_second 42.0\n",
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert 'model="test-model"' in response.text
    assert "llamacpp:tokens_per_second" in response.text
    assert "llamaqueue:requests_deferred" in response.text


def test_queue_size_limit(client, mock_llama):
    """Returns 503 when queue is full."""
    mock_llama.post("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200, content=b"data: done\n\n"
    )

    # Simulate a full queue by setting the global counter
    proxy.queued_requests = proxy.MAX_QUEUE_SIZE

    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 503
    assert "Queue Full" in response.text


def test_options_bypasses_queue(client, mock_llama):
    """CORS preflight OPTIONS bypasses queue."""
    mock_llama.options("http://mock-llama:8080/v1/chat/completions").respond(
        status_code=200,
        headers={"access-control-allow-origin": "*"},
    )

    response = client.options("/v1/chat/completions")

    assert response.status_code == 200


def test_metrics_include_queue_depth(client, mock_llama):
    """Metrics endpoint includes the internal queue count."""
    mock_llama.get(
        "http://mock-llama:8080/metrics",
        params={"model": "test-model", "autoload": "false"},
    ).respond(status_code=200, text="")

    # Simulate queued requests
    proxy.queued_requests = 5

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "llamaqueue:requests_deferred 5" in response.text
