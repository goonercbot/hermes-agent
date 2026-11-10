"""Exercise Hermes client builders against the installed SDK, without network."""
import json
from functools import partial

import anthropic
import httpx
import httpx2
import pytest

from agent import anthropic_adapter, azure_identity_adapter


_BODY = {
    "id": "msg_transport_test", "type": "message", "role": "assistant",
    "model": "claude-sonnet-5", "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


def _reply(request):
    assert json.loads(request.content)["messages"][0]["content"] == "hello"
    return httpx2.Response(200, json=_BODY)


def _message(client):
    return client.messages.create(
        model="claude-sonnet-5", max_tokens=8,
        messages=[{"role": "user", "content": "hello"}],
    )


@pytest.mark.parametrize("key,base,expected", [
    ("test-console-key", "https://api.anthropic.com", "x-api-key"),
    ("sk-ant-oat-test-only", "https://api.anthropic.com", "authorization"),
    ("test-bearer-key", "https://api.minimax.io/anthropic", "authorization"),
])
def test_real_sdk_static_auth_timeout_and_message(monkeypatch, key, base, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-unrelated-env-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-unrelated-env-token")
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", anthropic)
    monkeypatch.setattr(anthropic_adapter, "_get_claude_code_version", lambda: "test")
    seen = []

    def respond(request):
        seen.append(request)
        return _reply(request)

    with anthropic_adapter.build_anthropic_client(key, base_url=base, timeout=17) as original:
        assert isinstance(original.timeout, httpx2.Timeout)
        assert original.timeout.read == 17 and original.timeout.connect == 10
        with original.with_options(http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
            assert _message(client).content[0].text == "ok"
    assert len(seen) == 1
    headers = seen[0].headers
    assert headers[expected] == (key if expected == "x-api-key" else "Bearer " + key)
    assert ("authorization" if expected == "x-api-key" else "x-api-key") not in headers
    assert "test-unrelated-env" not in str(dict(headers))


@pytest.mark.parametrize("tokens,expected", [
    (["token-one", "token-two"], ["Bearer token-one", "Bearer token-two"]),
    (["token-one", ""], ["Bearer token-one", None]),
])
def test_real_sdk_callable_auth_refresh_and_empty_token(monkeypatch, tokens, expected):
    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-unrelated-env-key")
    token_iter = iter(tokens)
    seen = []

    def respond(request):
        seen.append(request)
        return _reply(request) if request.headers.get("authorization") else httpx2.Response(
            401, json={"type": "error", "error": {"type": "authentication_error", "message": "missing auth"}},
        )

    real_builder = azure_identity_adapter.build_bearer_http_client
    monkeypatch.setattr(azure_identity_adapter, "build_bearer_http_client", partial(
        real_builder, transport=httpx2.MockTransport(respond),
    ))
    with anthropic_adapter.build_anthropic_client(
        lambda: next(token_iter), base_url="https://example.openai.azure.com/anthropic", timeout=17,
    ) as client:
        assert isinstance(client._client, httpx2.Client)
        assert _message(client).content[0].text == "ok"
        if expected[-1] is None:
            with pytest.raises(anthropic.AuthenticationError):
                _message(client)
        else:
            assert _message(client).content[0].text == "ok"
    assert [request.headers.get("authorization") for request in seen] == expected
    assert all("x-api-key" not in request.headers for request in seen)
    assert all("entra-id-bearer-via-http-hook" not in str(dict(request.headers)) for request in seen)


def test_generic_bearer_client_keeps_legacy_httpx_default():
    with azure_identity_adapter.build_bearer_http_client(lambda: "test-token") as client:
        assert isinstance(client, httpx.Client)
