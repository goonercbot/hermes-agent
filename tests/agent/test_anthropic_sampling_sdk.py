"""Sampling must survive SDK 2.x removing native sampling keywords.

The vendor Messages wire still accepts them. Exercise the real SDK, not a
permissive mock that accepts every keyword and hides a local TypeError.
"""
import copy
import json

import anthropic
import httpx2
import pytest

from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message


def _sse(kind, **fields):
    return (f"event: {kind}\ndata: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


def _reply():
    return {"id": "msg_fixture", "type": "message", "role": "assistant", "model": "MiniMax-M3",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}


def _stream():
    start = dict(_reply(), content=[], stop_reason=None)
    return b"".join([
        _sse("message_start", message=start),
        _sse("content_block_start", index=0, content_block={"type": "text", "text": ""}),
        _sse("content_block_delta", index=0, delta={"type": "text_delta", "text": "ok"}),
        _sse("content_block_stop", index=0),
        _sse("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage={"output_tokens": 1}),
        _sse("message_stop"),
    ])


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("model,base", [("MiniMax-M3", "https://api.minimax.io/anthropic"),
                                        ("claude-sonnet-4-20250514", "https://api.anthropic.com")])
def test_real_sdk_preserves_manual_thinking_temperature(streaming, model, base):
    seen = []

    def respond(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["temperature"] == 1
        assert body["thinking"]["type"] == "enabled"
        if streaming:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_stream())
        return httpx2.Response(200, json=_reply())

    kwargs = build_anthropic_kwargs(model, [{"role": "user", "content": "hello"}], None, 100,
                                    {"enabled": True, "effort": "low"}, base_url=base)
    before = copy.deepcopy(kwargs)
    with anthropic.Anthropic(api_key="synthetic-test-key", base_url=base, max_retries=0,
                             http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        result = create_anthropic_message(client, kwargs, prefer_stream=streaming)
        assert result.content[0].text == "ok"
    assert len(seen) == 1
    assert kwargs == before


@pytest.mark.parametrize("streaming", [True, False])
def test_real_sdk_preserves_sampling_extra_body_precedence(streaming):
    seen = []

    def respond(request):
        seen.append(json.loads(request.content))
        if streaming:
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_stream())
        return httpx2.Response(200, json=_reply())

    kwargs = {"model": "MiniMax-M3", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 16,
              "temperature": 1, "top_p": 0.9, "top_k": 20,
              "extra_body": {"temperature": 0.8, "vendor_setting": "preserve"}}
    before = copy.deepcopy(kwargs)
    with anthropic.Anthropic(api_key="synthetic-test-key", max_retries=0,
                             http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        assert create_anthropic_message(client, kwargs, prefer_stream=streaming).content[0].text == "ok"
    assert seen[0]["temperature"] == 0.8
    assert seen[0]["top_p"] == 0.9 and seen[0]["top_k"] == 20
    assert seen[0]["vendor_setting"] == "preserve"
    assert kwargs == before


def test_legacy_sdk_keeps_native_sampling_arguments():
    from agent.anthropic_adapter import anthropic_sdk_kwargs

    def legacy(*, temperature=None, top_p=None, top_k=None, extra_body=None):
        pass

    kwargs = {"temperature": 1, "top_p": 0.9, "top_k": 20, "extra_body": {"other": True}}
    before = copy.deepcopy(kwargs)
    assert anthropic_sdk_kwargs(kwargs, legacy) == kwargs
    assert kwargs == before


def test_new_sdk_only_moves_sampling_not_arbitrary_unknown_fields():
    from agent.anthropic_adapter import anthropic_sdk_kwargs

    def modern(*, extra_body=None):
        pass

    kwargs = {"temperature": 1, "unknown_field": "must still fail"}
    result = anthropic_sdk_kwargs(kwargs, modern)
    assert result == {"unknown_field": "must still fail", "extra_body": {"temperature": 1}}
    assert kwargs == {"temperature": 1, "unknown_field": "must still fail"}
