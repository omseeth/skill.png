"""SkillOpt's optimizer over Anthropic's Messages API, through the anthropic SDK.

Upstream asks its optimizer through chat_optimizer(system, user) alone, which
the openai_chat route answers from the azure_openai backend. install replaces
that one function with a streamed Messages request: adaptive thinking, the
effort in output_config, no sampling parameters. The key goes to the client
directly and never enters the environment. Each call is appended to the same
call log as OpenRouter's, its cost worked out from PRICES.
"""

from __future__ import annotations

import functools
import time
from datetime import UTC, datetime

from openrouter import PROXY_HINT, CallLog

API = "https://api.anthropic.com"
# The effort levels the Messages API takes.
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# USD per million input and output tokens; the API reports no cost.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0)}


def _client(base_url: str, api_key: str, transport: object | None) -> object:
    """An Anthropic client holding the key in memory only.

    Args:
        base_url: Anthropic's API base URL.
        api_key: The Anthropic key.
        transport: Transport to use instead of the network, for tests.

    Returns:
        The client.
    """
    import anthropic

    http_client = anthropic.DefaultHttpxClient(
        transport=transport,  # type: ignore[arg-type]
        trust_env=transport is None,
    )

    return anthropic.Anthropic(api_key=api_key, base_url=base_url, http_client=http_client)


def preflight(
    base_url: str,
    api_key: str,
    model: str,
    effort: str,
    transport: object | None = None,
):
    """Fail fast unless the endpoint, the key, the model and the effort work.

    Reading the model's record is free and needs a valid key.

    Args:
        base_url: Anthropic's API base URL.
        api_key: The Anthropic key.
        model: The optimizer model id.
        effort: The effort that will be sent.
        transport: Transport to use instead of the network, for tests.
    """
    import anthropic

    if effort not in EFFORTS:
        raise SystemExit(f"the Messages API takes no effort {effort!r}; use one of {EFFORTS}")

    client = _client(base_url, api_key, transport)

    try:
        client.models.retrieve(model)  # type: ignore[attr-defined]
    except anthropic.APIConnectionError as error:
        raise SystemExit(
            f"cannot reach the optimizer endpoint {base_url} ({type(error).__name__}: "
            f"{error}). Compute nodes may have no outbound internet: {PROXY_HINT}."
        ) from error
    except anthropic.AuthenticationError as error:
        raise SystemExit(f"{base_url} refused the API key (HTTP 401)") from error
    except anthropic.NotFoundError as error:
        raise SystemExit(f"{base_url} lists no model {model!r}") from error
    except anthropic.APIStatusError as error:
        raise SystemExit(f"{base_url}/v1/models answered HTTP {error.status_code}") from error


def _usage(model: str, usage: object) -> dict[str, object]:
    """A call's usage under the names CallLog totals, plus the API's own.

    Thinking is billed as output, so completion_tokens includes it; the API
    reports no separate reasoning count.

    Args:
        model: The requested model id, which prices the call.
        usage: The response's usage.

    Returns:
        The API's usage fields with prompt_tokens, completion_tokens,
        total_tokens and cost added.
    """
    raw: dict[str, object] = usage.to_dict()  # type: ignore[attr-defined]
    prompt = int(raw.get("input_tokens") or 0)  # type: ignore[call-overload]
    completion = int(raw.get("output_tokens") or 0)  # type: ignore[call-overload]
    price = PRICES.get(model)
    cost = None if price is None else (prompt * price[0] + completion * price[1]) / 1e6

    return {
        **raw,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cost": cost,
    }


def _call(
    client: object, request: dict[str, object], log: CallLog
) -> tuple[str, dict[str, object]]:
    """Send one request, log it, and return its text.

    Args:
        client: The Anthropic client.
        request: The Messages request's parameters.
        log: Where the call is recorded.

    Returns:
        The reply's text and its usage.
    """
    import anthropic

    system = str(request["system"]).strip()
    output_config: dict[str, object] = request["output_config"]  # type: ignore[assignment]
    entry: dict[str, object] = {
        "time": None,
        "status": None,
        "path": "/v1/messages",
        "latency_s": None,
        "stage": system.splitlines()[0][:120] if system else "",
        "requested_model": request["model"],
        "reasoning_effort": output_config["effort"],
        "max_completion_tokens": request["max_tokens"],
        "generation_id": None,
        "model": None,
        "provider": "anthropic",
        "finish_reason": None,
        "usage": None,
        "error": None,
    }
    started = time.monotonic()

    try:
        with client.messages.stream(**request) as stream:  # type: ignore[attr-defined]
            message = stream.get_final_message()
    except anthropic.APIStatusError as error:
        entry.update(
            time=datetime.now(UTC).isoformat(),
            status=error.status_code,
            latency_s=round(time.monotonic() - started, 3),
            error=str(error.message)[:500],
        )
        log.write(entry)
        raise

    usage = _usage(str(request["model"]), message.usage)
    entry.update(
        time=datetime.now(UTC).isoformat(),
        status=200,
        latency_s=round(time.monotonic() - started, 3),
        generation_id=message.id,
        model=message.model,
        finish_reason=message.stop_reason,
        usage=usage,
    )

    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        entry["error"] = f"refusal: {getattr(details, 'category', None)}"

    log.write(entry)

    if entry["error"] is not None:
        raise RuntimeError(f"the optimizer declined: {entry['error']}")

    text = "".join(block.text for block in message.content if block.type == "text")

    return text, usage


def chat(
    client: object,
    model: str,
    effort: str,
    log: CallLog,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: int | None = None,
) -> tuple[str, dict[str, object]]:
    """Upstream's chat_optimizer, answered by one streamed Messages request.

    Retried as upstream retries its own backend; a refusal is a failed
    attempt.

    Args:
        client: The Anthropic client.
        model: The optimizer model id.
        effort: The run's effort, used when the caller names none.
        log: Where each call is recorded.
        system: The stage's system prompt.
        user: The stage's user message.
        max_completion_tokens: Cap on thinking and text together.
        retries: Attempts before giving up.
        stage: Upstream's stage name, for its token tracker.
        reasoning_effort: An effort for this call only, or None.
        timeout: Seconds to wait for the request, or None for the default.

    Returns:
        The reply's text and its token usage.
    """
    from skillopt.model.azure_openai import tracker

    request: dict[str, object] = {
        "model": model,
        "max_tokens": max_completion_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": reasoning_effort or effort},
    }

    if timeout is not None:
        request["timeout"] = timeout

    last: Exception | None = None

    for attempt in range(retries):
        try:
            text, usage = _call(client, request, log)
            tracker.record(stage, usage["prompt_tokens"], usage["completion_tokens"])  # type: ignore[arg-type]
            return text, usage
        except Exception as error:
            last = error
            time.sleep(min(2**attempt, 30))

    raise RuntimeError(f"LLM call failed after {retries} retries: {last}")


def install(
    base_url: str,
    api_key: str,
    model: str,
    effort: str,
    log: CallLog,
    transport: object | None = None,
):
    """Answer upstream's optimizer calls from model at base_url, logging each.

    Args:
        base_url: Anthropic's API base URL.
        api_key: The Anthropic key; kept in the client's memory only.
        model: The optimizer model id.
        effort: The effort sent with every call.
        log: Where each call is recorded.
        transport: Transport of the anthropic client's HTTP library to use
            instead of the network, for tests.
    """
    from skillopt.model import azure_openai

    client = _client(base_url, api_key, transport)
    # The openai_chat route looks this up on the module at call time.
    azure_openai.chat_optimizer = functools.partial(chat, client, model, effort, log)  # type: ignore[assignment]
