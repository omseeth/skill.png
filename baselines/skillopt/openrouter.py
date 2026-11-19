"""SkillOpt's optimizer over OpenRouter, through SkillOpt's own request path.

Upstream's azure_openai backend in its openai_compatible auth mode builds a
plain OpenAI client on any base URL and sends reasoning_effort as a top-level
field, which OpenRouter accepts. That path is used as it is. Two things are
added around it: the API key is handed to the backend directly, so it never
enters the trainer's config or the environment, and the client gets an
HTTP client whose response hook appends each call's usage and cost to a log.
The request bytes are upstream's.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

API = "https://openrouter.ai/api/v1"
# How proxies are configured for both endpoints; quoted in error messages.
PROXY_HINT = (
    "HTTPS_PROXY (or ALL_PROXY) routes the optimizer through a proxy, and NO_PROXY "
    "must then name the vLLM host, e.g. NO_PROXY=localhost,127.0.0.1"
)


class CallLog:
    """One JSON line per optimizer call: usage, cost and what was asked for.

    Headers are never read, so neither the key nor anything derived from it
    is written.
    """

    def __init__(self, path: Path):
        """Append to path, creating its directory.

        Args:
            path: The JSONL file calls are appended to.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._started: dict[int, float] = {}

    def stamp(self, request: object):
        """Note when a request was sent; a request event hook.

        Args:
            request: The outgoing request.
        """
        with self._lock:
            self._started[id(request)] = time.monotonic()

    def record(self, response: object):
        """Append one call to the log; a response event hook.

        Args:
            response: The response, of the HTTP library the client uses.
        """
        response.read()  # type: ignore[attr-defined]
        request = response.request  # type: ignore[attr-defined]

        with self._lock:
            started = self._started.pop(id(request), None)

        body = _json(request.content)
        payload = _json(response.content)  # type: ignore[attr-defined]
        status = int(response.status_code)  # type: ignore[attr-defined]
        choices = payload.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else {}
        error = payload.get("error")

        if status != 200 and error is None:
            error = response.text[:500]  # type: ignore[attr-defined]

        entry = {
            "time": datetime.now(UTC).isoformat(),
            "status": status,
            "path": str(request.url.path),
            "latency_s": None if started is None else round(time.monotonic() - started, 3),
            "stage": _stage(body),
            "requested_model": body.get("model"),
            "reasoning_effort": body.get("reasoning_effort"),
            "max_completion_tokens": body.get("max_completion_tokens"),
            "generation_id": payload.get("id"),
            "model": payload.get("model"),
            "provider": payload.get("provider"),
            "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
            "usage": payload.get("usage"),
            "error": error,
        }
        self.write(entry)

    def write(self, entry: dict[str, object]):
        """Append one entry, as record builds it, to the log.

        Args:
            entry: One call's fields.
        """
        with self._lock, self.path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def totals(self) -> dict[str, object]:
        """Sum the log: calls, failures, tokens and cost, also per stage.

        Args:
            None.

        Returns:
            The totals, with a per-stage breakdown under stages.
        """
        overall = _empty_totals()
        stages: dict[str, dict[str, object]] = {}

        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue

                entry = json.loads(line)
                stage = str(entry.get("stage") or "")

                for bucket in (overall, stages.setdefault(stage, _empty_totals())):
                    _add(bucket, entry)

        return {**overall, "stages": stages}


def _empty_totals() -> dict[str, object]:
    """Zeroed counters for one bucket of the call log.

    Args:
        None.

    Returns:
        Counters for calls, failures, tokens and cost.
    """
    return {
        "calls": 0,
        "failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
    }


def _add(bucket: dict[str, object], entry: dict[str, object]):
    """Add one logged call to a bucket of counters.

    Args:
        bucket: Counters from _empty_totals.
        entry: One line of the call log.
    """
    usage = entry.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    failed = entry.get("status") != 200 or entry.get("error") is not None
    bucket["calls"] = int(bucket["calls"]) + 1  # type: ignore[call-overload]
    bucket["failed"] = int(bucket["failed"]) + int(failed)  # type: ignore[call-overload]

    for key, value in (
        ("prompt_tokens", usage.get("prompt_tokens")),
        ("completion_tokens", usage.get("completion_tokens")),
        ("reasoning_tokens", details.get("reasoning_tokens")),
    ):
        bucket[key] = int(bucket[key]) + int(value or 0)  # type: ignore[call-overload]

    bucket["cost_usd"] = float(bucket["cost_usd"]) + float(usage.get("cost") or 0.0)  # type: ignore[arg-type]


def _json(raw: bytes) -> dict[str, object]:
    """Decode a JSON object body, or nothing.

    Args:
        raw: The body bytes.

    Returns:
        The object, or an empty dict when the body is not one.
    """
    try:
        value = json.loads(raw)
    except ValueError:
        return {}

    return value if isinstance(value, dict) else {}


def _stage(body: dict[str, object]) -> str:
    """Name the optimizer stage by the first line of its system prompt.

    Args:
        body: The decoded request body.

    Returns:
        The first line of the system message, shortened.
    """
    messages = body.get("messages")

    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict) and message.get("role") == "system":
            text = str(message.get("content") or "").strip()
            return text.splitlines()[0][:120] if text else ""

    return ""


def preflight(
    base_url: str,
    api_key: str,
    model: str,
    effort: str | None,
    transport: httpx.BaseTransport | None = None,
):
    """Fail fast unless the optimizer endpoint, the key and the model work.

    Both requests are free: one reads the key's own record, one lists the
    models with their supported parameters.

    Args:
        base_url: OpenRouter's API base URL.
        api_key: The OpenRouter key.
        model: The optimizer model id.
        effort: The reasoning effort that will be sent, or None.
        transport: Transport to use instead of the network, for tests.
    """
    base = base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}

    with httpx.Client(timeout=30.0, transport=transport, trust_env=transport is None) as client:
        try:
            key = client.get(f"{base}/key", headers=headers)
            models = client.get(f"{base}/models")
        except httpx.TransportError as error:
            raise SystemExit(
                f"cannot reach the optimizer endpoint {base} ({type(error).__name__}: "
                f"{error}). Compute nodes may have no outbound internet: {PROXY_HINT}."
            ) from error

    if key.status_code != 200:
        raise SystemExit(f"{base} refused the API key (HTTP {key.status_code})")

    if models.status_code != 200:
        raise SystemExit(f"{base}/models answered HTTP {models.status_code}")

    listed = {
        str(entry.get("id")): entry
        for entry in models.json().get("data", [])
        if isinstance(entry, dict)
    }

    if model not in listed:
        raise SystemExit(f"{base} lists no model {model!r}")

    supported = listed[model].get("supported_parameters") or []

    if effort and "reasoning_effort" not in supported:
        raise SystemExit(
            f"{model} does not take reasoning_effort on {base}; the effort would be dropped"
        )


def install(base_url: str, api_key: str, log: CallLog, transport: object | None = None):
    """Point SkillOpt's optimizer role at base_url and log every call.

    Args:
        base_url: OpenRouter's API base URL.
        api_key: The OpenRouter key; kept in the backend's memory only.
        log: Where each call is recorded.
        transport: Transport of the OpenAI client's HTTP library to use
            instead of the network, for tests.
    """
    import openai
    from skillopt.model import azure_openai, configure_azure_openai

    configure_azure_openai(optimizer_endpoint=base_url, optimizer_auth_mode="openai_compatible")
    # Set directly rather than through configure_azure_openai, which would
    # also export the key to every child process, e.g. T1's code executor.
    azure_openai.OPTIMIZER_API_KEY = api_key
    http_client = openai.DefaultHttpxClient(
        transport=transport,  # type: ignore[arg-type]
        trust_env=transport is None,
        event_hooks={"request": [log.stamp], "response": [log.record]},
    )
    azure_openai.OpenAI = functools.partial(openai.OpenAI, http_client=http_client)  # type: ignore[attr-defined, assignment]
