"""vLLM adapter: inference against a running OpenAI-compatible vLLM server.

The server owns the weights and applies the checkpoint's own chat template.
This process only reads the processor off the same local path, for the one
thing the request cannot carry: the resolution the image is fed at, and the
image token count that resolution costs, both of which land in the results
row exactly as models.hf reports them.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any

import httpx
import torch
from PIL import Image
from torch import Tensor

from ..config import ModelCfg
from ..image.preprocess import PreprocessSpec, resize_image
from .base import VLM
from .registry import MODELS

log = logging.getLogger(__name__)

NO_GRADIENT = (
    "model.name=vllm is an inference backend: the server returns sampled text, "
    "not the graph behind it, so nothing carries gradient back to the image. "
    "Train with model.name=hf and score the saved PNG with model.name=vllm."
)

_RENAMED = {"max_new_tokens": "max_tokens", "num_return_sequences": "n"}
_IGNORED = frozenset({"do_sample", "pad_token_id", "eos_token_id", "use_cache"})
_ACCEPTED = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "stop_token_ids",
        "frequency_penalty",
        "ignore_eos",
        "max_tokens",
        "n",
        "logprobs",
    }
)


@MODELS.register("vllm")
class VLLMClient(VLM):
    """A frozen local checkpoint served by vLLM, driven over its OpenAI API.

    The constructor blocks until the server answers, so a job that starts the
    server and the evaluation together does not race the weight load.
    """

    def __init__(self, cfg: ModelCfg) -> None:
        """Read the processor off the local checkpoint and wait for the server.

        Args:
            cfg: Model configuration naming the checkpoint and the server.
        """
        if not cfg.model_id:
            raise ValueError("model.model_id is empty: point it at the local checkpoint served")
        self.cfg = cfg
        self.base_url = cfg.vllm.base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(cfg.vllm.timeout, connect=30.0),
            headers={"Authorization": f"Bearer {cfg.vllm.api_key}"},
        )
        self._spec = _load_spec(cfg)
        self._size = self._spec.resolve_resolution(cfg.resolution)
        self._warned_truncation = False
        self.model_name = self._resolve_model()

        if cfg.device != "auto":
            log.info(
                "model.device=%s is ignored by this backend: the vLLM server owns placement",
                cfg.device,
            )
        log.info(
            "serving %s as %r at %s, feeding %dx%d, expected image tokens: %s",
            cfg.model_id,
            self.model_name,
            self.base_url,
            *self._size,
            self._spec.token_count(*self._size),
        )

    @property
    def preprocess(self) -> PreprocessSpec:
        """The preprocessing constants read off the local checkpoint's processor.

        Returns:
            The spec the image is resized to before it is sent.
        """
        return self._spec

    @property
    def size(self) -> tuple[int, int]:
        """The resolution the image is sent at.

        Returns:
            Height and width the server receives.
        """
        return self._size

    @property
    def image_token_count(self) -> int | None:
        """Image tokens the sent resolution costs.

        Returns:
            The count derived from the processor's patch and merge size, or
            None when the spec exposes no patch size.
        """
        return self._spec.token_count(*self._size)

    @property
    def trainable(self) -> bool:
        """Nothing a served model returns carries gradient to the image.

        Returns:
            Always False.
        """
        return False

    def loss_on_target(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> Tensor:
        """Refuse: this backend cannot produce a differentiable loss.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None.
            target: The gold continuation the loss would be taken on.

        Returns:
            Never returns; always raises NotImplementedError.
        """
        raise NotImplementedError(NO_GRADIENT)

    def check_grad_route(
        self, messages: list[dict[str, Any]], target: str = "ok"
    ) -> dict[str, Any]:
        """Refuse: there is no gradient route through a served model.

        Args:
            messages: Chat messages in the processor's template format.
            target: A short gold continuation the loss would be taken on.

        Returns:
            Never returns; always raises NotImplementedError.
        """
        raise NotImplementedError(NO_GRADIENT)

    def generate(self, messages: list[dict[str, Any]], image: Tensor | None, **kw: Any) -> str:
        """Send one chat completion and read the continuation back off it.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None for a text-only prompt.
            **kw: Generation overrides, applied on top of the configured ones.

        Returns:
            The decoded continuation, stripped of surrounding whitespace.
        """
        body = self._body(messages, image, kw)
        return self._read_choice(self._post("/chat/completions", body))

    def describe(self) -> dict[str, Any]:
        """Row material for results: what identifies this model instance.

        The checkpoint path, not the served name, so a row joins against the
        same checkpoint evaluated through models.hf.

        Returns:
            The checkpoint id and the dtype it was configured to run in.
        """
        return {"model": self.cfg.model_id, "model_dtype": self.cfg.dtype}

    def _body(
        self, messages: list[dict[str, Any]], image: Tensor | None, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Assemble the chat completion request for one item.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None for a text-only prompt.
            overrides: Per-call generation overrides beating the configured ones.

        Returns:
            The JSON body to post to the chat completions endpoint.
        """
        generate = {"max_new_tokens": self.cfg.max_new_tokens, **self.cfg.generate, **overrides}
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": to_openai_messages(
                messages, self._data_uri(image) if image is not None else None
            ),
            **sampling_body(generate),
        }

        if self.cfg.chat_template_kwargs:
            # The server renders the template, so the family's own switches
            # (enable_thinking and friends) have to ride along on the request.
            body["chat_template_kwargs"] = dict(self.cfg.chat_template_kwargs)

        return body

    def _data_uri(self, image: Tensor) -> str:
        """Resize the canvas to the fed resolution and encode it as a PNG data URI.

        The canvas is already on the 8-bit grid (ImageParam.tensor quantizes
        straight through), so when its size already matches the fed resolution
        the server receives exactly the pixels that were optimized.

        Args:
            image: A [0,1] image tensor of shape (3,H,W), or (1,3,H,W).

        Returns:
            A data URI holding the PNG the request carries.
        """
        resized = resize_image(image.detach().float().cpu(), self._spec, self._size)
        array = (resized[0].clamp(0.0, 1.0) * 255).round().to(torch.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB").save(buffer, format="PNG")

        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    def _read_choice(self, payload: dict[str, Any]) -> str:
        """Pull the continuation out of a chat completion response.

        Args:
            payload: The decoded JSON body the server answered with.

        Returns:
            The generated text, stripped of surrounding whitespace.
        """
        choices = payload.get("choices") or []

        if not choices:
            raise RuntimeError(f"vLLM returned no choices: {payload}")

        choice = choices[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")

        if not content.strip():
            # With a reasoning parser enabled server-side the trace is split
            # off into its own field, and a trace that ran out of budget
            # before the answer leaves content empty.
            content = str(message.get("reasoning_content") or "")

        if choice.get("finish_reason") == "length" and not self._warned_truncation:
            self._warned_truncation = True
            log.warning(
                "a generation hit model.max_new_tokens=%d and was cut off before it "
                "finished; a reasoning model that is truncated mid-trace scores as a "
                "wrong answer. Raise model.max_new_tokens. Further truncations are "
                "not logged.",
                self.cfg.max_new_tokens,
            )

        return content.strip()

    def _resolve_model(self) -> str:
        """Confirm the server on this port is the one serving model.model_id.

        Whatever answers base_url decides what gets generated, while the
        results row is written from model.model_id: a server that is not this
        checkpoint's would file another model's answers under this model's
        name, so the mismatch is fatal rather than logged.

        Returns:
            The name the server files the configured checkpoint under.
        """
        wanted = self.cfg.vllm.served_model_name or self.cfg.model_id
        served = self._await_models()

        if wanted not in served:
            names = ", ".join(repr(name) for name in served) or "nothing"
            raise RuntimeError(
                f"the vLLM server at {self.base_url} serves {names}, not "
                f"{wanted!r}: this is someone else's server on that port, and "
                "its generations would be scored as this model's. Give the "
                "checkpoint a port of its own, or set "
                "model.vllm.served_model_name to the name this server does "
                "file it under."
            )

        return wanted

    def _await_models(self) -> list[str]:
        """Wait for the server and read back the names it serves.

        Doubles as the readiness check: /v1/models only answers once the
        weights are loaded, so an evaluation started alongside the server
        blocks here instead of failing on the first item.

        Returns:
            The model names the server lists.
        """
        deadline = time.monotonic() + self.cfg.vllm.startup_timeout
        announced = False

        while True:
            try:
                response = self._client.get(f"{self.base_url}/models", timeout=30.0)
                response.raise_for_status()
                served = [str(entry["id"]) for entry in response.json().get("data", [])]

                if served:
                    return served
                last = f"the server lists no models at {self.base_url}/models"
            except (httpx.HTTPError, KeyError, ValueError) as error:
                last = f"{type(error).__name__}: {error}"

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"no vLLM server answered at {self.base_url} within "
                    f"model.vllm.startup_timeout={self.cfg.vllm.startup_timeout}s ({last}). "
                    f"Start one with: vllm serve {self.cfg.model_id}"
                )

            if not announced:
                announced = True
                log.info("waiting for the vLLM server at %s to come up", self.base_url)
            time.sleep(5.0)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Post one request, retrying what a warming or saturated server returns.

        A 4xx is the request's own fault and is raised immediately; a 5xx or a
        dropped connection produced no generation, so replaying it changes
        nothing about what is scored.

        Args:
            path: Endpoint path below the configured base URL.
            body: JSON body to send.

        Returns:
            The decoded JSON response.
        """
        url = f"{self.base_url}{path}"
        last = ""

        for attempt in range(1, self.cfg.vllm.max_retries + 1):
            try:
                response = self._client.post(url, json=body)
            except httpx.RequestError as error:
                last = f"{type(error).__name__}: {error}"
            else:
                if 400 <= response.status_code < 500:
                    raise RuntimeError(
                        f"vLLM rejected the request ({response.status_code}): "
                        f"{response.text[:500]}"
                    )

                if response.status_code < 400:
                    return dict(response.json())
                last = f"HTTP {response.status_code}: {response.text[:500]}"

            if attempt < self.cfg.vllm.max_retries:
                backoff = min(2.0**attempt, 30.0)
                log.warning("vLLM request failed (%s); retrying in %.0fs", last, backoff)
                time.sleep(backoff)

        raise RuntimeError(
            f"vLLM request to {url} failed {self.cfg.vllm.max_retries} times; last: {last}"
        )


def sampling_body(generate: dict[str, Any]) -> dict[str, Any]:
    """Translate HF generate kwargs into vLLM request fields.

    Unknown keys are refused rather than dropped: decoding settings are the
    difference between two runs of the same split, and silently ignoring one
    would make the two look comparable when they are not.

    Args:
        generate: HF-style generation kwargs from model.generate plus overrides.

    Returns:
        The request fields controlling decoding.
    """
    out: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in generate.items():
        if key in _IGNORED:
            continue
        name = _RENAMED.get(key, key)

        if name in _ACCEPTED:
            out[name] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"model.generate key(s) {sorted(unknown)} have no vLLM request field; "
            f"accepted: {sorted(_ACCEPTED | set(_RENAMED) | _IGNORED)}"
        )

    if generate.get("do_sample") is False:
        # Greedy. vLLM has no do_sample
        if out.get("temperature"):
            log.warning(
                "model.generate has do_sample=false and temperature=%s; sending "
                "temperature=0, i.e. greedy",
                out["temperature"],
            )
        out["temperature"] = 0.0

    return out


def to_openai_messages(
    messages: list[dict[str, Any]], image_url: str | None
) -> list[dict[str, Any]]:
    """Rewrite the chat-template message format into the OpenAI content blocks.

    Args:
        messages: Chat messages carrying text blocks and image placeholders.
        image_url: Data URI every image block is replaced with, or None.

    Returns:
        The same turns in the format the chat completions endpoint takes.
    """
    out: list[dict[str, Any]] = []

    for message in messages:
        content = message.get("content")

        if isinstance(content, str):
            out.append({"role": message["role"], "content": content})
            continue

        blocks: list[dict[str, Any]] = []

        for block in content or []:
            kind = block.get("type")

            if kind == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                if image_url is None:
                    raise ValueError("messages carry an image block but no image was passed")
                blocks.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                raise ValueError(f"unsupported content block type {kind!r}")

        out.append({"role": message["role"], "content": blocks})

    return out


def _load_spec(cfg: ModelCfg) -> PreprocessSpec:
    """Read the preprocessing constants off the served checkpoint's processor.

    The processor is config, not weights, so this is cheap. When the
    checkpoint is not reachable from this process — the server may be the only
    machine holding it — the configured resolution stands in, and the results
    row loses only its image token count.

    Args:
        cfg: Model configuration naming the checkpoint and the resolution.

    Returns:
        The spec the image is resized against.
    """
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
    except Exception as error:
        if not cfg.resolution:
            raise ValueError(
                f"could not read a processor from {cfg.model_id!r} "
                f"({error}), and model.resolution is null, so there is no "
                "resolution to send the image at. Set model.resolution, or "
                "point model.model_id at a local checkpoint."
            ) from error
        height, width = cfg.resolution
        log.warning(
            "could not read a processor from %r (%s); falling back to model."
            "resolution %dx%d, and the results row will carry no image token "
            "count",
            cfg.model_id,
            error,
            height,
            width,
        )

        return PreprocessSpec(height=height, width=width)

    return PreprocessSpec.from_processor(processor)
