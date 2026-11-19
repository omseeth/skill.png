"""The vLLM backend's request and response handling, without a server.

Everything that decides what the server is asked for — the decoding fields,
the message rewrite, the image encoding — is pure and tested here. The
transport is faked by handing the client a stub httpx client, so no port is
opened and vLLM itself is never installed.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from skillpng.benchmarks.base import Benchmark, Item, inject_image
from skillpng.benchmarks.registry import BENCHMARKS
from skillpng.config import Config, ModelCfg
from skillpng.evaluate import run_eval
from skillpng.image.preprocess import PreprocessSpec
from skillpng.models.vllm import VLLMClient, sampling_body, to_openai_messages
from skillpng.train import Trainer

SIZE = (16, 16)


class FakeResponse:
    """Enough of an httpx.Response for the client to read a result off it."""

    def __init__(self, payload: dict[str, object], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict[str, object]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeHTTP:
    """Records what was posted and replays a canned completion."""

    def __init__(self, payload: dict[str, object] | None = None):
        self.payload = payload or _completion("blue")
        self.posted: list[dict[str, object]] = []

    served = "/local/ckpt"

    def get(self, url: str, timeout: float | None = None) -> FakeResponse:
        return FakeResponse({"data": [{"id": self.served}]})

    def post(self, url: str, json: dict[str, object]) -> FakeResponse:
        self.posted.append(json)
        return FakeResponse(self.payload)


def _completion(content: str, **extra: object) -> dict[str, object]:
    return {"choices": [{"message": {"content": content, **extra}}]}


def _client(monkeypatch: pytest.MonkeyPatch, http: FakeHTTP, **model: object) -> VLLMClient:
    """Build a client whose processor and transport are both stubbed out."""
    monkeypatch.setattr(
        "skillpng.models.vllm._load_spec",
        lambda cfg: PreprocessSpec(height=16, width=16, patch_size=8),
    )
    monkeypatch.setattr("httpx.Client", lambda **kw: http)
    cfg = ModelCfg.model_validate({"model_id": "/local/ckpt", "resolution": SIZE, **model})

    return VLLMClient(cfg)


def _sent_image_url(http: FakeHTTP) -> str:
    """Dig the image data URI out of the body that was posted."""
    messages = http.posted[0]["messages"]
    assert isinstance(messages, list)

    return str(messages[0]["content"][0]["image_url"]["url"])


def test_greedy_becomes_temperature_zero() -> None:
    assert sampling_body({"do_sample": False})["temperature"] == 0.0


def test_sampling_kwargs_pass_through_under_vllm_names() -> None:
    body = sampling_body(
        {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "max_new_tokens": 8000}
    )
    assert body == {"temperature": 0.7, "top_p": 0.8, "max_tokens": 8000}


def test_do_sample_false_overrides_a_configured_temperature() -> None:
    # Greedy and a temperature is a contradiction; greedy has to win, or a
    # run labelled deterministic would quietly sample.
    assert sampling_body({"do_sample": False, "temperature": 0.7})["temperature"] == 0.0


def test_unknown_generate_key_is_refused_not_dropped() -> None:
    with pytest.raises(ValueError, match="no vLLM request field"):
        sampling_body({"num_beams": 4})


def test_hf_only_plumbing_is_dropped() -> None:
    assert sampling_body({"pad_token_id": 0, "use_cache": True}) == {}


def test_image_block_becomes_an_image_url_block() -> None:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]},
    ]
    out = to_openai_messages(messages, "data:image/png;base64,AAA")
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "text", "text": "what?"},
    ]


def test_image_block_without_an_image_is_an_error() -> None:
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    with pytest.raises(ValueError, match="no image was passed"):
        to_openai_messages(messages, None)


def test_unknown_block_type_is_an_error() -> None:
    messages = [{"role": "user", "content": [{"type": "video"}]}]
    with pytest.raises(ValueError, match="unsupported content block"):
        to_openai_messages(messages, None)


def test_generate_posts_the_configured_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP()
    client = _client(
        monkeypatch,
        http,
        max_new_tokens=8000,
        generate={"do_sample": True, "temperature": 0.7},
        chat_template_kwargs={"enable_thinking": False},
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": "clue"}]}]

    assert client.generate(messages, None) == "blue"
    body = http.posted[0]
    assert body["model"] == "/local/ckpt"
    assert body["max_tokens"] == 8000
    assert body["temperature"] == 0.7
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_served_model_name_is_discovered_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, FakeHTTP())
    assert client.model_name == "/local/ckpt"


def test_a_server_holding_the_port_for_another_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http = FakeHTTP()
    http.served = "/local/other-ckpt"
    with pytest.raises(RuntimeError, match="not '/local/ckpt'"):
        _client(monkeypatch, http)


def test_an_explicit_served_name_is_checked_against_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="serves '/local/ckpt'"):
        _client(monkeypatch, FakeHTTP(), vllm={"served_model_name": "elsewhere"})


def test_the_image_rides_as_a_png_at_the_fed_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    http = FakeHTTP()
    client = _client(monkeypatch, http)
    image = torch.rand(3, 16, 16)
    messages = [{"role": "user", "content": [{"type": "image"}]}]
    client.generate(messages, image)

    url = _sent_image_url(http)
    assert url.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert decoded.size == (16, 16)
    assert decoded.mode == "RGB"


def test_the_sent_png_is_the_canvas_that_was_optimized(monkeypatch: pytest.MonkeyPatch) -> None:
    # The canvas is already on the 8-bit grid, so at the fed resolution the
    # server must receive those pixels back unchanged.
    http = FakeHTTP()
    client = _client(monkeypatch, http)
    raw = torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)
    client.generate([{"role": "user", "content": [{"type": "image"}]}], raw.float() / 255.0)

    url = _sent_image_url(http)
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    sent = torch.from_numpy(np.asarray(decoded).copy()).permute(2, 0, 1)
    assert torch.equal(sent, raw)


def test_reasoning_content_is_read_when_content_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # A server with a reasoning parser splits the trace off; a trace cut off
    # before the answer leaves content empty and only the trace behind.
    http = FakeHTTP(_completion("", reasoning_content="thinking out loud"))
    client = _client(monkeypatch, http)
    output = client.generate([{"role": "user", "content": "clue"}], None)
    assert output == "thinking out loud"


def test_a_rejected_request_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class Rejecting(FakeHTTP):
        def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            self.posted.append(json)
            return FakeResponse({"error": "bad request"}, status_code=400)

    http = Rejecting()
    client = _client(monkeypatch, http)
    with pytest.raises(RuntimeError, match="rejected the request"):
        client.generate([{"role": "user", "content": "clue"}], None)
    assert len(http.posted) == 1


def test_training_through_a_served_model_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(monkeypatch, FakeHTTP())
    with pytest.raises(NotImplementedError, match=r"model\.name=hf"):
        client.loss_on_target([{"role": "user", "content": "clue"}], torch.rand(3, 16, 16), "yes")


def test_the_results_row_names_the_checkpoint_not_the_served_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # So a vllm row names the same checkpoint as one evaluated through
    # models.hf.
    client = _client(monkeypatch, FakeHTTP())
    assert client.describe()["model"] == "/local/ckpt"
    assert client.image_token_count == 4  # 16x16 at patch 8


@BENCHMARKS.register("_test_vllm_toy")
class ToyBenchmark(Benchmark):
    """Two grouped items and a scorer, so run_eval has something to drive."""

    def items(self, split: str) -> list[Item]:
        return [
            Item(id=f"i{i}", target="blue", ground_truth="blue", group=f"g{i}") for i in range(2)
        ]

    def build_messages(
        self, item: Item, with_image: bool, site: str = "pre_query"
    ) -> list[dict[str, object]]:
        messages: list[dict[str, object]] = [
            {"role": "user", "content": [{"type": "text", "text": f"clue {item.id}"}]}
        ]

        return inject_image(messages, site) if with_image else messages

    def score(self, output: str, item: Item) -> float:
        return float(output.strip() == item.ground_truth)


def test_run_eval_drives_the_served_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The eval loop end to end: no torch device, no local weights, one row.

    The backend holds no torch module, so this also pins that the condition
    image is still built somewhere the client can read it.
    """
    http = FakeHTTP()
    client = _client(monkeypatch, http)
    cfg = Config.model_validate(
        {
            "model": {"name": "vllm", "model_id": "/local/ckpt", "resolution": list(SIZE)},
            "benchmark": {"name": "_test_vllm_toy"},
            "param": {"height": 16, "width": 16},
            "eval": {"split": "test", "condition": "noise"},
            "train": {"run_dir": str(tmp_path), "run_name": "vllm"},
            "results": {"path": str(tmp_path / "results.jsonl")},
        }
    )
    row = run_eval(cfg, model=client)

    assert row.score == 1.0
    assert row.n_items == 2
    assert row.model == "/local/ckpt"
    assert row.resolution == [16, 16]
    # Every item saw the same image, sent as a PNG.
    assert len(http.posted) == 2
    assert _sent_image_url(http).startswith("data:image/png;base64,")


def test_training_refuses_before_it_spends_an_evaluation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refusal lands in the constructor, not at the first backward."""
    client = _client(monkeypatch, FakeHTTP())
    cfg = Config.model_validate(
        {
            "model": {"name": "vllm", "model_id": "/local/ckpt"},
            "benchmark": {"name": "_test_vllm_toy"},
            "train": {"run_dir": str(tmp_path), "run_name": "vllm"},
        }
    )
    with pytest.raises(ValueError, match="inference-only backend"):
        Trainer(cfg, model=client)
