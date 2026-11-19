"""The LoRA baseline: adapter targets, the fit, the frozen adapter, the text-only checkpoint.

Offline and without weights. The models are tiny decoders whose module names are
the families' own, the processor a ChatML stand-in, and the adapter real PEFT,
fitted through LoraVLM's own loss path.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from skillpng.benchmarks.base import Benchmark, Item, inject_image
from skillpng.benchmarks.registry import BENCHMARKS
from skillpng.config import BenchmarkCfg, Config, ModelCfg, TrainCfg
from skillpng.evaluate import condition_image, condition_skill, run_eval
from skillpng.image.preprocess import PreprocessSpec
from skillpng.models.base import VLM
from skillpng.models.hf import _forward_accepts, _forward_takes, _requires_pixel_values

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "lora"
sys.path.insert(0, str(BASELINE))

import finetune as lora  # noqa: E402
import text_only  # noqa: E402

needs_peft = pytest.mark.skipif(
    importlib.util.find_spec("peft") is None, reason="needs the lora extra (peft)"
)

VOCAB = 48
WIDTH = 16
SPECIAL = re.compile(r"<\|[^|]+\|>|<[a-z_]+\|>")
TOKENS = re.compile(r"<\|[^|]+\|>|<[a-z_]+\|>|[^\s<]+")


class Processor:
    """A ChatML template over a word-level tokenizer whose ids are handed out on first sight."""

    eos_token = "<|endoftext|>"
    pad_token_id = None
    eos_token_id = 0

    def __init__(self, close: str = "<|im_end|>") -> None:
        """Start the vocabulary with eos and the turn's closing marker.

        Args:
            close: What the template puts after each turn's content.
        """
        self.close = close
        self.vocab: dict[str, int] = {self.eos_token: 0, close: 1}

    def _encode(self, text: str) -> list[int]:
        """Split off the special markers and the words, and id each.

        Args:
            text: Text to tokenize.

        Returns:
            One id per token.
        """
        return [
            self.vocab.setdefault(token, len(self.vocab) % VOCAB) for token in TOKENS.findall(text)
        ]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **kwargs: Any,
    ) -> str:
        """Render the turns as ChatML.

        Args:
            messages: Chat messages, string or text-block content.
            add_generation_prompt: Whether to open an assistant turn.
            tokenize: Ignored; the text is always returned.
            kwargs: Template switches, ignored.

        Returns:
            The rendered text.
        """
        turns = []

        for message in messages:
            content = message["content"]

            if not isinstance(content, str):
                content = " ".join(block.get("text", "") for block in content)
            turns.append(f"<|im_start|> {message['role']} {content}{self.close}")

        if add_generation_prompt:
            turns.append("<|im_start|> assistant")

        return " ".join(turns) + "\n"

    def __call__(
        self, text: Any, return_tensors: str | None = None, add_special_tokens: bool = True
    ) -> dict[str, Any]:
        """Tokenize a batch as the processor does, or one text as the tokenizer does.

        Args:
            text: A list of texts, or one text.
            return_tensors: Ignored; a batch always comes back as tensors.
            add_special_tokens: Ignored.

        Returns:
            The ids, and for a batch its attention mask.
        """
        if isinstance(text, list):
            ids = torch.tensor([self._encode(entry) for entry in text])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        return {"input_ids": self._encode(text)}

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        """Join the tokens of some ids back into text.

        Args:
            ids: Token ids.
            skip_special_tokens: Whether to drop the markers.

        Returns:
            The tokens, space-separated.
        """
        words = {index: token for token, index in self.vocab.items()}
        tokens = [words.get(int(index), "?") for index in ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if not SPECIAL.fullmatch(token)]

        return " ".join(tokens)


class Attention(nn.Module):
    """q_proj, k_proj, v_proj and o_proj, mixed by a gate."""

    def __init__(self) -> None:
        """Build the four projections."""
        super().__init__()
        self.q_proj = nn.Linear(WIDTH, WIDTH)
        self.k_proj = nn.Linear(WIDTH, WIDTH)
        self.v_proj = nn.Linear(WIDTH, WIDTH)
        self.o_proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden: torch.Tensor) -> Any:
        """Mix the hidden states.

        Args:
            hidden: Hidden states.

        Returns:
            The block's output.
        """
        gate = torch.sigmoid(self.q_proj(hidden) * self.k_proj(hidden))
        return self.o_proj(gate * self.v_proj(hidden))


class MLP(nn.Module):
    """gate_proj, up_proj and down_proj, as a gated MLP."""

    def __init__(self) -> None:
        """Build the three projections."""
        super().__init__()
        self.gate_proj = nn.Linear(WIDTH, WIDTH)
        self.up_proj = nn.Linear(WIDTH, WIDTH)
        self.down_proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden: torch.Tensor) -> Any:
        """Transform the hidden states.

        Args:
            hidden: Hidden states.

        Returns:
            The block's output.
        """
        return self.down_proj(torch.relu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Expert(nn.Module):
    """An up_proj and a down_proj, as Nemotron's non-gated experts have."""

    def __init__(self) -> None:
        """Build the two projections."""
        super().__init__()
        self.up_proj = nn.Linear(WIDTH, WIDTH)
        self.down_proj = nn.Linear(WIDTH, WIDTH)

    def forward(self, hidden: torch.Tensor) -> Any:
        """Transform the hidden states.

        Args:
            hidden: Hidden states.

        Returns:
            The expert's output.
        """
        return self.down_proj(torch.relu(self.up_proj(hidden)))


class Mixer(nn.Module):
    """Attention, an MLP, routed experts and a shared one, as Nemotron's layers mix them."""

    def __init__(self) -> None:
        """Build the blocks."""
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()
        self.experts = nn.ModuleList([Expert() for _ in range(2)])
        self.shared_experts = Expert()

    def forward(self, hidden: torch.Tensor) -> Any:
        """Add every block's output to the residual stream.

        Args:
            hidden: Hidden states.

        Returns:
            The layer's output.
        """
        hidden = hidden + self.self_attn(hidden) + self.mlp(hidden)
        routed = sum(expert(hidden) for expert in self.experts)

        return hidden + 0.5 * routed + self.shared_experts(hidden)


class Decoder(nn.Module):
    """A causal LM in name only: two layers over embeddings, logits over the vocab."""

    def __init__(self) -> None:
        """Build the embeddings, the layers and the head."""
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.layers = nn.ModuleList([Mixer() for _ in range(2)])
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int = 0,
    ) -> Any:
        """Run the ids through the layers to logits.

        Args:
            input_ids: Token ids.
            attention_mask: Ignored.
            use_cache: Ignored.
            logits_to_keep: Trailing positions to return logits for, 0 for all.

        Returns:
            An object carrying logits, as a model output does.
        """
        hidden = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden = layer(hidden)

        logits = self.lm_head(hidden)
        kept = logits[:, -logits_to_keep:] if logits_to_keep else logits

        return types.SimpleNamespace(logits=kept)

    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 4, **kwargs: Any
    ) -> torch.Tensor:
        """Continue greedily until the turn closes or the budget runs out.

        Args:
            input_ids: Prompt ids.
            max_new_tokens: Most tokens to add.
            kwargs: Generation settings, ignored.

        Returns:
            The prompt ids followed by the new ones.
        """
        ids = input_ids

        for _ in range(max_new_tokens):
            following = self(ids).logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, following], dim=1)

            if int(following) == 1:
                break

        return ids


class Tower(nn.Module):
    """A vision tower carrying a q_proj of its own, which the adapter must not touch."""

    def __init__(self) -> None:
        """Build the projection."""
        super().__init__()
        self.q_proj = nn.Linear(WIDTH, WIDTH)


class Inner(nn.Module):
    """model in a Qwen3.5 checkpoint: the language model next to the vision tower."""

    def __init__(self) -> None:
        """Build both."""
        super().__init__()
        self.language_model = Decoder()
        self.visual = Tower()


class QwenShaped(nn.Module):
    """model.language_model next to model.visual and an mtp head, with a text-only forward."""

    def __init__(self) -> None:
        """Build the model and the head."""
        super().__init__()
        self.model = Inner()
        self.mtp = Tower()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        logits_to_keep: int = 0,
    ) -> Any:
        """Run the language model; this stand-in is never shown an image.

        Args:
            input_ids: Token ids.
            attention_mask: Ignored.
            pixel_values: Ignored.
            logits_to_keep: Trailing positions to return logits for, 0 for all.

        Returns:
            The language model's output.
        """
        return self.model.language_model(input_ids, logits_to_keep=logits_to_keep)

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate with the language model.

        Args:
            input_ids: Prompt ids.
            kwargs: Generation settings.

        Returns:
            The prompt ids followed by the new ones.
        """
        return self.model.language_model.generate(input_ids, **kwargs)


class NemotronShaped(nn.Module):
    """language_model next to the towers, with a forward that requires pixel_values."""

    def __init__(self) -> None:
        """Build the language model and the tower."""
        super().__init__()
        self.language_model = Decoder()
        self.vision_model = Tower()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
    ) -> Any:
        """Refuse: a text-only loss must go around this forward.

        Args:
            pixel_values: The tiles.
            input_ids: Token ids.
            attention_mask: Mask over input_ids.
            image_flags: The tile mask.

        Returns:
            Never returns; always raises AssertionError.
        """
        raise AssertionError("a text-only loss never reaches the multimodal forward")

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate with the language model, as the checkpoint does without pixels.

        Args:
            input_ids: Prompt ids.
            kwargs: Generation settings.

        Returns:
            The prompt ids followed by the new ones.
        """
        return self.language_model.generate(input_ids, **kwargs)


@BENCHMARKS.register("_test_lora")
class Toy(Benchmark):
    """Every answer is yes; the score reads the first word."""

    def items(self, split: str) -> list[Item]:
        """The items of one split.

        Args:
            split: train, val or test.

        Returns:
            Items whose gold answer is yes.
        """
        count = {"train": 6, "val": 3, "test": 2}[split]
        return [Item(id=f"{split}-{index}", target="yes") for index in range(count)]

    def build_messages(
        self, item: Item, with_image: bool, site: str = "pre_query"
    ) -> list[dict[str, Any]]:
        """The system turn and the item's question.

        Args:
            item: Item to build the prompt for.
            with_image: Whether to reserve an image slot.
            site: Injection site for the image.

        Returns:
            Chat messages for the item.
        """
        question = {"type": "text", "text": f"question {item.id}"}
        messages = [*self.system_messages(), {"role": "user", "content": [question]}]
        return inject_image(messages, site) if with_image else messages

    def score(self, output: str, item: Item) -> float:
        """Whether the answer opens with yes.

        Args:
            output: Text generated by the model.
            item: Item the output was generated for.

        Returns:
            1.0 or 0.0.
        """
        return float(output.split()[:1] == ["yes"])


def _vlm(model: nn.Module, cfg: ModelCfg, processor: Processor) -> Any:
    """A LoraVLM around a stand-in model, built the way HFVLM leaves one.

    Args:
        model: The stand-in the wrapper calls.
        cfg: The model block, its target_suffix resolved.
        processor: The processor the targets were resolved against.

    Returns:
        A wrapper whose loss and generate routes are HFVLM's own.
    """
    vlm = lora.LoraVLM.__new__(lora.LoraVLM)
    vlm.cfg = cfg
    vlm.model = model
    vlm.processor = processor
    vlm.device = torch.device("cpu")
    vlm.dtype = torch.float32
    vlm._forward_accepts = _forward_accepts(model)
    vlm._keeps_logits = _forward_takes(model, "logits_to_keep")
    vlm._needs_pixels = _requires_pixel_values(model)
    vlm._dropped_extras = set()
    vlm._warned_extras = False
    model.eval()
    model.requires_grad_(False)

    return vlm


def _config(tmp_path: Path, **model: Any) -> Config:
    return Config(
        seed=0,
        model=ModelCfg(
            model_id=str(tmp_path / "ckpt"),
            max_new_tokens=4,
            chat_template_kwargs={"enable_thinking": False},
            **model,
        ),
        benchmark=BenchmarkCfg(name="_test_lora", system_prompt="Answer yes."),
        train=TrainCfg(run_dir=str(tmp_path / "runs" / "_test_lora" / "tiny"), run_name="lora_run"),
    )


def _options(**overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "lr": 0.05,
        "epochs": 8,
        "batch_size": 3,
        "warmup": 0.0,
    }
    settings.update(overrides)

    return lora.Options(**settings)


def _fit(tmp_path: Path, model: nn.Module, **overrides: Any) -> tuple[Path, Config]:
    """Run a whole fit on a stand-in model, through lora.run.

    Args:
        tmp_path: Directory the checkpoint path, runs and adapters live under.
        model: The stand-in model.
        overrides: Options to change.

    Returns:
        The frozen adapter's directory and the config it was fitted under.
    """
    cfg = _config(tmp_path)
    processor = Processor()
    dest = lora.run(
        cfg,
        _options(**overrides),
        build=lambda model_cfg: _vlm(model, model_cfg, processor),
        processor=processor,
    )
    assert dest is not None

    return dest, cfg


def _adapter_keys(dest: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(dest / lora.WEIGHTS_NAME), framework="pt") as handle:
        return list(handle.keys())


def test_the_end_of_turn_marker_is_read_off_the_chat_template() -> None:
    assert lora.end_of_turn(Processor(), {}) == "<|im_end|>"
    # Gemma 4's eos is <eos>, while its turns close with this.
    assert lora.end_of_turn(Processor(close="<turn|>"), {}) == "<turn|>"


def test_a_template_closing_no_turn_is_refused() -> None:
    with pytest.raises(SystemExit, match="target_suffix"):
        lora.end_of_turn(Processor(close=""), {})


def test_a_configured_suffix_or_eos_is_kept() -> None:
    cfg = Config(model=ModelCfg(model_id="m", target_suffix="<end>"))
    assert lora.with_end_of_turn(cfg, Processor()).model.target_suffix == "<end>"

    cfg = Config(model=ModelCfg(model_id="m", append_eos=True))
    assert lora.with_end_of_turn(cfg, Processor()).model.target_suffix is None

    cfg = Config(model=ModelCfg(model_id="m"))
    assert lora.with_end_of_turn(cfg, Processor()).model.target_suffix == "<|im_end|>"


@pytest.mark.parametrize(
    ("name", "adapted"),
    [
        ("model.language_model.layers.3.self_attn.q_proj", True),
        ("model.language_model.layers.3.mlp.down_proj", True),
        ("model.language_model.layers.0.linear_attn.in_proj_qkv", False),
        ("model.language_model.layers.0.linear_attn.out_proj", False),
        ("model.visual.blocks.0.attn.qkv", False),
        ("mtp.layers.0.self_attn.q_proj", False),
        ("model.audio_tower.layers.0.self_attn.q_proj", False),
        ("language_model.backbone.layers.4.mixer.q_proj", True),
        ("language_model.backbone.layers.1.mixer.shared_experts.up_proj", True),
        ("language_model.backbone.layers.1.mixer.experts.17.up_proj", False),
        ("sound_encoder.encoder.layers.0.self_attn.q_proj", False),
    ],
)
def test_only_the_language_models_projections_are_adapted(name: str, adapted: bool) -> None:
    pattern = lora.target_pattern(lora.DEFAULT_TARGETS, scoped=True)
    assert bool(re.fullmatch(pattern, name)) is adapted


def test_on_the_language_model_itself_routed_experts_stay_out() -> None:
    pattern = lora.target_pattern(lora.DEFAULT_TARGETS, scoped=False)
    assert re.fullmatch(pattern, "backbone.layers.4.mixer.o_proj")
    assert re.fullmatch(pattern, "backbone.layers.1.mixer.shared_experts.down_proj")
    assert not re.fullmatch(pattern, "backbone.layers.1.mixer.experts.0.down_proj")


def test_the_name_records_what_changes_the_fit() -> None:
    assert lora.Options().name() == "r16_alpha32_lr0.0002_ep5_bs8_nothink_seed42"
    assert lora.Options(select_on="val_loss").name().endswith("_valloss_nothink_seed42")
    assert "_targets" in lora.Options(targets=("q_proj",)).name()
    assert lora.Options(limit=4).run_name().endswith("_limit4")


def test_training_refuses_thinking_and_otherwise_forces_it_off() -> None:
    path = ROOT / "configs" / "searchqa.yaml"

    with pytest.raises(SystemExit, match="thinking off"):
        lora.resolve_target(path, ["model.chat_template_kwargs.enable_thinking=true"])

    with pytest.raises(SystemExit, match="local weights"):
        lora.resolve_target(path, ["model.name=vllm"])

    cfg = lora.resolve_target(path, ["model.model_id=/m"])
    assert cfg.model.chat_template_kwargs["enable_thinking"] is False


@needs_peft
def test_a_text_only_family_takes_its_loss_through_the_language_model(tmp_path: Path) -> None:
    processor = Processor()
    cfg = lora.with_end_of_turn(_config(tmp_path), processor)
    vlm = _vlm(NemotronShaped(), cfg.model, processor)
    assert vlm.text_only and vlm.host is vlm.model.language_model

    lora.attach_lora(vlm, _options())
    messages = Toy(cfg.benchmark).build_messages(Item(id="x", target="yes"), with_image=False)
    loss = vlm.loss_on_target(messages, None, "yes")
    loss.backward()

    grads = [p.grad for p in vlm.host.parameters() if p.requires_grad]
    assert torch.isfinite(loss) and grads and all(g is not None for g in grads[:2])
    assert not any(p.requires_grad for p in vlm.model.vision_model.parameters())


@needs_peft
def test_a_fit_selects_on_val_and_freezes_the_best_adapter(tmp_path: Path) -> None:
    model = QwenShaped()
    frozen = model.model.language_model.lm_head.weight.detach().clone()
    dest, cfg = _fit(tmp_path, model)

    assert dest == tmp_path / "ckpt-lora" / "_test_lora" / _options().name()
    assert not dest.with_name(dest.name + ".partial").exists()
    meta = json.loads((dest / lora.META_NAME).read_text())
    selection = meta["selection"]
    assert meta["served_on"] == "checkpoint"
    assert meta["target"]["target_suffix"] == "<|im_end|>"
    assert selection["best_step"] > 0
    assert selection["best_score"] > selection["val_clean"]
    assert meta["splits"] == {"train": 6, "val": 3, "limit": None}

    keys = _adapter_keys(dest)
    assert keys and all(".language_model." in key for key in keys)
    assert not [key for key in keys if ".experts.0." in key or "visual" in key or "mtp" in key]
    assert any(".shared_experts." in key for key in keys)
    assert torch.equal(model.model.language_model.lm_head.weight, frozen)

    lines = (cfg.run_path / "history.jsonl").read_text().splitlines()
    history = [json.loads(line) for line in lines]
    assert history[0]["step"] == 0 and history[0]["new_best"] is True
    summary = json.loads((cfg.run_path / "summary.json").read_text())
    assert summary["adapter_dir"] == str(dest)
    assert summary["adapter_sha256"] == meta["adapter_sha256"]


@needs_peft
def test_a_tracked_fit_mirrors_its_history_to_wandb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = MagicMock()
    # Where the run would keep its files, which are pruned after finish.
    run.dir = str(tmp_path / "wandb" / "run" / "files")
    calls: list[dict[str, Any]] = []

    def fake_init(**kwargs: Any) -> MagicMock:
        calls.append(kwargs)
        return run

    monkeypatch.setattr("skillpng.train.wandb.init", fake_init)
    cfg = _config(tmp_path)
    cfg = cfg.model_copy(update={"wandb": cfg.wandb.model_copy(update={"enabled": True})})
    processor = Processor()
    lora.run(
        cfg,
        _options(),
        build=lambda model_cfg: _vlm(QwenShaped(), model_cfg, processor),
        processor=processor,
    )

    assert len(calls) == 1
    assert calls[0]["name"] == "lora_run"
    assert calls[0]["config"]["lora"]["rank"] == 8
    lines = (cfg.run_path / "history.jsonl").read_text().splitlines()
    history = [json.loads(line) for line in lines]
    logged = run.log.call_args_list
    assert [call.kwargs["step"] for call in logged] == [record["step"] for record in history]
    assert "val_score" in logged[0].args[0] and "metric_name" not in logged[0].args[0]
    assert run.summary.update.call_args.args[0]["best_step"] > 0
    run.finish.assert_called_once()


@needs_peft
def test_a_text_only_adapter_is_saved_under_the_language_models_own_names(
    tmp_path: Path,
) -> None:
    dest, _ = _fit(tmp_path, NemotronShaped(), select_on="val_loss")
    meta = json.loads((dest / lora.META_NAME).read_text())

    assert meta["served_on"] == "text_only"
    keys = _adapter_keys(dest)
    assert keys and all(key.startswith("base_model.model.layers.") for key in keys)
    assert not [key for key in keys if ".experts." in key and ".shared_experts." not in key]
    assert meta["selection"]["best_step"] > 0


def test_an_existing_adapter_is_never_overwritten(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    dest = lora.adapter_dir(cfg, _options(), None)
    dest.mkdir(parents=True)

    def build(model_cfg: ModelCfg) -> Any:
        raise AssertionError("the weights were loaded for a run that cannot be frozen")

    with pytest.raises(SystemExit, match="never overwritten"):
        lora.run(cfg, _options(), build=build, processor=Processor())


def test_a_capped_run_lands_beside_the_frozen_ones(tmp_path: Path) -> None:
    dest = lora.adapter_dir(_config(tmp_path), _options(limit=2), tmp_path / "root")
    assert dest == tmp_path / "root" / "_test_lora" / f"{_options().name()}_limit2"


def test_a_dry_run_shows_the_target_and_loads_no_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def build(model_cfg: ModelCfg) -> Any:
        raise AssertionError("a dry run loaded the weights")

    out = lora.run(_config(tmp_path), _options(dry_run=True), build=build, processor=Processor())
    shown = capsys.readouterr().out

    assert out is None
    assert "yes<|im_end|>" in shown and "question train-0" in shown


def _shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    from safetensors.torch import save_file

    save_file(tensors, str(path), metadata={"format": "pt"})


def test_the_text_only_checkpoint_holds_the_language_model_under_its_own_names(
    tmp_path: Path,
) -> None:
    from safetensors import safe_open

    source = tmp_path / "omni"
    source.mkdir()
    llm = {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h"}
    (source / "config.json").write_text(json.dumps({"llm_config": llm, "vision_config": {}}))
    embed, head = torch.randn(4, 2), torch.randn(3, 2)
    shards = {
        "model-1.safetensors": {
            "language_model.backbone.embeddings.weight": embed,
            "vision_model.patch.weight": torch.randn(2, 2),
        },
        "model-2.safetensors": {"language_model.lm_head.weight": head},
        "model-3.safetensors": {"sound_encoder.linear.weight": torch.randn(2, 2)},
    }
    weight_map = {}

    for name, tensors in shards.items():
        _shard(source / name, tensors)
        weight_map.update(dict.fromkeys(tensors, name))

    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (source / "tokenizer.json").write_text("{}")
    (source / "modeling_nemotron_h.py").write_text("# remote code\n")

    out = text_only.convert(source, tmp_path / "text")

    assert json.loads((out / "config.json").read_text()) == llm
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        "backbone.embeddings.weight": "model-1.safetensors",
        "lm_head.weight": "model-2.safetensors",
    }
    assert not (out / "model-3.safetensors").exists()
    assert (out / "tokenizer.json").exists() and (out / "modeling_nemotron_h.py").exists()

    with safe_open(str(out / "model-1.safetensors"), framework="pt") as handle:
        assert list(handle.keys()) == ["backbone.embeddings.weight"]
        assert torch.equal(handle.get_tensor("backbone.embeddings.weight"), embed)

    with pytest.raises(SystemExit, match="never overwritten"):
        text_only.convert(source, out)


class Recorder(VLM):
    """Answers yes and records what each request carried."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], Any]] = []

    def loss_on_target(self, messages: Any, image: Any, target: str) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, messages: Any, image: Any, **kw: Any) -> str:
        self.calls.append((messages, image))
        return "yes"

    @property
    def preprocess(self) -> PreprocessSpec:
        return PreprocessSpec(height=8, width=8)

    def describe(self) -> dict[str, object]:
        return {"model": "/ckpt", "model_dtype": "bfloat16"}


def test_the_lora_condition_sends_the_request_of_none_and_keeps_its_own_row(
    tmp_path: Path,
) -> None:
    rows = {}
    calls = {}

    for condition in ("none", "lora"):
        cfg = _config(tmp_path).model_copy(
            update={"eval": _config(tmp_path).eval.model_copy(update={"condition": condition})}
        )
        assert condition_image(cfg) is None and condition_skill(cfg) is None
        model = Recorder()
        rows[condition] = run_eval(cfg, model=model, split="test")
        calls[condition] = model.calls

    assert calls["lora"] == calls["none"]
    assert all(image is None for _, image in calls["lora"])
    assert rows["lora"].condition == "lora" and rows["lora"].skill_digest == "none"
    dumped = rows["lora"].model_dump(mode="json")
    assert "image_digest" not in dumped and "injection_site" not in dumped


def _analysis() -> types.ModuleType:
    """Import analysis/analyze_runs.py.

    Returns:
        The module.
    """
    sys.path.insert(0, str(ROOT / "analysis"))

    try:
        return importlib.import_module("analyze_runs")
    finally:
        sys.path.pop(0)


def test_the_adapter_name_is_read_off_both_modes_run_names() -> None:
    analyze_runs = _analysis()
    name = lora.Options().name()
    stem = f"searchqa_qwen3.5-4B_test_lora_{name}"

    for suffix in ("_temperature_07_seed42", "_reasoning_temperature_07_seed42"):
        assert analyze_runs.lora_name({"run_name": stem + suffix}) == name


def test_lora_has_its_own_row_and_a_fit_stuck_at_step_0_shows_the_baseline(
    tmp_path: Path,
) -> None:
    analyze_runs = _analysis()
    name = lora.Options().name()
    group = tmp_path / "searchqa" / "qwen3.5-4B"
    group.mkdir(parents=True)
    rows = [
        {"condition": "none", "score": 0.5, "run_name": "searchqa_qwen3.5-4B_test"},
        {"condition": "lora", "score": 0.7, "run_name": f"x_test_lora_{name}_temperature_07"},
    ]
    lines = [json.dumps({**row, "seed": 42, "timestamp": "2026-10-26T00:00:00"}) for row in rows]
    (group / "results.jsonl").write_text("\n".join(lines) + "\n")
    selected, _ = analyze_runs.select_latest(analyze_runs.load_rows(tmp_path, 42))

    def cells() -> tuple[dict[str, Any], set[str]]:
        failed = analyze_runs.find_failed(
            selected, {}, {}, analyze_runs.load_failed_loras(tmp_path)
        )
        entry = analyze_runs.build_table(selected, False, failed)[0]
        return entry["cells"]["searchqa"], entry["replaced"].get("searchqa", set())

    cell, replaced = cells()
    assert cell["lora"] == 0.7 and cell["baseline"] == 0.5 and not replaced

    run = group / f"lora_{name}"
    run.mkdir()
    selection = {"best_step": 0, "best_score": 0.4, "select_on": "val_metric"}
    (run / "summary.json").write_text(json.dumps({"name": name, "selection": selection}))

    cell, replaced = cells()
    assert cell["lora"] == 0.5 and replaced == {"lora"}
