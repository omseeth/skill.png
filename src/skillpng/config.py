from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BASE_KEY = "_base_"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VLLMCfg(_Strict):
    """How to reach a vLLM server. Read only when model.name is vllm."""

    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    # What the server files the checkpoint under. Empty means model_id,
    # which is what vllm serve <path> answers with; either way the server
    # is refused unless it lists that name.
    served_model_name: str = ""
    # One request can be a whole reasoning trace, so the read timeout is
    # generous. startup_timeout covers waiting for the server to load weights.
    timeout: float = 3600.0
    startup_timeout: float = 1800.0
    # Retried on a connection error or a 5xx, which is what a server that is
    # still warming up or briefly out of KV cache answers with.
    max_retries: int = Field(default=3, ge=1)


class ModelCfg(_Strict):
    """The registry key plus everything needed to load one local checkpoint."""

    name: str = "hf"
    model_id: str = ""
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    device: str = "auto"
    # GPUs the decoder layers are split across, in contiguous runs. One long
    # sequence's backward graph is what outgrows a GPU, so the layers are
    # split, not the batch. Everything else stays on the first.
    num_gpus: int = Field(default=1, ge=1)
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    resolution: tuple[int, int] | None = (256, 256)
    max_new_tokens: int = 512
    generate: dict[str, object] = Field(default_factory=lambda: {"do_sample": False})
    # Extra kwargs for `processor.apply_chat_template`, e.g. {enable_thinking: 
    # false} on families
    chat_template_kwargs: dict[str, object] = Field(default_factory=dict)
    append_eos: bool = False
    target_suffix: str | None = None
    max_target_tokens: int | None = 512
    max_sequence_tokens: int | None = 10240
    vllm: VLLMCfg = Field(default_factory=VLLMCfg)


class BenchmarkCfg(_Strict):
    name: str = ""
    path: str | None = None
    splits: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.6, "val": 0.2, "test": 0.2}
    )
    allow_ungrouped: bool = False
    # The system turn this benchmark asks every item under. The shipped
    # prompt lives in configs/base.yaml, which every experiment config
    # inherits; empty sends no system message at all.
    system_prompt: str = ""
    # null runs every item; a count caps each split. 0 is refused rather than
    # read as "no cap", so a mistyped smoke run cannot become a full one.
    limit: int | None = Field(default=None, ge=1)
    options: dict[str, object] = Field(default_factory=dict)


class ParamCfg(_Strict):
    # One parameterization: a free canvas. The optimization is constrained only 
    # by a clamp into [0,1] after each step — no perturbation budget, no 
    # projection onto an L-inf ball.
    kind: Literal["full"] = "full"
    height: int = 256
    width: int = 256
    # "gray" | "random" | a path to a PNG
    init: str = "gray"


class TrainCfg(_Strict):
    epochs: int = Field(default=1, ge=1)
    # Accumulation steps, not a batched forward: memory stays flat
    # in batch_size, only wall time grows.
    batch_size: int = 2
    lr: float = 0.03
    # Adam's first-moment decay. The torch default is 0.9; lower it to let the
    # step follow the current gradient more closely.
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    # constant holds lr for the whole run; cosine decays it from lr to zero
    # over the run's total steps.
    lr_schedule: Literal["constant", "cosine"] = "constant"
    grad_clip: float | None = None
    log_every: int = 1
    eval_every: int = Field(default=50, ge=1)
    eval_items: int | None = Field(default=64, ge=1)
    select_on: Literal["val_metric", "val_loss"] = "val_metric"
    # A skill document kept in the system turn for the whole run: every loss,
    # every val check and the clean baseline. It is composed through
    # injection.skill_template, the slot the skill-carrying eval conditions
    # use; score the image with the same skill under skillpng_conditioned.
    # null trains under the bare system prompt.
    skill_path: str | None = None
    # Everything one benchmark and model produces lives under run_dir; each
    # run is a directory named run_name inside it. Set run_dir per experiment,
    # e.g. runs/searchqa/qwen3.5-9B.
    run_dir: str = "runs"
    run_name: str = "run"
    # A run never resumes. One aimed at a directory that already holds an
    # image is refused, unless overwrite lets it start over in its place.
    overwrite: bool = False


class EvalCfg(_Strict):
    split: str = "test"
    condition: Literal[
        "none",
        "noise",
        "skillpng",
        "llmskill",
        "skillopt",
        "skillpng_llmskill",
        "skillpng_conditioned",
        # The request none sends, to a server that applies a LoRA adapter
        # (baselines/lora), named by model.vllm.served_model_name.
        "lora",
    ] = "skillpng"
    # The optimized image; read only when condition is skillpng,
    # skillpng_llmskill or skillpng_conditioned.
    image_path: str | None = None
    # The skill document; read only when condition is llmskill, skillopt,
    # skillpng_llmskill or skillpng_conditioned.
    skill_path: str | None = None
    # Score only the first N items of the split, for debugging a model against
    # a handful of instances instead of the whole thing. null scores all of it.
    limit: int | None = Field(default=None, ge=1)


# Where llmskill, skillopt, skillpng_llmskill and skillpng_conditioned put
# the skill in the system turn, and where train.skill_path puts it.
SKILL_TEMPLATE = "{system_prompt}\n\n## Skill\n{skill}\n"


class InjectionCfg(_Strict):
    site: Literal["pre_query", "post_query"] = "pre_query"
    skill_template: str = SKILL_TEMPLATE


class ResultsCfg(_Strict):
    # null files the scored rows under train.run_dir, next to the run
    # directories they came from. A path set here wins over that.
    path: str | None = None


class WandbCfg(_Strict):
    # The only knob: everything else (project, entity, group, job type, mode)
    # is read by wandb itself from the standard WANDB_* environment variables.
    enabled: bool = False


class MemberCfg(_Strict):
    """One more model the image is optimized against, next to model."""

    # The model keys this member sets, e.g. {model_id: ..., resolution:
    # [768, 768]}. Every key it leaves out, nested ones too, keeps the
    # primary's resolved value rather than the ModelCfg default.
    model: dict[str, object] = Field(default_factory=dict)
    # Loss weight relative to the other models' weights.
    weight: float = Field(default=1.0, gt=0.0)

    @field_validator("model")
    @classmethod
    def _model_keys(cls, keys: dict[str, object]) -> dict[str, object]:
        """Refuse keys the model block does not have.

        Args:
            keys: The model keys this member sets.

        Returns:
            The keys unchanged.
        """
        unknown = sorted(set(keys) - set(ModelCfg.model_fields))

        if unknown:
            raise ValueError(f"ensemble member sets unknown model keys {unknown}")

        return keys


class EnsembleCfg(_Strict):
    """Extra models one image is optimized against at once; none by default.

    model stays the primary: the image lives on its device, and run names
    and sidecars follow it. From the command line the members are one YAML
    list, since --set replaces lists whole:

        --set 'ensemble.members=[{model: {model_id: /path, resolution: [768, 768]}}]'
    """

    members: tuple[MemberCfg, ...] = ()
    # The primary's loss weight. The weights of all models are normalized to
    # sum to one, so losses stay comparable to a single-model run.
    primary_weight: float = Field(default=1.0, gt=0.0)
    # Rescale each model's image gradient to unit norm before the weighted
    # sum, so the model with the larger loss cannot drown out the others.
    normalize_grads: bool = False

    @model_validator(mode="after")
    def _needs_members(self) -> EnsembleCfg:
        """Refuse ensemble settings on a run that has no members.

        Returns:
            The ensemble config unchanged.
        """
        if not self.members and (self.normalize_grads or self.primary_weight != 1.0):
            raise ValueError(
                "ensemble.primary_weight and ensemble.normalize_grads need ensemble.members"
            )

        return self


class Config(_Strict):
    seed: int = 0
    model: ModelCfg = Field(default_factory=ModelCfg)
    benchmark: BenchmarkCfg = Field(default_factory=BenchmarkCfg)
    param: ParamCfg = Field(default_factory=ParamCfg)
    train: TrainCfg = Field(default_factory=TrainCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    injection: InjectionCfg = Field(default_factory=InjectionCfg)
    results: ResultsCfg = Field(default_factory=ResultsCfg)
    wandb: WandbCfg = Field(default_factory=WandbCfg)
    ensemble: EnsembleCfg = Field(default_factory=EnsembleCfg)

    @model_validator(mode="after")
    def _resolvable_members(self) -> Config:
        """Resolve every ensemble member once, so a bad one fails at load.

        Returns:
            The config unchanged.
        """
        _ = self.member_models

        return self

    @property
    def member_models(self) -> tuple[ModelCfg, ...]:
        """Resolve each ensemble member onto the primary's model block.

        Returns:
            One full model config per member, in member order.
        """
        primary = self.model.model_dump()

        return tuple(
            ModelCfg.model_validate(deep_merge(primary, member.model))
            for member in self.ensemble.members
        )

    @property
    def loss_weights(self) -> tuple[float, ...]:
        """Normalize the loss weights of the primary and the members.

        Returns:
            One weight per model, the primary first, summing to one.
        """
        raw = (self.ensemble.primary_weight, *(m.weight for m in self.ensemble.members))
        total = sum(raw)

        return tuple(weight / total for weight in raw)

    @property
    def run_path(self) -> Path:
        """Directory this one run writes its images, history and evals into.

        Args:
            None.

        Returns:
            train.run_dir joined with train.run_name.
        """
        return Path(self.train.run_dir) / self.train.run_name

    @property
    def results_path(self) -> Path:
        """File the scored rows of every run in this group are appended to.

        Args:
            None.

        Returns:
            results.path when set, else results.jsonl inside train.run_dir.
        """
        if self.results.path:
            return Path(self.results.path)

        return Path(self.train.run_dir) / "results.jsonl"

    def hash(self) -> str:
        """Fingerprint the whole config so runs with identical settings collide.

        Tracking, GPU placement and train.overwrite are excluded: neither
        whether a run is watched, nor how many GPUs hold its layers, nor
        whether it may replace an earlier image changes what it optimizes.
        The same goes for where the ensemble members sit, and a run without
        members hashes as it did before the ensemble existed.

        Returns:
            The first 16 hex chars of the sha256 of the sorted JSON dump.
        """
        placement = {"model": {"num_gpus", "device"}}
        excluded: dict[str, object] = {
            "wandb": True,
            "model": {"num_gpus"},
            "train": {"overwrite"},
            "ensemble": {"members": {"__all__": placement}} if self.ensemble.members else True,
        }
        payload = json.dumps(self.model_dump(mode="json", exclude=excluded), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def deep_merge(
    base: dict[str, object], override: dict[str, object]
) -> dict[str, object]:
    """Merge two mappings recursively; nested dicts are merged key by key,
    every other value (lists included) is replaced outright.

    Args:
        base: The mapping supplying defaults.
        override: The mapping whose values win on conflict.

    Returns:
        A new dict; neither input is mutated.
    """
    out = dict(base)
    
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value

    return out


def _read_with_bases(
    path: Path, seen: tuple[Path, ...] = ()
) -> dict[str, object]:
    """Load one YAML config and fold it onto the chain of parents named by its
    _base_ key, walking parents first so the child overrides them.

    Args:
        path: The config file to read; a relative _base_ resolves against it.
        seen: Files already visited on this chain, used to detect a cycle.

    Returns:
        The merged mapping, with the _base_ key stripped out.
    """
    path = path.resolve()

    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"circular {BASE_KEY} chain: {chain}")
    
    raw = yaml.safe_load(path.read_text()) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a mapping, got {type(raw).__name__}")
    
    base_ref = raw.pop(BASE_KEY, None)

    if base_ref is None:
        return raw
    
    base_path = Path(base_ref)

    if not base_path.is_absolute():
        base_path = path.parent / base_path

    return deep_merge(_read_with_bases(base_path, (*seen, path)), raw)


def apply_override(
    data: dict[str, object], assignment: str
) -> dict[str, object]:
    """Apply one a.b.c=value override by expanding the dotted key into a nested
    patch and merging it in. The value is parsed as YAML, so 3, true, [1,2] and
    null arrive typed rather than as strings.

    Args:
        data: The config mapping to override.
        assignment: One dotted key.path=value string.

    Returns:
        A new mapping with the override applied.
    """
    if "=" not in assignment:
        raise ValueError(f"override {assignment!r} is not of the form key.path=value")
    
    dotted, _, raw_value = assignment.partition("=")
    value = yaml.safe_load(raw_value)
    keys = dotted.strip().split(".")
    patch: dict[str, object] = {}
    cursor = patch

    for key in keys[:-1]:
        cursor[key] = {}
        cursor = cursor[key]

    cursor[keys[-1]] = value
    
    return deep_merge(data, patch)


def load_config(
    path: str | Path | None = None, overrides: list[str] | None = None
) -> Config:
    """Build the validated Config: read the file and its _base_ chain, apply the
    command-line overrides in order, then validate the result.

    Args:
        path: The config file to read; None starts from the defaults alone.
        overrides: Dotted key.path=value strings, applied left to right.

    Returns:
        The validated Config.
    """
    data: dict[str, object] = {} if path is None else _read_with_bases(Path(path))

    for assignment in overrides or []:
        data = apply_override(data, assignment)

    return Config.model_validate(data)
