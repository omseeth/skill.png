"""HuggingFace VLM adapter that keeps the image a differentiable leaf.

The processor is called with the real chat text and a single-colour probe image
at the pinned resolution, and everything it returns is kept except
pixel_values, which we produce ourselves and check once against its own.

Our tensor goes straight to model(...) and its vision tower runs inside the
forward, which is what carries the gradient back to the image. Families that
detach or re-quantize pixel_values internally would need a route around that;
none of the ones we run do.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from PIL import Image
from torch import Tensor

from ..config import ModelCfg
from ..image.preprocess import (
    PreprocessSpec,
    pack_padded_patch_sequence,
    pack_patch_sequence,
    preprocess_image,
)
from .base import VLM, target_span, token_ce
from .compat import (
    patch_hub_kernels,
    patch_mask_factories,
    patch_remote_caches,
    promote_to_sdpa,
    repair_radio_summary_idxs,
    require_sdpa,
    verify_mamba_fast_path,
)
from .linear_attention import load_fla_kernels, verify_fla_fast_path
from .placement import gpus, split_layers
from .registry import MODELS

log = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
_AUTO_CLASSES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModelForCausalLM")
# Tried in order when the config names no implementation of its own: the
# checkpoint's own choice first, then the two kernels torch always has.
# A checkpoint whose config asks for flash_attention_2 cannot load at all
# in an environment with no wheel for it, and sdpa runs the same math.
_ATTN_FALLBACKS: tuple[str | None, ...] = (None, "sdpa", "eager")
# Substrings of the load error that mean the kernel, not the checkpoint,
# was refused.
_ATTN_ERROR_MARKS = (
    "flashattention",
    "flash_attn",
    "flash attention",
    "attn_implementation",
    "scaled_dot_product_attention",
)
# Keys the processor returns that we own ourselves.
_OWNED_KEYS = frozenset({"input_ids", "attention_mask", "pixel_values"})
# The probe's one colour. Unequal channels, so a per-channel constant or a
# channel order read wrong shows up in the pixel parity check.
_PROBE_RGB = (200, 120, 40)
# How far our probe pixels may sit from the processor's. A misread constant
# is off by far more; half precision from the processor rounds coarser.
_PARITY_ATOL = {torch.float32: 1e-3}
_PARITY_ATOL_LOW_PRECISION = 2e-2


@MODELS.register("hf")
class HFVLM(VLM):
    """A frozen HuggingFace vision-language model driven from a local checkpoint.

    Subclass it for families whose pixel packing cannot be derived from the
    processor.
    """

    # A wrapper that promoted nothing owes no proof that it did, which is
    # also what a wrapper built without going through __init__ owes.
    _sdpa_pending: bool = False
    # Checked once, against the first processor call that carries the probe.
    _pixels_checked: bool = False

    def __init__(self, cfg: ModelCfg) -> None:
        """Load the processor and the weights, freeze them, and pin the resolution.

        Args:
            cfg: Model configuration naming the checkpoint, dtype, and routes.
        """
        from transformers import AutoProcessor

        if not cfg.model_id:
            raise ValueError("model.model_id is empty: point it at a local checkpoint or repo id")
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        # Inputs, the image and the logits live on the first GPU of a split.
        self._gpus = gpus(self.device, cfg.num_gpus) if cfg.num_gpus > 1 else []

        if self._gpus:
            self.device = self._gpus[0]
        self.dtype = _DTYPES[cfg.dtype]

        if cfg.trust_remote_code:
            # Before the checkpoint's own modeling code is imported, so that
            # what it binds at import time is already the shim. The kernel
            # lookup is bound by name at its line 38 and answered once, so
            # it has to be in place first too.
            patch_mask_factories()
            patch_hub_kernels()

        # Also before the modeling code is imported: that is when
        # transformers looks for the gated delta rule kernel, once.
        load_fla_kernels()
        self.processor = AutoProcessor.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        self.model = _load_model(cfg, self.dtype)

        if cfg.trust_remote_code:
            # After the load, which is what put those classes in sys.modules
            # and is what left the tower's index buffer uninitialized.
            patch_remote_caches()
            repair_radio_summary_idxs(self.model)
            # The mixers read is_fast_path_available at every forward and
            # fall back silently, so it is checked once, here, rather than
            # discovered from the memory a run uses.
            verify_mamba_fast_path()

        # The gated delta rule's torch fallback would first show as an OOM
        # in the backward, so it is checked here instead.
        verify_fla_fast_path(self.model, self.device)

        # A kernel the load refused still runs when the loaded model is told
        # about it afterwards, and the first call into the model proves it.
        self._sdpa_pending = _promote_sdpa(cfg, self.model)

        if self._gpus:
            split_layers(self.model, self._gpus)
        else:
            self.model.to(self.device)
        self.model.eval()
        # The image is the only leaf that carries gradient. Enforced, not
        # assumed.
        self.model.requires_grad_(False)

        self._spec = PreprocessSpec.from_processor(self.processor)
        self._size = self._spec.resolve_resolution(cfg.resolution)
        self._image_token_count: int | None = None
        self._ref_pv_shape: tuple[int, ...] | None = None
        self._warned_extras = False
        self._forward_accepts = _forward_accepts(self.model)
        self._keeps_logits = _forward_takes(self.model, "logits_to_keep")
        self._needs_pixels = _requires_pixel_values(self.model)
        self._dropped_extras: set[str] = set()

        if cfg.resolution and tuple(cfg.resolution) != self._size:
            log.warning(
                "model.resolution %dx%d is not a size this processor feeds its" " tower; using %dx%d, the nearest one it keeps",
                *cfg.resolution,
                *self._size,
            )
        log.info(
            "loaded %s on %s (%s), feeding %dx%d, expected image tokens: %s",
            cfg.model_id,
            self.device,
            cfg.dtype,
            *self._size,
            self._spec.token_count(*self._size),
        )

    @property
    def preprocess(self) -> PreprocessSpec:
        """The preprocessing constants read off the loaded processor.

        Returns:
            The spec our pixels are built to match.
        """
        return self._spec

    @property
    def size(self) -> tuple[int, int]:
        """The pinned resolution fed to the vision tower.

        Returns:
            Height and width actually fed to the tower.
        """
        return self._size

    @property
    def image_token_count(self) -> int | None:
        """Image tokens the pinned resolution produces.

        Returns:
            The count established by the first forward, or None before it.
        """
        return self._image_token_count

    def loss_on_target(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> Tensor:
        """Teacher-forced CE on the target span only, differentiable w.r.t. image.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None for a text-only prompt.
            target: The gold continuation the loss is taken on.

        Returns:
            The mean cross-entropy over the target tokens.
        """
        return token_ce(*self.target_logits(messages, image, target))

    def target_logits(
        self, messages: list[dict[str, Any]], image: Tensor | None, target: str
    ) -> tuple[Tensor, Tensor, tuple[int, int]]:
        """Run the teacher-forced forward the target loss is taken on.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None for a text-only prompt.
            target: The gold continuation the loss is taken on.

        Returns:
            The logits, the prompt and target ids, and the target's span.
        """
        prompt_ids, _, extra = self._prompt_inputs(messages, with_image=image is not None)
        target_ids = self._encode_target(target, prompt_len=prompt_ids.shape[1])
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        span = target_span(prompt_ids.shape[1], target_ids.shape[1])
        attention_mask = torch.ones_like(input_ids)
        extra = self._extend_extras(extra, prompt_ids.shape[1], input_ids.shape[1])
        pixel_values = self._pixel_values(image) if image is not None else None
        # The span is the tail of the sequence, so the head only has to run on
        # its last target_len + 1 positions: everything earlier is masked out
        # of the loss, and a vocab-wide row per prompt token is the largest
        # tensor in the step.
        logits = self._forward(
            input_ids, attention_mask, pixel_values, extra, keep=target_ids.shape[1] + 1
        )

        return logits, input_ids, span

    def generate(self, messages: list[dict[str, Any]], image: Tensor | None, **kw: Any) -> str:
        """Continue the prompt and decode the continuation without it.

        Args:
            messages: Chat messages in the processor's template format.
            image: A [0,1] image of shape (3,H,W), or None for a text-only prompt.
            **kw: Generation overrides, applied on top of the configured ones.

        Returns:
            The decoded continuation, stripped of special tokens and whitespace.
        """
        tokenizer = self.tokenizer
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            **self.cfg.generate,
            **kw,
        }
        gen_kwargs.setdefault(
            "pad_token_id",
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        )

        with torch.no_grad():
            prompt_ids, attention_mask, extra = self._prompt_inputs(
                messages, with_image=image is not None
            )
            pixel_values = self._pixel_values(image) if image is not None else None

            call: dict[str, Any] = {
                "input_ids": prompt_ids,
                "attention_mask": attention_mask,
                **extra,
            }

            if pixel_values is not None:
                call["pixel_values"] = pixel_values

            with self._proven_sdpa():
                out = self.model.generate(**call, **gen_kwargs)
            new_tokens = out[0, prompt_ids.shape[1] :]

        return str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()

    def describe(self) -> dict[str, Any]:
        """Row material for results: what identifies this model instance.

        Returns:
            The checkpoint id and the dtype it runs in.
        """
        return {"model": self.cfg.model_id, "model_dtype": self.cfg.dtype}

    def _pack_pixel_values(self, pixel_values: Tensor) -> Tensor:
        """Reshape (1,3,H,W) into whatever layout this family's tower expects.

        Args:
            pixel_values: Our differentiable pixels of shape (1,3,H,W).

        Returns:
            The same pixels in the tower's layout, still differentiable.
        """
        if self._spec.max_patches:
            return pack_padded_patch_sequence(
                pixel_values, self._spec, self._merges_patch_rows()
            )

        if self._wants_patch_sequence():
            return pack_patch_sequence(pixel_values, self._spec)

        return pixel_values

    def _merges_patch_rows(self) -> bool:
        """Does this processor hand over one row per soft token, not per patch?

        Two processors of the same budget-packing family differ here: one pads
        a row per patch and lets the tower's merger fold them, the other folds
        the merge x merge block itself and pads a row per soft token. Nothing
        in the config says which, but the row count the processor produced for
        the probe does, and the two budgets only coincide when merge is 1 —
        where the layouts coincide too.

        Returns:
            True when a row holds a whole merge x merge block of patches.
        """
        if self._ref_pv_shape is None or len(self._ref_pv_shape) != 3:
            return False

        rows = self._ref_pv_shape[1]

        return rows == self._spec.max_soft_tokens and rows != self._spec.max_patches

    def _wants_patch_sequence(self) -> bool:
        """Does this processor hand the tower a flat patch sequence, not an image batch?

        Returns:
            True when pixel_values must be packed into a 2-D patch sequence.
        """
        if self._ref_pv_shape is not None:
            return len(self._ref_pv_shape) == 2
        return self._spec.temporal_patch_size is not None

    @property
    def tokenizer(self) -> Any:
        """The processor's tokenizer, or the processor when it is one itself.

        Returns:
            The tokenizer used for targets and decoding.
        """
        return getattr(self.processor, "tokenizer", self.processor)

    def _probe_image(self) -> Image.Image:
        """Single-colour image at the pinned resolution.

        Placeholder expansion only reads its size. Its pixels are checked once
        against ours, then thrown away.

        Returns:
            A uniform RGB image at the pinned resolution.
        """
        height, width = self._size
        return Image.new("RGB", (width, height), color=_PROBE_RGB)

    def _check_pixel_parity(self, reference: Tensor) -> None:
        """Our pixel_values for the probe must equal the processor's. If not, fail loudly.

        Training builds pixel_values itself, from constants read off the
        processor. Misread one and the image is optimized against inputs the
        model never gets anywhere else: validation runs the same path and
        cannot tell, and vLLM, which preprocesses the PNG itself, scores the
        image as if it were blank.

        Args:
            reference: The pixel_values the processor built for the probe.
        """
        height, width = self._size
        colour = torch.tensor(_PROBE_RGB, dtype=torch.float32).view(3, 1, 1)
        probe = colour.expand(3, height, width) / 255.0
        ours = self._packed_pixels(probe).detach().float().cpu()
        theirs = reference.detach().float().cpu()
        atol = _PARITY_ATOL.get(reference.dtype, _PARITY_ATOL_LOW_PRECISION)

        gap = float((ours - theirs).abs().max())

        if torch.allclose(ours, theirs, rtol=0.0, atol=atol):
            log.info("pixel parity with the processor: max gap %.2e", gap)
            return

        spec = self._spec
        raise RuntimeError(
            f"our pixel_values for the probe differ from what "
            f"{spec.raw.get('class', 'the processor')} builds by up to "
            f"{gap:.4g} (ours span "
            f"[{float(ours.min()):.4g}, {float(ours.max()):.4g}], the "
            f"processor's [{float(theirs.min()):.4g}, {float(theirs.max()):.4g}]). "
            "An image trained on these would see different inputs at test "
            f"time. Read off the processor: mean {spec.image_mean}, std "
            f"{spec.image_std}, do_rescale {spec.do_rescale}, rescale_factor "
            f"{spec.rescale_factor:.6g}, do_normalize {spec.do_normalize}. "
            "Fix PreprocessSpec.from_processor for this family."
        )

    def _prompt_inputs(
        self, messages: list[dict[str, Any]], with_image: bool
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        """Run the chat template and the processor over the prompt.

        Args:
            messages: Chat messages in the processor's template format.
            with_image: Whether to pass the probe image, expanding placeholders.

        Returns:
            The prompt ids, its attention mask, and the processor keys we do not own.
        """
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **self.cfg.chat_template_kwargs,
        )
        call: dict[str, Any] = {"text": [text], "return_tensors": "pt"}

        if with_image:
            call["images"] = [self._probe_image()]

        encoded = self.processor(**call)
        moved = {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in dict(encoded).items()
        }

        if with_image and "pixel_values" in moved:
            self._ref_pv_shape = tuple(moved["pixel_values"].shape)

            if not self._pixels_checked:
                self._check_pixel_parity(moved["pixel_values"])
                self._pixels_checked = True

        input_ids = moved["input_ids"]
        attention_mask = moved.get("attention_mask", torch.ones_like(input_ids))
        extra = {key: value for key, value in moved.items() if key not in _OWNED_KEYS}
        extra = self._accepted_extras(extra)

        if with_image:
            self._check_token_count(input_ids)

        return input_ids, attention_mask, extra

    def _target_text(self, target: str) -> str:
        """The gold continuation plus whatever ends the turn.

        Args:
            target: The gold continuation.

        Returns:
            The target with the configured suffix and optional eos appended.
        """
        suffix = self.cfg.target_suffix or ""

        if self.cfg.append_eos:
            eos = getattr(self.tokenizer, "eos_token", None)

            if not eos:
                raise ValueError(
                    "model.append_eos is set but this tokenizer has no eos_token; set "
                    "model.target_suffix to the family's end-of-turn marker instead"
                )
            suffix += str(eos)

        return target + suffix

    def _encode_target(self, target: str, prompt_len: int = 0) -> Tensor:
        """Tokenize the target, truncating it to the configured budgets.

        Args:
            target: The gold continuation.
            prompt_len: Prompt length in tokens, counted against the sequence budget.

        Returns:
            The target ids of shape (1,T), on the model's device.
        """
        if not target:
            raise ValueError("empty target: there would be no loss")
        
        ids = list(self.tokenizer(self._target_text(target), add_special_tokens=False)["input_ids"])
        limit = self.cfg.max_target_tokens

        if limit and len(ids) > limit:
            # Right truncation: the head of the target carries the behaviour,
            # and the boundary with the prompt must stay exact — truncating
            # from the left would move it.
            ids = ids[:limit]

        budget = self.cfg.max_sequence_tokens

        if budget and prompt_len + len(ids) > budget:
            # The target is the only part of the sequence we can cut. The
            # prompt carries the image placeholders and the chat template's
            # structure.
            room = budget - prompt_len
            if room <= 0:
                raise ValueError(
                    f"prompt is {prompt_len} tokens, at or over "
                    f"model.max_sequence_tokens={budget}, leaving no room for "
                    "the target. Raise the budget, or shorten the prompt "
                    "(fewer context documents, a smaller model.resolution — "
                    "image tokens count toward it)."
                )
            ids = ids[:room]

        if not ids:
            raise ValueError(f"target tokenized to nothing: {target[:60]!r}")

        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _accepted_extras(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Drop the processor keys this model's forward has no argument for.

        A processor may return more than its own model reads. Nemotron 3 Nano
        Omni's returns num_patches, which its forward does not take, and the
        call raises TypeError before a step runs. A key the forward cannot
        name is a key it cannot be reading, so it is dropped here, once, for
        both the loss and the generate route.

        Args:
            extra: Processor keys we do not own.

        Returns:
            The subset of them this forward accepts.
        """
        if self._forward_accepts is None:
            return extra

        kept = {key: value for key, value in extra.items() if key in self._forward_accepts}
        dropped = sorted(set(extra) - set(kept) - self._dropped_extras)

        if dropped:
            self._dropped_extras.update(dropped)
            log.warning(
                "this processor returns keys its model's forward does not take,"
                " dropping them: %s",
                ", ".join(dropped),
            )

        return kept

    def _image_flags(self, pixel_values: Tensor, extra: dict[str, Any]) -> dict[str, Any]:
        """Supply the tile mask an InternVL-shaped forward reads before anything else.

        Those forwards, Nemotron 3 Nano Omni's among them, index the tower's
        output with image_flags and call a method on it first, so None is an
        AttributeError rather than a default. No processor returns one: it
        marks the real tiles among padded ones, and every tile we pass is
        real.

        Args:
            pixel_values: Our differentiable pixels, tiles along dim 0.
            extra: Processor keys we do not own.

        Returns:
            The image_flags argument, or nothing when this forward has none.
        """
        if self._forward_accepts is None or "image_flags" in extra:
            return {}

        if "image_flags" not in self._forward_accepts:
            return {}

        flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long, device=self.device)

        return {"image_flags": flags}

    def _extend_extras(
        self, extra: dict[str, Any], prompt_len: int, full_len: int
    ) -> dict[str, Any]:
        """Extend sequence-aligned extra tensors over the appended target positions.

        The one key we actually see here is Qwen's mm_token_type_ids, which
        marks image tokens apart from text ones. The prompt ends on the
        assistant generation prompt, a text token, so the value repeated over
        the target is the text one — right, but only because the boundary
        happens to fall there. A family whose last prompt token were an image
        token would need its own rule.

        Args:
            extra: Processor keys we do not own.
            prompt_len: Length a key must have to count as sequence-aligned.
            full_len: Length of prompt and target concatenated.

        Returns:
            The same keys, with aligned tensors padded out to full_len.
        """
        out: dict[str, Any] = {}

        for key, value in extra.items():

            if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == prompt_len:

                if not self._warned_extras:
                    log.warning(
                        "extending sequence-aligned processor key %r over the target span by "
                        "repeating its last value; check this is right for this model",
                        key,
                    )
                    self._warned_extras = True
                pad = value[:, -1:].expand(-1, full_len - prompt_len)
                out[key] = torch.cat([value, pad], dim=1)
            else:
                out[key] = value

        return out

    def _pixel_values(self, image: Tensor) -> Tensor:
        """Preprocess and pack our image into the tower's pixel_values.

        Args:
            image: A [0,1] image tensor of shape (3,H,W), or (1,3,H,W).

        Returns:
            Differentiable pixels on the model's device and dtype.
        """
        return self._packed_pixels(image).to(device=self.device, dtype=self.dtype)

    def _packed_pixels(self, image: Tensor) -> Tensor:
        """Preprocess and pack our image, on its own device and in float.

        Args:
            image: A [0,1] image tensor of shape (3,H,W), or (1,3,H,W).

        Returns:
            Differentiable pixels in the tower's layout.
        """
        pixel_values = preprocess_image(image, self._spec, self._size)
        pixel_values = self._pack_pixel_values(pixel_values)
        if self._ref_pv_shape is not None and tuple(pixel_values.shape) != self._ref_pv_shape:
            raise NotImplementedError(
                f"our pixel_values {tuple(pixel_values.shape)} do not match "
                f"what this processor produces {self._ref_pv_shape}, and "
                "the image-batch nor the patch-sequence layout fits. This "
                "family packs pixels its own way — override HFVLM. "
                "_pack_pixel_values for it."
            )
        return pixel_values

    @contextmanager
    def _proven_sdpa(self) -> Iterator[None]:
        """Guard the first model call after a promotion, and only that one."""
        if not self._sdpa_pending:
            yield

            return

        # Cleared before the call rather than after it: the proof is owed
        # once, and a call that dies of something else does not owe it again.
        self._sdpa_pending = False

        with require_sdpa(f"the first call into {self.cfg.model_id}"):
            yield

    def _forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        pixel_values: Tensor | None,
        extra: dict[str, Any],
        keep: int = 0,
    ) -> Tensor:
        """Hand the pixels to the model and read the logits back off it.

        Args:
            input_ids: Prompt and target ids concatenated.
            attention_mask: Mask over input_ids.
            pixel_values: Our differentiable pixels, or None for a text-only prompt.
            extra: Processor keys we do not own.
            keep: Trailing positions to want logits for, 0 for the whole sequence.

        Returns:
            The logits, over the whole sequence or over its last keep positions.
        """
        trailing: dict[str, Any] = {"logits_to_keep": keep} if keep and self._keeps_logits else {}

        if pixel_values is None:

            if self._needs_pixels:
                raise NotImplementedError(
                    "this model's forward takes pixel_values as a required "
                    "argument, so it has no text-only route: the clean "
                    "condition can only be measured through generate here. "
                    "Leave train.select_on on val_metric for this family."
                )
            with self._proven_sdpa():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, **trailing)

            return cast(Tensor, out.logits)

        with self._proven_sdpa():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **self._image_flags(pixel_values, extra),
                **trailing,
                **extra,
            )

        return cast(Tensor, out.logits)

    def _check_token_count(self, input_ids: Tensor) -> None:
        """Image token count is part of the objective. If it moves mid-run, fail loudly.

        Args:
            input_ids: Prompt ids whose image placeholders are counted.
        """
        token_id = _find_image_token_id(self.processor, self.tokenizer, self.model.config)
        count = (
            int((input_ids == token_id).sum())
            if token_id is not None
            else (self._spec.token_count(*self._size) or -1)
        )

        if self._image_token_count is None:
            self._image_token_count = count
            log.info("image token count: %d at %dx%d", count, *self._size)
            self._check_patch_grid(count)
        elif count != self._image_token_count:
            raise RuntimeError(
                f"image token count changed mid-run: {self._image_token_count} "
                f"-> {count}. Pin model.resolution; a moving token count "
                "silently changes the objective."
            )

    def _check_patch_grid(self, count: int) -> None:
        """A padded patch sequence hides a wrong grid; every other layout raises on it.

        Feed a resolution the processor would have resized and the packed rows
        come out the wrong shape — except here, where the padding swallows the
        difference and our patches end up under position ids describing a grid
        they do not have. The placeholders the processor expanded for the probe
        say what grid it used, so they are what this is checked against.

        Args:
            count: Image placeholder tokens the processor expanded for the probe.
        """
        expected = self._spec.token_count(*self._size)

        if self._spec.max_patches is None or expected is None or count == expected:
            return

        raise RuntimeError(
            f"the processor expanded {count} image placeholders for a probe at "
            f"{self._size[0]}x{self._size[1]}, but that resolution is "
            f"{expected} image tokens of pixels: it is resizing the image "
            "before the tower sees it, and our patches would sit under a grid "
            "that is not theirs. Set model.resolution to a size it keeps."
        )

    def check_grad_route(
        self, messages: list[dict[str, Any]], target: str = "ok"
    ) -> dict[str, Any]:
        """Does gradient actually reach the image on this checkpoint? Needs real weights.

        Args:
            messages: Chat messages in the processor's template format.
            target: A short gold continuation to take the loss on.

        Returns:
            The route, the loss, and whether the image gradient exists and is finite.
        """
        height, width = self._size
        image = torch.full((3, height, width), 0.5, device=self.device, requires_grad=True)
        loss = self.loss_on_target(messages, image, target)
        loss.backward()
        grad = image.grad
        return {
            "loss": float(loss.detach()),
            "image_grad_is_none": grad is None,
            "image_grad_norm": float(grad.norm()) if grad is not None else None,
            "image_grad_finite": bool(torch.isfinite(grad).all()) if grad is not None else False,
            "image_token_count": self._image_token_count,
        }


def _forward_accepts(model: Any) -> frozenset[str] | None:
    """Keyword names this model's forward takes, or None when it takes any.

    Args:
        model: The loaded model.

    Returns:
        The names its forward accepts, or None when it has a **kwargs or no
        signature to read.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError, AttributeError):
        return None

    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return None

    return frozenset(
        name
        for name, param in params.items()
        if param.kind is not inspect.Parameter.VAR_POSITIONAL
    )


def _forward_takes(model: Any, name: str) -> bool:
    """Does this model's forward declare a parameter of its own by that name?

    Read separately from _forward_accepts, which collapses a forward with
    **kwargs to "anything goes": passing an unknown keyword into one of those
    lands somewhere downstream, so an optimization only rides on a name the
    signature actually carries.

    Args:
        model: The loaded model.
        name: Parameter name to look for.

    Returns:
        True when the forward names it, ignoring any **kwargs it also has.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError, AttributeError):
        return False

    param = params.get(name)

    return param is not None and param.kind is not inspect.Parameter.VAR_KEYWORD


def _requires_pixel_values(model: Any) -> bool:
    """Does this forward take pixel_values as an argument with no default?

    Args:
        model: The loaded model.

    Returns:
        True when calling the forward without pixels is a TypeError.
    """
    try:
        param = inspect.signature(model.forward).parameters.get("pixel_values")
    except (TypeError, ValueError, AttributeError):
        return False

    return param is not None and param.default is inspect.Parameter.empty


def _load_model(cfg: ModelCfg, dtype: torch.dtype) -> Any:
    """Load the checkpoint with the first auto class that accepts it.

    Args:
        cfg: Model configuration naming the checkpoint and the attention impl.
        dtype: Torch dtype the weights are loaded in.

    Returns:
        The loaded model.
    """
    import transformers

    base: dict[str, Any] = {"torch_dtype": dtype, "trust_remote_code": cfg.trust_remote_code}
    attns = _attn_attempts(cfg.attn_implementation)
    errors: list[str] = []

    for name in _AUTO_CLASSES:
        auto_class = getattr(transformers, name, None)

        if auto_class is None:
            continue

        for attn in attns:
            kwargs = dict(base)

            if attn:
                kwargs["attn_implementation"] = attn
            try:
                model = auto_class.from_pretrained(cfg.model_id, **kwargs)
            except Exception as error:
                label = attn or "checkpoint default"
                errors.append(f"{name} ({label}): {type(error).__name__}: {error}")
                # Logged where it happens, not only in the message a total
                # failure raises: a load that goes on to succeed on another
                # kernel is exactly the one whose reason is worth reading.
                log.info("%s refused %s: %s: %s", name, label, type(error).__name__, error)

                # Another kernel is worth a try; anything else is this
                # auto class refusing the checkpoint itself.
                if not _is_attn_failure(error):
                    break
                continue

            if attn != attns[0]:
                log.warning(
                    "%s was refused at load; the weights came up on %s instead",
                    attns[0] or "the implementation this checkpoint asks for",
                    attn,
                )

            return model

    joined = "\n  ".join(errors)

    raise RuntimeError(f"could not load {cfg.model_id!r} with any auto class:\n  {joined}")


def _attn_attempts(named: str | None) -> tuple[str | None, ...]:
    """The attention implementations a load tries, most wanted first.

    A named implementation is the caller's decision and is not second
    guessed; only the checkpoint's own choice gets a fallback, and sdpa,
    which transformers refuses by class attribute rather than by the math.
    A checkpoint that will not load with it is loaded with eager and told
    about sdpa afterwards, which is a route around the attribute and not
    around the check it stands for.

    Args:
        named: The implementation the caller named, or None for the checkpoint's.

    Returns:
        The implementations to try, in order.
    """
    if not named:
        return _ATTN_FALLBACKS

    if named == "sdpa":
        return ("sdpa", "eager")

    return (named,)


def _promote_sdpa(cfg: ModelCfg, model: Any) -> bool:
    """Set sdpa on a model the load could only bring up on another kernel.

    Args:
        cfg: Model configuration naming the attention implementation.
        model: The loaded model.

    Returns:
        True when the implementation moved and the first call has to prove it.
    """
    if cfg.attn_implementation != "sdpa" or model.config._attn_implementation == "sdpa":
        return False

    promote_to_sdpa(model)

    return True


def _is_attn_failure(error: Exception) -> bool:
    """Did the load fail on the attention kernel rather than on the checkpoint?

    Args:
        error: The exception a from_pretrained attempt raised.

    Returns:
        True when retrying with a different attention implementation could help.
    """
    text = str(error).lower()

    return any(mark in text for mark in _ATTN_ERROR_MARKS)


def _find_image_token_id(processor: Any, tokenizer: Any, config: Any) -> int | None:
    """Lookup chain, in order of reliability. No literal token id or string appears here.

    Args:
        processor: The loaded processor.
        tokenizer: The processor's tokenizer.
        config: The model config.

    Returns:
        The image token id, or None when no source holds one.
    """
    for holder, attr in (
        (processor, "image_token_id"),
        (config, "image_token_index"),
        (config, "image_token_id"),
    ):
        value = getattr(holder, attr, None)
        if isinstance(value, int):
            return value
    token = getattr(processor, "image_token", None)
    if isinstance(token, str) and token:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id
    return None


def _pick_device(name: str) -> torch.device:
    """Resolve model.device, saying so loudly when auto lands on the CPU.

    Args:
        name: Configured device, or auto to take the best one available.

    Returns:
        The device the model is placed on.
    """
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    # A GPU job that ends up here has not failed, it has slowed down by orders
    # of magnitude, which is much harder to notice: the log looks healthy and
    # generation simply never finishes. Worth a warning rather than a shrug.
    log.warning(
        "model.device=auto found no GPU and is falling back to the CPU, which "
        "is orders of magnitude slower for generation. This torch is built for "
        "CUDA %s; if the node does have a GPU, its driver is probably older "
        "than that build. Set model.device=cuda to fail fast instead.",
        torch.version.cuda or "nothing (this is a CPU-only build)",
    )

    return torch.device("cpu")
