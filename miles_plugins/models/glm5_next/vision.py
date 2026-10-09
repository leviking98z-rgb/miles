# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen GLM-5.3 visual tower and Megatron model provider."""

from __future__ import annotations

import json
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchMerger,
)

from miles.utils.hf_utils.config import load_hf_config

if TYPE_CHECKING:
    from megatron.core.models.gpt import GPTModel

_VISUAL_PREFIX = "model.visual."


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config) -> None:
        super().__init__(config, bias=config.attention_bias)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mlp = Glm5NextVisionMLP(config)


class Glm5NextVisionPatchMerger(GlmOcrVisionPatchMerger):
    def __init__(self, config) -> None:
        super().__init__(
            dim=config.out_hidden_size,
            context_dim=config.projection_intermediate_size,
            hidden_act=config.hidden_act,
            bias=False,
        )
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act1(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(max=self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(self.act_fn(gate) * up)


class Glm5NextVisionModel(GlmOcrVisionModel):
    """Vision-only portion of the upstream GLM-5.3 model."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.blocks = torch.nn.ModuleList(Glm5NextVisionBlock(config) for _ in range(config.depth))
        self.merger = Glm5NextVisionPatchMerger(config)


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _visual_weight_files(hf_checkpoint: str) -> dict[str, list[str]]:
    checkpoint = Path(hf_checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        weights_by_file: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            if name.startswith(_VISUAL_PREFIX):
                weights_by_file.setdefault(filename, []).append(name)
        return weights_by_file

    single_file = checkpoint / "model.safetensors"
    if not single_file.is_file():
        raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors found in {hf_checkpoint}")
    with safe_open(single_file, framework="pt") as shard:
        names = [name for name in shard.keys() if name.startswith(_VISUAL_PREFIX)]
    return {single_file.name: names}


def _load_visual_state_dict(hf_checkpoint: str) -> dict[str, torch.Tensor]:
    weights_by_file = _visual_weight_files(hf_checkpoint)
    if not any(weights_by_file.values()):
        raise RuntimeError(f"No {_VISUAL_PREFIX} weights found in {hf_checkpoint}")

    state_dict = {}
    for filename, names in weights_by_file.items():
        with safe_open(Path(hf_checkpoint) / filename, framework="pt") as shard:
            for name in names:
                state_dict[name.removeprefix(_VISUAL_PREFIX)] = shard.get_tensor(name)
    return state_dict


def build_glm5_next_visual(
    hf_checkpoint: str,
    *,
    device: torch.device | int,
    dtype: torch.dtype,
) -> Glm5NextVisionModel:
    """Construct and strictly load the checkpoint's frozen visual tower."""
    hf_config = load_hf_config(hf_checkpoint)
    vision_config = hf_config.vision_config
    vision_config._attn_implementation = "sdpa"
    with torch.device(device), _default_dtype(dtype):
        visual = Glm5NextVisionModel(vision_config)

    visual.load_state_dict(_load_visual_state_dict(hf_checkpoint), strict=True)
    visual.requires_grad_(False)
    visual.eval()
    return visual


def _scatter_image_embeddings(
    model: GPTModel,
    decoder_input: torch.Tensor,
    positions: torch.Tensor,
    image_embeddings: torch.Tensor,
) -> torch.Tensor:
    from megatron.core import parallel_state

    positions = positions.to(decoder_input.device)
    image_embeddings = image_embeddings.to(device=decoder_input.device, dtype=decoder_input.dtype)
    if model.config.sequence_parallel and parallel_state.get_tensor_model_parallel_world_size() > 1:
        sequence_length = decoder_input.shape[0]
        rank = parallel_state.get_tensor_model_parallel_rank()
        local_positions = positions - rank * sequence_length
        selected = (local_positions >= 0) & (local_positions < sequence_length)
        local_positions = local_positions[selected]
        image_embeddings = image_embeddings[selected]
    else:
        local_positions = positions

    decoder_input = decoder_input.clone()
    if local_positions.numel():
        decoder_input[local_positions, 0] = image_embeddings
    return decoder_input


def _reject_non_image_inputs(kwargs: dict) -> None:
    unsupported_keys = (
        "pixel_values_videos",
        "video_grid_thw",
        "videos",
        "audio",
        "audios",
        "input_features",
        "input_values",
        "audio_values",
    )
    for key in unsupported_keys:
        value = kwargs.pop(key, None)
        if value is not None:
            nonempty = value.numel() > 0 if isinstance(value, torch.Tensor) else bool(value)
            if nonempty:
                raise ValueError(f"GLM-5.3 image SFT does not support video or audio inputs: {key}")


def wire_glm5_next_visual(
    model: GPTModel,
    hf_checkpoint: str,
    *,
    device: torch.device | None = None,
) -> None:
    """Attach frozen image encoding while leaving language parameter names unchanged."""
    original_forward = model.forward
    if not model.pre_process:

        def _passthrough(*args, pixel_values=None, image_grid_thw=None, **kwargs):
            _reject_non_image_inputs(kwargs)
            return original_forward(*args, **kwargs)

        model.forward = _passthrough
        return

    device = device or torch.device("cuda", torch.cuda.current_device())
    hf_config = load_hf_config(hf_checkpoint)
    visual = build_glm5_next_visual(hf_checkpoint, device=device, dtype=model.config.params_dtype)
    visual.requires_grad_(False).eval()
    # Keep the frozen tower outside the Megatron optimizer and distributed checkpoint.
    model.__dict__["_glm5_next_visual"] = visual
    image_token_id = hf_config.image_token_id

    def _multimodal_forward(*args, pixel_values=None, image_grid_thw=None, **kwargs):
        _reject_non_image_inputs(kwargs)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if pixel_values is None:
            if image_grid_thw is not None or (input_ids is not None and (input_ids == image_token_id).any()):
                raise ValueError("GLM-5.3 image tokens and image_grid_thw require pixel_values")
            return original_forward(*args, **kwargs)
        if image_grid_thw is None:
            raise ValueError("pixel_values requires image_grid_thw")
        if kwargs.get("decoder_input") is not None:
            raise ValueError("GLM-5.3 image SFT cannot combine pixel_values with precomputed decoder_input")

        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("GLM-5.3 VLM training expects packed input_ids with shape [1, sequence]")
        position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
        decoder_input = model.embedding(input_ids=input_ids, position_ids=position_ids)

        with torch.no_grad():
            image_embeddings = model.__dict__["_glm5_next_visual"](
                pixel_values.to(device=device, dtype=visual.dtype),
                grid_thw=image_grid_thw.to(device=device),
            ).pooler_output
        image_positions = torch.nonzero(input_ids.reshape(-1) == image_token_id, as_tuple=False).flatten()
        if image_positions.numel() != image_embeddings.shape[0]:
            raise ValueError(
                f"GLM-5.3 image token/feature mismatch: {image_positions.numel()} token(s), "
                f"{image_embeddings.shape[0]} feature row(s)"
            )
        kwargs["decoder_input"] = _scatter_image_embeddings(
            model,
            decoder_input,
            image_positions,
            image_embeddings,
        )
        return original_forward(*args, **kwargs)

    model.forward = _multimodal_forward


def glm5_next_vlm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
) -> GPTModel:
    """Build the validated GLM-5.3 text model and attach its frozen visual tower."""
    from megatron.training import get_args

    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    args = get_args()
    if getattr(args, "context_parallel_size", 1) != 1:
        raise ValueError("GLM-5.3 VLM training currently requires --context-parallel-size 1")
    if getattr(args, "qkv_format", "thd") != "thd":
        raise ValueError("GLM-5.3 VLM training currently requires packed --qkv-format thd")

    provider_args = copy(args)
    provider_args.custom_model_provider_path = None
    base_provider = get_model_provider_func(provider_args, role="actor")

    model = base_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
    wire_glm5_next_visual(model, args.hf_checkpoint)
    return model
