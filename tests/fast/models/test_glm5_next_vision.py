import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

from miles_plugins.models.glm5_next import vision as vision_module
from miles_plugins.models.glm5_next.vision import Glm5NextVisionModel


def _tiny_vision_config():
    return GlmOcrVisionConfig(
        attention_bias=True,
        attention_dropout=0.0,
        depth=1,
        hidden_act="silu",
        hidden_size=8,
        in_channels=3,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        patch_size=2,
        projection_intermediate_size=16,
        rms_norm_eps=1e-5,
        spatial_merge_size=2,
        swiglu_limit=10.0,
        temporal_patch_size=2,
        _attn_implementation="sdpa",
    )


def test_glm5_next_visual_tower_emits_one_embedding_per_merged_patch():
    config = _tiny_vision_config()
    visual = Glm5NextVisionModel(config)
    pixel_values = torch.randn(4, config.in_channels * config.temporal_patch_size * config.patch_size**2)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    output = visual(pixel_values, grid_thw=image_grid_thw).pooler_output

    assert output.shape == (1, config.out_hidden_size)
    assert output.isfinite().all()


def test_build_glm5_next_visual_loads_strictly_and_freezes(tmp_path, monkeypatch):
    config = _tiny_vision_config()
    reference = Glm5NextVisionModel(config)
    save_file(
        {f"model.visual.{key}": value for key, value in reference.state_dict().items()}, tmp_path / "model.safetensors"
    )
    monkeypatch.setattr(vision_module, "load_hf_config", lambda path: SimpleNamespace(vision_config=config))

    visual = vision_module.build_glm5_next_visual(str(tmp_path), device=torch.device("cpu"), dtype=torch.float32)

    assert not visual.training
    assert all(not parameter.requires_grad for parameter in visual.parameters())
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(visual.state_dict()[key], value)


class _Embedding(torch.nn.Embedding):
    def forward(self, input_ids, position_ids=None):
        return super().forward(input_ids).transpose(0, 1).contiguous()


class _LanguageModel(torch.nn.Module):
    def __init__(self, pre_process=True):
        super().__init__()
        self.pre_process = pre_process
        self.config = SimpleNamespace(params_dtype=torch.float32, sequence_parallel=False)
        self.embedding = _Embedding(100, 4)
        self.output = torch.nn.Linear(4, 1, bias=False)
        self.calls = []

    def forward(self, input_ids, position_ids=None, decoder_input=None):
        self.calls.append(decoder_input)
        hidden_states = decoder_input if decoder_input is not None else self.embedding(input_ids, position_ids)
        return self.output(hidden_states)


class _Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)

    @property
    def dtype(self):
        return self.projection.weight.dtype

    def forward(self, pixel_values, grid_thw):
        return SimpleNamespace(pooler_output=self.projection(pixel_values))


@pytest.fixture
def wired_model(monkeypatch):
    model = _LanguageModel()
    visual = _Visual()
    monkeypatch.setattr(vision_module, "build_glm5_next_visual", lambda *args, **kwargs: visual)
    monkeypatch.setattr(vision_module, "load_hf_config", lambda path: SimpleNamespace(image_token_id=99))
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    vision_module.wire_glm5_next_visual(model, "/checkpoint", device=torch.device("cpu"))
    return model, visual


def test_visual_wrapper_keeps_language_gradients_and_excludes_visual_from_checkpoint(wired_model):
    model, visual = wired_model
    input_ids = torch.tensor([[10, 99, 99, 20]])
    pixels = torch.ones(2, 4, requires_grad=True)
    expected_images = visual(pixels, torch.tensor([[1, 2, 4]])).pooler_output.detach()

    model(input_ids, pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 4]])).sum().backward()

    torch.testing.assert_close(model.calls[0][1:3, 0], expected_images)
    assert model.embedding.weight.grad[10].abs().sum() > 0
    assert model.embedding.weight.grad[99].abs().sum() == 0
    assert model.output.weight.grad.abs().sum() > 0
    assert pixels.grad is None
    assert not visual.training
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in visual.parameters())
    assert model.state_dict().keys() == {"embedding.weight", "output.weight"}
    model.train()
    assert not visual.training


@pytest.mark.parametrize(
    "multimodal_inputs",
    [
        {"videos": ["video"]},
        {"pixel_values_videos": torch.ones(1, 4)},
        {"video_grid_thw": torch.tensor([[1, 2, 2]])},
        {"input_features": torch.ones(1, 4)},
        {"audio": ["audio"]},
    ],
)
def test_visual_wrapper_rejects_non_image_inputs(wired_model, multimodal_inputs):
    model, _ = wired_model

    with pytest.raises(ValueError, match="video or audio"):
        model(torch.tensor([[10, 20]]), **multimodal_inputs)


def test_visual_wrapper_rejects_image_token_feature_mismatch(wired_model):
    model, _ = wired_model

    with pytest.raises(ValueError, match="token/feature mismatch"):
        model(
            torch.tensor([[10, 99, 99, 20]]),
            pixel_values=torch.ones(1, 4),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )


def test_visual_wrapper_rejects_image_tokens_without_pixels(wired_model):
    model, _ = wired_model

    with pytest.raises(ValueError, match="require pixel_values"):
        model(torch.tensor([[10, 99, 20]]))


def test_visual_wrapper_requires_image_grid(wired_model):
    model, _ = wired_model

    with pytest.raises(ValueError, match="requires image_grid_thw"):
        model(torch.tensor([[10, 99, 20]]), pixel_values=torch.ones(1, 4))


def test_visual_wrapper_rejects_precomputed_decoder_input(wired_model):
    model, _ = wired_model

    with pytest.raises(ValueError, match="precomputed decoder_input"):
        model(
            torch.tensor([[10, 99, 20]]),
            pixel_values=torch.ones(1, 4),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            decoder_input=torch.ones(3, 1, 4),
        )


def test_visual_wrapper_later_pipeline_stage_consumes_media_kwargs_only():
    model = _LanguageModel(pre_process=False)
    vision_module.wire_glm5_next_visual(model, "/checkpoint", device=torch.device("cpu"))
    decoder_input = torch.ones(4, 1, 4)

    result = model(
        torch.tensor([[10, 99, 99, 20]]),
        decoder_input=decoder_input,
        pixel_values=torch.ones(2, 4),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
    )

    torch.testing.assert_close(result, model.output(decoder_input))
    assert "_glm5_next_visual" not in model.__dict__


@pytest.mark.parametrize("rank", (0, 1))
def test_visual_embedding_scatter_selects_sequence_parallel_slice(monkeypatch, rank):
    core = ModuleType("megatron.core")
    core.parallel_state = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: rank,
    )
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    model = SimpleNamespace(config=SimpleNamespace(sequence_parallel=True))
    decoder_input = torch.zeros(4, 1, 4, requires_grad=True)
    images = torch.tensor([[1.0] * 4, [2.0] * 4])

    output = vision_module._scatter_image_embeddings(model, decoder_input, torch.tensor([1, 6]), images)

    expected = torch.zeros_like(output)
    expected[1 if rank == 0 else 2, 0] = images[rank]
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert decoder_input.grad.sum() == 12


def test_glm5_next_provider_decorates_default_model(monkeypatch):
    args = SimpleNamespace(
        hf_checkpoint="/checkpoint",
        context_parallel_size=1,
        custom_model_provider_path="miles_plugins.models.glm5_next.vision.glm5_next_vlm_model_provider",
    )
    model = object()
    calls = []
    megatron_module = ModuleType("megatron")
    megatron_training_module = ModuleType("megatron.training")
    megatron_training_module.get_args = lambda: args
    megatron_module.training = megatron_training_module
    model_provider_module = ModuleType("miles.backends.megatron_utils.model_provider")

    def fake_get_provider(provider_args, role):
        calls.append((provider_args, role, provider_args.custom_model_provider_path))
        return lambda **kwargs: calls.append(kwargs) or model

    model_provider_module.get_model_provider_func = fake_get_provider
    monkeypatch.setitem(sys.modules, "megatron", megatron_module)
    monkeypatch.setitem(sys.modules, "megatron.training", megatron_training_module)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.model_provider", model_provider_module)
    monkeypatch.setattr(
        vision_module,
        "wire_glm5_next_visual",
        lambda wired_model, checkpoint: calls.append((wired_model, checkpoint)),
    )

    result = vision_module.glm5_next_vlm_model_provider(
        pre_process=False,
        post_process=True,
        vp_stage=2,
    )

    assert result is model
    assert args.custom_model_provider_path.endswith("glm5_next_vlm_model_provider")
    assert calls[0][0] is not args
    assert calls[0][0].hf_checkpoint == args.hf_checkpoint
    assert calls[0][1:] == ("actor", None)
    assert calls[1:] == [
        {"pre_process": False, "post_process": True, "vp_stage": 2},
        (model, "/checkpoint"),
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"context_parallel_size": 2}, "context-parallel-size 1"),
        ({"qkv_format": "bshd"}, "qkv-format thd"),
    ),
)
def test_glm5_next_provider_rejects_unsupported_packing_and_cp(monkeypatch, overrides, message):
    args = SimpleNamespace(hf_checkpoint="/checkpoint", **overrides)
    training_module = ModuleType("megatron.training")
    training_module.get_args = lambda: args
    provider_module = ModuleType("miles.backends.megatron_utils.model_provider")
    provider_module.get_model_provider_func = lambda *args, **kwargs: pytest.fail(
        "Invalid topology reached model setup"
    )
    monkeypatch.setitem(sys.modules, "megatron.training", training_module)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.model_provider", provider_module)

    with pytest.raises(ValueError, match=message):
        vision_module.glm5_next_vlm_model_provider()
