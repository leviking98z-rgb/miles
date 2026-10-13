import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools import glm5_next_vision_export as exporter
from tools.glm5_next_vision_export import finalize_hf_export


def _checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path.mkdir()
    save_file(tensors, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, "model.safetensors"), "metadata": {"total_size": 0}})
    )
    return path


@pytest.fixture
def export_assets(tmp_path):
    visual = {"model.visual.norm.weight": torch.ones(4, dtype=torch.bfloat16)}
    language = {
        "model.language_model.embed_tokens.weight": torch.ones(4, 4, dtype=torch.bfloat16),
        "model.language_model.layers.0.hc_attn_scale": torch.ones(3),
        "model.language_model.norm.weight": torch.ones(4, dtype=torch.bfloat16),
        "lm_head.weight": torch.ones(4, 4, dtype=torch.bfloat16),
    }
    origin = _checkpoint(
        tmp_path / "origin",
        {**language, **visual, "model.language_model.layers.1.eh_proj.weight": torch.ones(4, 4)},
    )
    origin_visual = _checkpoint(tmp_path / "origin_visual", visual)
    exported = {key.replace("model.language_model.", "model."): value for key, value in language.items()}
    return origin, origin_visual, exported, visual


def test_frozen_visual_is_supplemented_exactly_and_finalization_is_idempotent(tmp_path, export_assets):
    origin, origin_visual, exported, visual = export_assets
    output = _checkpoint(tmp_path / "output", exported)
    values = dict(output=output, origin=origin, origin_visual=origin_visual, vision_trainable="none", num_layers=1)

    first = finalize_hf_export(**values)
    second = finalize_hf_export(**values)

    assert first == second
    assert first["changed_visual_tensors"] == 0
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert "model.language_model.layers.0.hc_attn_scale" in index["weight_map"]
    assert not any(key.startswith("model.layers.") for key in index["weight_map"])
    restored = load_file(output / "frozen-visual.safetensors")
    assert torch.equal(restored["model.visual.norm.weight"], visual["model.visual.norm.weight"])
    assert index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in (*exported.values(), *visual.values())
    )


def test_full_preserves_trained_tower_and_reports_cumulative_updates(tmp_path, export_assets):
    origin, origin_visual, exported, visual = export_assets
    output = _checkpoint(
        tmp_path / "output", {**exported, "model.visual.norm.weight": visual["model.visual.norm.weight"] + 1}
    )

    report = finalize_hf_export(
        output=output, origin=origin, origin_visual=origin_visual, vision_trainable="full", num_layers=1
    )

    assert report["changed_visual_tensors"] == 1
    assert report["max_abs_visual_update"] == 1.0
    assert not (output / "frozen-visual.safetensors").exists()
    assert torch.equal(
        load_file(output / "model.safetensors")["model.visual.norm.weight"], torch.full((4,), 2.0).bfloat16()
    )


@pytest.mark.parametrize("with_visual", [False, True])
def test_full_rejects_missing_or_unchanged_visual_weights(tmp_path, export_assets, with_visual):
    origin, origin_visual, exported, visual = export_assets
    output = _checkpoint(tmp_path / "output", {**exported, **(visual if with_visual else {})})

    with pytest.raises(ValueError, match="trained visual|no cumulative"):
        finalize_hf_export(
            output=output, origin=origin, origin_visual=origin_visual, vision_trainable="full", num_layers=1
        )


def test_incomplete_language_export_is_rejected_before_modifying_shards(tmp_path, export_assets):
    origin, origin_visual, exported, _ = export_assets
    del exported["model.layers.0.hc_attn_scale"]
    output = _checkpoint(tmp_path / "output", exported)

    with pytest.raises(ValueError, match="Language export differs"):
        finalize_hf_export(
            output=output, origin=origin, origin_visual=origin_visual, vision_trainable="none", num_layers=1
        )

    assert "model.embed_tokens.weight" in load_file(output / "model.safetensors")


def test_correct_namespace_validates_only_headers(monkeypatch, tmp_path):
    tensors = {
        "model.language_model.layers.0.weight": torch.ones(2, 4, dtype=torch.bfloat16),
        "model.language_model.layers.0.weight_scale_inv": torch.ones(2, dtype=torch.float32),
    }
    output = _checkpoint(tmp_path / "output", tensors)
    original = exporter.safe_open

    class HeaderOnly:
        def __init__(self, *args, **kwargs):
            self.reader = original(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_slice(self, key):
            return self.reader.get_slice(key)

        def metadata(self):
            return self.reader.metadata()

        def get_tensor(self, key):
            raise AssertionError("Already-normalized shards must not read tensor data")

    monkeypatch.setattr(exporter, "safe_open", HeaderOnly)
    weight_map = dict.fromkeys(tensors, "model.safetensors")

    assert exporter._rewrite_shards(output, weight_map) == 24
