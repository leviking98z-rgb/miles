from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tools import convert_torch_dist_to_hf_ray as converter
from tools.convert_torch_dist_to_hf_ray import plan_whole_source_tasks


def test_quantization_moves_cpu_parameters_to_gpu_and_results_back(monkeypatch):
    movements = []

    class Tensor:
        def __init__(self, location):
            self.location = location

        def to(self, *, device):
            movements.append(("to", str(device)))
            return Tensor(str(device))

        def cpu(self):
            movements.append(("cpu", self.location))
            return Tensor("cpu")

    def quantize(args, source_name, named_tensors, config):
        assert source_name == "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight0"
        assert all(tensor.location == "cuda:2" for _, tensor in named_tensors)
        return named_tensors

    monkeypatch.setattr(converter.torch.cuda, "set_device", lambda device: movements.append(("set", device)))
    monkeypatch.setattr(converter.m2hf, "quantize_params", quantize)
    group = converter.PreparedTensorGroup(
        "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight0",
        (("model.layers.3.mlp.experts.0.gate_proj.weight", Tensor("cpu")),),
    )

    result = converter.quantize_prepared_group(group, None, {"quant_method": "fp8"}, 2)

    assert result[0][1].location == "cpu"
    assert movements == [("set", 2), ("to", "cuda:2"), ("cpu", "cuda:2")]


def test_quantization_requires_an_assigned_gpu():
    group = converter.PreparedTensorGroup("weight", ())
    with pytest.raises(ValueError, match="conversion GPU"):
        converter.quantize_prepared_group(group, None, {"quant_method": "fp8"}, None)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.mlp.experts.1.up_proj.weight", "model.language_model.layers.0.mlp.experts.1.up_proj.weight"),
        (
            "model.layers.0.self_attn.q_proj.weight_scale_inv",
            "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv",
        ),
        ("language_model.model.layers.1.hc_attn_scale", "model.language_model.layers.1.hc_attn_scale"),
        ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"),
        ("model.norm.weight", "model.language_model.norm.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("model.visual.norm.weight", "model.visual.norm.weight"),
    ],
)
def test_multimodal_namespace_is_restored_without_changing_other_models(name, expected):
    assert converter.restore_multimodal_hf_name(name, "glm5_next") == expected
    assert converter.restore_multimodal_hf_name(name, "deepseekv3") == name
    assert converter.restore_multimodal_hf_name(name, None) == name


@pytest.mark.parametrize("q_lora_rank", [None, 1536])
@pytest.mark.parametrize("task_group_bytes", [0, 4, 32])
def test_hyper_connection_scales_stay_in_one_conversion_task(q_lora_rank, task_group_bytes):
    sites = (
        "decoder.layers.0.self_attention_hyper_connection",
        "decoder.layers.0.mlp_hyper_connection",
        "decoder.layers.1.self_attention_hyper_connection",
    )
    metadata = {
        f"{site}.{alpha}": (torch.Size([1]), torch.float32)
        for site in sites
        for alpha in ("alpha_pre", "alpha_post", "alpha_res")
    }
    metadata["decoder.layers.0.input_layernorm.weight"] = (torch.Size([8]), torch.float32)

    tasks = plan_whole_source_tasks(metadata, q_lora_rank, task_group_bytes)

    for site in sites:
        keys = {f"{site}.{alpha}" for alpha in ("alpha_pre", "alpha_post", "alpha_res")}
        owners = [task for task in tasks if keys.intersection(task.keys)]
        assert len(owners) == 1
        assert keys.issubset(owners[0].keys)
    assert sorted(key for task in tasks for key in task.keys) == sorted(metadata)


def test_mla_dependencies_and_unrelated_parameters_keep_their_groups():
    prefix = "decoder.layers.3.self_attention"
    paired_keys = {
        f"{prefix}.linear_q_down_proj.weight",
        f"{prefix}.linear_kv_down_proj.weight",
    }
    metadata = {key: (torch.Size([8, 8]), torch.bfloat16) for key in paired_keys}
    metadata["decoder.layers.3.input_layernorm.weight"] = (torch.Size([8]), torch.bfloat16)

    tasks = plan_whole_source_tasks(metadata, q_lora_rank=1536, task_group_bytes=0)

    assert {frozenset(task.keys) for task in tasks} == {
        frozenset(paired_keys),
        frozenset({"decoder.layers.3.input_layernorm.weight"}),
    }


@pytest.mark.parametrize("scale_format", [None, "0", "1"])
def test_conversion_worker_receives_explicit_fp8_scale_format(monkeypatch, scale_format):
    if scale_format is None:
        monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    else:
        monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", scale_format)
    options = {}

    def remote(**kwargs):
        options.update(kwargs)
        return lambda cls: cls

    monkeypatch.setattr(converter.ray, "remote", remote)

    assert converter.make_conversion_actor() is converter.ConversionWorker
    worker_env = options["runtime_env"]["env_vars"]
    assert worker_env.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES") == scale_format
    assert worker_env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"


def test_four_node_conversion_uses_all_eight_gpu_slots_per_node(monkeypatch):
    placements = []
    nodes = [f"{index + 1:056x}" for index in range(4)]

    class Worker:
        def __init__(self, node):
            self.node = node

        def options(self, *, scheduling_strategy):
            return Worker(scheduling_strategy.node_id)

        def remote(self, actor_id, *args, cuda_device_index):
            placements.append((self.node, cuda_device_index))
            return SimpleNamespace(convert=SimpleNamespace(remote=lambda task: task.task_id))

    monkeypatch.setattr(converter, "make_conversion_actor", lambda: Worker(None))
    monkeypatch.setattr(converter, "live_ray_node_ids", lambda: nodes)
    monkeypatch.setattr(converter, "ProgressReporter", Mock())
    monkeypatch.setattr(converter.ray, "wait", lambda refs, **kwargs: (refs[:1], refs[1:]))
    monkeypatch.setattr(
        converter.ray,
        "get",
        lambda ref: SimpleNamespace(task_id=ref, node="node", output_bytes=0, weights=0),
    )
    tasks = [converter.TaskSpec(index, (f"key-{index}",), 4) for index in range(32)]

    results = converter.collect_ray_results(
        tasks=tasks,
        input_dir="input",
        staging_dir="output",
        megatron_args=None,
        model_name="glm5_next",
        quantization_config={},
        max_file_bytes=1024,
        concurrency=32,
        metadata_ref=None,
        progress=True,
        progress_interval_seconds=30,
    )

    assert len(results) == 32
    for node in nodes:
        assert sorted(slot for placed_node, slot in placements if placed_node == node) == list(range(8))
