"""GLM-5.3-Flash 4-layer image SFT: trainable visual tower end to end.

The fast tests build the tower on a CPU config with hidden_size 8, so nothing there
reaches the Megatron model provider, the sequence-parallel image-embedding scatter,
the per-parameter `sequence_parallel` TP gradient sum, or the DDP/optimizer/
dist-checkpoint integration. This runs the real recipe on 8 GPUs and then reads the
saved checkpoints back.

It trains twice against one output directory. The second launch is the point: by then
the checkpoint already carries `visual.*`, so the load takes the branch that keeps the
tower attached rather than the first-run branch that hides it.

Two things to know before reading a failure here:

`prepare()` has never run. Every expected value below was measured against an existing
local copy of the checkpoint, so the download, the processor_config.json write and the
conversion execute for the first time in CI. If this goes red on its first run, start
there rather than with the assertions.

`--optimizer-cpu-offload` pins host memory. The run those numbers come from raised
RLIMIT_MEMLOCK first, outside the repository; nothing here does. A 64 KiB limit on the
runner surfaces as `CUDA error: invalid argument` from pin_memory(), which is a runner
image to fix rather than something to work around in this file.
"""

import dataclasses
import json
import os
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from scripts.run_glm5_3_flash_vision_sft import ScriptArgs, execute as _launch

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

# suite:    the launcher raises on `total_gpus < 8 or total_gpus % 4`, so 8 GPUs is the
#           floor for this --model-name and the 8-GPU row is the first stage that fits.
#           h200 over h100 for CPU-offload headroom, matching the sibling GLM-5.3 test.
# num_gpus: 8, the smallest allocation that keeps every case. Lowering it would mean
#           relaxing that assert, which is exactly what the rules forbid.
# hardware: hopper only. This recipe has only ever run on H20; claiming blackwell would
#           be asserting a capability nobody has exercised.
# est_time: measured on 8xH20, 1341s total -- hf download 143s (46 GB), conversion 163s,
#           first launch 450s, resumed launch 338s, checkpoint comparison 247s (9 projector
#           tensors in 7s, 338 frozen in 240s). Rounded up, since under-estimating packs
#           this next to another heavy test in one shard.
register_cuda_ci(
    est_time=1400,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts", "ckpt"],
    hardware=["hopper"],
    num_gpus=8,
)

# Only these two exist on this path: sft_loss_function returns metrics {"loss": ...} and
# grad_norm comes from the Megatron loop. The sibling test's ppo_kl, train_rollout_* and
# raw_reward have no producer under --debug-train-only, so gating them would be dead.
register_ci_gate(metric_key="train/loss")
register_ci_gate(metric_key="train/grad_norm")

_MODEL_REPO = "CharyZeng/GLM-5.3-Flash-4layer"
_MEGATRON_MODEL_TYPE = "glm5.3-flash-4layer"

# Observed on 8xH20 with this exact configuration.
_EXPECTED_VISUAL_KEYS = 347
_PROJECTOR_PREFIXES = ("visual.post_layernorm.", "visual.downsample.", "visual.merger.")
_FROZEN_PREFIXES = ("visual.patch_embed.", "visual.blocks.")

# The published repository carries model_type glm5_next, vision_config, every
# model.visual.* tensor, chat_template.jinja and the tokenizer, but no
# processor_config.json -- and load_processor requires that file to dispatch to the
# GLM-5.3 processor at all, so without it the run fails on the first multimodal row.
#
# Spelled out rather than written as "{}". These are the values the production export
# carries, and they are equal to the Glm5NextImageProcessor class defaults field for
# field; pinning them makes this test notice if one of those defaults ever moves,
# which an empty dict would silently follow.
_PROCESSOR_CONFIG = {
    "image_processor": {
        "do_rescale": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "max_image_tokens": 8000,
        "merge_size": 2,
        "min_image_tokens": 16,
        "patch_expand_factor": 1,
        "patch_size": 14,
        "temporal_patch_size": 2,
    }
}


def _args() -> ScriptArgs:
    # The launcher defaults already match the partial-model CI convention: rollout 16 /
    # global 16 with --n-samples-per-prompt 1 hardcoded is 16 samples per rollout, and
    # 16 % data_parallel_size (8 // 4 = 2) == 0 passes the launcher's own check. The
    # batch fields are deliberately left alone.
    return ScriptArgs.from_env(
        model_name="GLM-5.3-Flash-4layer",
        num_nodes=1,
        num_gpus_per_node=8,
        vision_trainable="projector",
        num_rollout=2,
        save_interval=1,
        max_tokens_per_gpu=512,
        lr=1e-6,
    )


def _build_image_sft_fixture(args: ScriptArgs) -> None:
    """Synthesize the 16-row image SFT set, offline and self-contained.

    Synthetic rather than a public image dataset: it needs no download, and it covers a
    single-image row, a two-image row and a masked multi-turn row, which a uniform
    single-image set would not.
    """
    from PIL import Image, ImageDraw

    data_dir = Path(args.prompt_data).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for number in ("42", "17"):
        path = data_dir / f"glm53_image_sft_{number}.png"
        canvas = Image.new("RGB", (112, 112), (255, 255, 255))
        ImageDraw.Draw(canvas).text((24, 38), number, fill=(0, 0, 0), font_size=36)
        canvas.save(path)
        images.append(str(path))

    records = []
    for index in range(16):
        multi_image = index % 4 == 0
        row = {
            "messages": [
                {
                    "role": "user",
                    "content": "<image><image>Read both numbers." if multi_image else "<image>Read the number.",
                },
                {"role": "assistant", "content": "42, 17" if multi_image else "42"},
            ],
            "images": images if multi_image else images[:1],
        }
        if index % 4 == 1:
            # step_loss_mask 0 plus a further turn, so the GLM loss mask has to skip one
            # assistant span and still supervise the next.
            row["messages"][-1]["step_loss_mask"] = 0
            row["messages"].extend(
                [
                    {"role": "user", "content": "Repeat the number."},
                    {"role": "assistant", "content": "The number is 42."},
                ]
            )
        records.append(row)

    Path(args.prompt_data).write_text("\n".join(json.dumps(row) for row in records) + "\n")


def _metadata(checkpoints: Path, iteration: int):
    return dcp.FileSystemReader(checkpoints / f"iter_{iteration:07d}").read_metadata().state_dict_metadata


def _float_tensors(metadata, prefixes) -> list[str]:
    """Every floating-point parameter under `prefixes`, weights included.

    An earlier version skipped tensors over 65536 elements to keep the reads small. That
    was the wrong trade: it left 0.008% of the projector and 0.092% of the frozen tower
    actually compared, because the excluded tensors are precisely the weight matrices.
    Reading all of them is a fifth of this file's runtime; est_time above has the figure,
    which is where it is measured and the only place it should be written down.
    """
    return [
        key
        for key, value in metadata.items()
        if isinstance(value, TensorStorageMetadata)
        and key.startswith(prefixes)
        and value.properties.dtype.is_floating_point
        and "extra_state" not in key
    ]


def _max_abs_update(checkpoints: Path, metadata, key: str, iterations: list[int]) -> float:
    """Largest absolute change in `key` across consecutive saved iterations."""
    value = metadata[key]
    states = []
    for iteration in iterations:
        state = {key: torch.empty(value.size, dtype=value.properties.dtype)}
        dcp.load(state, checkpoint_id=checkpoints / f"iter_{iteration:07d}")
        states.append(state[key].float())
    return max((states[i + 1] - states[i]).abs().max().item() for i in range(len(states) - 1))


def prepare(args: ScriptArgs):
    U = args.create_backend()
    os.environ["CONVERT_KEEP_PP1"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_REPO} --local-dir {args.hf_checkpoint}")

    config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    text_config = config["text_config"]
    if text_config["dtype"] != "bfloat16" or text_config["num_hidden_layers"] != 4:
        raise RuntimeError(f"{_MODEL_REPO} must be a 4-layer BF16 slice")
    if "quantization_config" in config or "quantization_config" in text_config:
        raise RuntimeError(f"{_MODEL_REPO} must not carry a quantized weight configuration")
    (Path(args.hf_checkpoint) / "processor_config.json").write_text(json.dumps(_PROCESSOR_CONFIG, indent=2))

    _build_image_sft_fixture(args)

    # Language-only conversion. The tower is loaded straight from the HF checkpoint on
    # the first pipeline stage, which is what makes the first launch exercise the
    # hide-the-tower-during-load branch.
    U.convert_checkpoint(
        model_name=_MEGATRON_MODEL_TYPE,
        megatron_model_type=_MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ref_load,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args=(
            "--tensor-model-parallel-size 2 --pipeline-model-parallel-size 2 "
            "--expert-model-parallel-size 2 --expert-tensor-parallel-size 1"
        ),
    )


def execute(args: ScriptArgs):
    checkpoints = Path(args.output_dir) / "checkpoints"

    # First launch: the reference conversion holds no visual.* keys, so the load has to
    # detach the tower to keep them from arriving as unexpected.
    _launch(args)

    # Second launch against the same output directory. latest_checkpointed_iteration now
    # points at a checkpoint that does carry visual.*, so this one takes the other
    # branch and trains on from there. Nothing else covers that path.
    _launch(dataclasses.replace(args, num_rollout=args.num_rollout + 1))

    iterations = [0, 1, 2]
    assert (checkpoints / "iter_0000002").is_dir(), "the second launch must resume and save a third iteration"
    assert int((checkpoints / "latest_checkpointed_iteration.txt").read_text().strip()) == 2

    metadata = _metadata(checkpoints, 1)
    visual_keys = [key for key in metadata if key.split(".")[0] == "visual"]
    assert len(visual_keys) == _EXPECTED_VISUAL_KEYS, f"expected {_EXPECTED_VISUAL_KEYS}, got {len(visual_keys)}"

    # A resumable save, not just weights: without these the second launch could not pick
    # the optimizer state back up.
    assert any("optimizer" in key for key in metadata)
    assert any("rng_state" in key for key in metadata)

    # The trainable scope has to move, across the fresh step *and* the resumed one. A
    # tower that only updated before the restart would pass a weaker check.
    trainable = _float_tensors(metadata, _PROJECTOR_PREFIXES)
    assert trainable, "projector parameters must be in the checkpoint"
    moved = {key: _max_abs_update(checkpoints, metadata, key, iterations) for key in trainable}

    # Every weight matrix must move -- those carry essentially all of the projector's
    # 159,396,864 parameters, and they are what "the projector trained" has to mean.
    # The 1-D RMSNorm scales are deliberately not required to: they sit at 1.0, where
    # bf16 cannot represent the 3.81e-06 update the biases show, so they stay put while
    # training is working perfectly. Asserting on them would make this test's outcome
    # depend on the learning rate.
    matrices = [key for key in trainable if len(metadata[key].size) > 1]
    assert matrices, "the projector must contribute weight matrices"
    unmoved = {key: moved[key] for key in matrices if moved[key] == 0}
    assert not unmoved, f"projector weight matrices did not change: {unmoved}"

    # The frozen half is saved but must not move: "projector" has to mean projector. This
    # reads the ViT's own projection weights, which is what makes it an answer to "did the
    # tower train when it should not have" rather than a statement about its norms.
    frozen = _float_tensors(metadata, _FROZEN_PREFIXES)
    assert frozen, "frozen visual parameters must still be in the checkpoint"
    for key in frozen:
        update = _max_abs_update(checkpoints, metadata, key, iterations)
        assert update == 0, f"{key} is outside --vision-trainable projector but moved by {update}"
