"""Replay the measured direct-label, frozen-language geometric_stack_count recipe.

Requires an allocated 32-GPU external Ray cluster, a converted none-v2 iteration-95
language checkpoint, the matching HF visual checkpoint, and materialized scene data.
Does not start, stop, or clean up Ray or other jobs.

Args:
  --hf-checkpoint: HF checkpoint containing the unchanged starting visual tower.
  --ref-load: Converted language checkpoint directory containing iteration 95.
  --prompt-data: sft_exact.jsonl produced by prepare.py.
  --output-dir: A new output directory for the final audited visual snapshot.
  --megatron-path: Checkout used by the allocated cluster runtime.
  Remaining flags are inherited from the native image-SFT launcher.

Example:
  MILES_SCRIPT_EXTERNAL_RAY=1 RAY_ADDRESS=http://127.0.0.1:8265 \
    python -m examples.glm53_vision_sft.run_glm5_3_flash_corner \
    --hf-checkpoint /models/visual --ref-load /models/none-v2/checkpoints \
    --prompt-data /data/corner/sft_exact.jsonl --output-dir /runs/corner
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from scripts.run_glm5_3_flash_vision_sft import ScriptArgs as NativeScriptArgs
from scripts.run_glm5_3_flash_vision_sft import _train_request

import miles.utils.external_utils.command_utils as U
from miles.utils.external_utils.command_utils.common import _pythonpath_with_sources
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.external_utils.ray_job import run_ray_job

_RECORDED = json.loads((Path(__file__).parent / "recorded/result.json").read_text())
_TRAIN = _RECORDED["training"]
_ENV_KEYS = (
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "OMP_NUM_THREADS",
    "NCCL_NVLS_ENABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "HF_HOME",
    "WANDB_API_KEY",
    "WANDB_MODE",
    "WANDB_DIR",
    "no_proxy",
    "NO_PROXY",
)


@dataclass
class ScriptArgs(NativeScriptArgs):
    model_name: str = _RECORDED["model"]
    num_nodes: int = _TRAIN["num_nodes"]
    num_gpus_per_node: int = _TRAIN["num_gpus_per_node"]
    num_rollout: int = _TRAIN["updates"]
    save_interval: int = _TRAIN["updates"]
    rollout_batch_size: int = _TRAIN["batch_size"]
    global_batch_size: int = _TRAIN["batch_size"]
    max_tokens_per_gpu: int = _TRAIN["max_tokens_per_gpu"]
    lr: float = _TRAIN["lr"]
    rollout_seed: int = _TRAIN["rollout_seed"]
    rotary_base: int | None = _TRAIN["rotary_base"]
    vision_trainable: str = _TRAIN["vision_trainable"]
    freeze_language: bool = _TRAIN["freeze_language"]
    ref_ckpt_step: int = _RECORDED["baseline"]["iteration"]
    seed: int = _TRAIN["seed"]

    def __post_init__(self):
        super().__post_init__()
        if not self.freeze_language or self.vision_trainable != "full":
            raise ValueError("This reproduction trains only the full visual tower with the LLM frozen")
        if self.num_rollout != self.save_interval:
            raise ValueError("This reproduction saves only the final snapshot")
        if self.ref_ckpt_step < 0:
            raise ValueError("Reference checkpoint iteration must be nonnegative")


def execute(args: ScriptArgs) -> None:
    if os.environ.get("MILES_SCRIPT_EXTERNAL_RAY") != "1" or not os.environ.get("RAY_ADDRESS"):
        raise ValueError(
            "Join the allocated Ray cluster externally and set MILES_SCRIPT_EXTERNAL_RAY=1 and RAY_ADDRESS"
        )
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Use a new output directory; visual-only snapshots cannot resume training")
    if not (Path(args.ref_load) / f"iter_{args.ref_ckpt_step:07d}/.metadata").is_file():
        raise FileNotFoundError("The pinned converted starting language checkpoint is missing")
    data = Path(args.prompt_data)
    materialization = json.loads((data.parent / "materialization.json").read_text())
    if (
        materialization["training_rows"] != _TRAIN["examples"]
        or not materialization["all_pixels_match"]
        or not materialization["exact_targets_only"]
        or materialization["oracle_in_model_input"]
        or hashlib.sha256(data.read_bytes()).hexdigest() != materialization["files_sha256"][data.name]
    ):
        raise ValueError("Training data differs from the verified materialization")
    request = _train_request(args)
    required = (
        f"--seed {args.seed} --ref-ckpt-step {args.ref_ckpt_step} " "--no-save-optim --glm5-visual-snapshot-only"
    )
    entrypoint = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(U.repo_base_dir / request['train_script']))} "
        f"{shell_safe_model_args(request['megatron_model_type'])} {request['train_args']} {required}"
    )
    environment = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
    environment.update(
        U.resolve_extra_env_vars(request["extra_env_vars"], args),
        NVTE_FP8_BLOCK_SCALING_FP32_SCALES="1",
        RAY_DEDUP_LOGS="0",
        RAY_memory_monitor_refresh_ms="0",
        PYTHONUNBUFFERED="1",
        MASTER_ADDR=os.environ.get("MASTER_ADDR", os.environ.get("MILES_NODE_IP", "127.0.0.1")),
    )
    environment["PYTHONPATH"] = _pythonpath_with_sources(args.megatron_path, os.environ.get("PYTHONPATH"))
    # Reuse native argv and Ray submission, but never run the generic launcher's global
    # pkill preamble on an externally managed allocation.
    run_ray_job(entrypoint=entrypoint, runtime_env={"env_vars": environment}, job_lifetime="launcher")


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
