"""Audited visual-only evaluation snapshots for explicitly frozen GLM language models."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

_AUDITS = {}


def validate_snapshot_args(args) -> None:
    if not getattr(args, "glm5_visual_snapshot_only", False):
        return
    if not getattr(args, "glm5_freeze_language", False):
        raise ValueError("Visual-only snapshots require --glm5-freeze-language")
    if getattr(args, "glm5_vision_trainable", "none") == "none":
        raise ValueError("Visual-only snapshots require trainable visual parameters")
    if not getattr(args, "save", None):
        raise ValueError("Visual-only snapshots require a save directory")
    if getattr(args, "save_hf", None) or getattr(args, "custom_megatron_post_save_hook_path", None):
        raise ValueError("Visual-only snapshots cannot be combined with full HF export or a native post-save hook")
    if getattr(args, "async_save", False):
        raise ValueError("Visual-only snapshots are synchronous; disable --async-save")


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _named_parameters(model_chunks, visual):
    return [
        (f"chunk{index}.{name}", parameter)
        for index, chunk in enumerate(model_chunks)
        for name, parameter in sorted(_unwrap(chunk).named_parameters())
        if name.startswith("visual.") == visual
    ]


def _hash_parameters(parameters):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        value = parameter.detach().contiguous().reshape(-1)
        # Bound staging memory even for a large embedding or expert tensor.
        for block in value.split(16 * 1024**2):
            digest.update(block.cpu().view(torch.uint8).numpy().tobytes())
        count += parameter.numel()
    return digest.hexdigest(), count


def _write_json(path, record):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _gradient_evidence(model_chunks):
    totals = []
    for _, parameter in _named_parameters(model_chunks, visual=False):
        if parameter.requires_grad or parameter.grad is not None or getattr(parameter, "main_grad", None) is not None:
            raise RuntimeError("Frozen language parameters unexpectedly have gradients")
    for _, parameter in _named_parameters(model_chunks, visual=True):
        if not parameter.requires_grad:
            continue
        gradient = getattr(parameter, "main_grad", None)
        gradient = parameter.grad if gradient is None else gradient
        if gradient is None or not torch.isfinite(gradient).all().item():
            raise RuntimeError("Trainable visual parameters have missing or nonfinite gradients")
        totals.append(gradient.detach().float().abs().sum())
    return torch.stack(totals).sum().item() if totals else 0.0


def initialize_frozen_audit(args, model_chunks, optimizer) -> None:
    """Capture initial language hashes and check visual gradients at each optimizer update."""
    validate_snapshot_args(args)
    key = str(Path(args.save).absolute())
    if key in _AUDITS:
        raise RuntimeError("A frozen-language audit was initialized twice")
    language_hash, language_parameters = _hash_parameters(_named_parameters(model_chunks, visual=False))
    visual_hash, visual_parameters = _hash_parameters(_named_parameters(model_chunks, visual=True))
    if any(parameter.requires_grad for _, parameter in _named_parameters(model_chunks, visual=False)):
        raise RuntimeError("The language model must be frozen before optimizer construction")
    audit = {
        "rank": dist.get_rank(),
        "initial_language_sha256": language_hash,
        "initial_visual_sha256": visual_hash,
        "language_parameters": language_parameters,
        "visual_parameters": visual_parameters,
        "successful_steps": 0,
        "max_visual_gradient_abs_sum": 0.0,
    }
    if optimizer is None:
        raise RuntimeError("Frozen pipeline ranks require Megatron's collective-aware stub optimizer")
    original_step = optimizer.step

    def audited_step(*inputs, **kwargs):
        gradient = _gradient_evidence(model_chunks)
        if visual_parameters and gradient <= 0:
            raise RuntimeError("The visual tower did not receive a nonzero gradient")
        result = original_step(*inputs, **kwargs)
        if result[0]:
            audit["successful_steps"] += 1
        audit["max_visual_gradient_abs_sum"] = max(audit["max_visual_gradient_abs_sum"], gradient)
        print(
            "GLM53_FROZEN_STEP "
            + json.dumps(
                {
                    "rank": audit["rank"],
                    "step": audit["successful_steps"],
                    "visual_gradient_abs_sum": gradient,
                    "update_successful": bool(result[0]),
                }
            ),
            flush=True,
        )
        return result

    optimizer.step = audited_step
    _AUDITS[key] = audit
    path = Path(args.save) / "frozen_audit" / f"initial_rank{dist.get_rank():05d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, audit)


def before_train_step(args, rollout_id, step_id, model, optimizer, scheduler) -> None:
    """Initialize the audit once after the actor has loaded its starting checkpoint."""
    if str(Path(args.save).absolute()) not in _AUDITS:
        initialize_frozen_audit(args, model, optimizer)


def save_visual_snapshot(args, iteration, model_chunks) -> None:
    """Save only the replicated visual tower after validating all frozen language ranks."""
    validate_snapshot_args(args)
    audit = _AUDITS.get(str(Path(args.save).absolute()))
    if audit is None:
        raise RuntimeError("Frozen language audit is missing")
    language_hash, count = _hash_parameters(_named_parameters(model_chunks, visual=False))
    if language_hash != audit["initial_language_sha256"] or count != audit["language_parameters"]:
        raise RuntimeError("Language parameters changed during visual-only training")
    visual_hash, _ = _hash_parameters(_named_parameters(model_chunks, visual=True))
    if audit["successful_steps"] < 1 or (audit["visual_parameters"] and visual_hash == audit["initial_visual_sha256"]):
        raise RuntimeError("Visual-only training has not produced a verified optimizer update")
    destination = Path(args.save) / f"iter_{iteration:07d}"
    destination.mkdir(parents=True, exist_ok=True)
    evidence = {
        **audit,
        "iteration": iteration,
        "final_language_sha256": language_hash,
        "final_visual_sha256": visual_hash,
        "language_unchanged": True,
    }
    _write_json(destination / f"frozen_audit_rank{dist.get_rank():05d}.json", evidence)
    if dist.get_rank() == 0:
        towers = [getattr(_unwrap(chunk), "visual", None) for chunk in model_chunks]
        towers = [tower for tower in towers if tower is not None]
        if len(towers) != 1:
            raise RuntimeError("Global rank zero must own exactly one replicated visual tower")
        state = {
            "model.visual." + name: value.detach().cpu().contiguous() for name, value in towers[0].state_dict().items()
        }
        temporary = destination / ".visual.safetensors.tmp"
        save_file(state, temporary)
        os.replace(temporary, destination / "visual.safetensors")
    dist.barrier()
    if dist.get_rank() == 0:
        _write_json(
            destination / "visual_snapshot.json",
            {
                "format": "glm5_visual_eval_snapshot_v1",
                "iteration": iteration,
                "vision_trainable": args.glm5_vision_trainable,
                "freeze_language": True,
                "resumable": False,
                "world_size": dist.get_world_size(),
                "ref_load": args.ref_load,
                "ref_ckpt_step": getattr(args, "ref_ckpt_step", None),
                "hf_checkpoint": args.hf_checkpoint,
                "visual_sha256": hashlib.sha256((destination / "visual.safetensors").read_bytes()).hexdigest(),
            },
        )
    dist.barrier()
