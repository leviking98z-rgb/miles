"""Build a node-local HF view with audited visual weights and unchanged language shards."""

import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from examples.glm53_vision_sft.cache_eval_model import (
    DEFAULT_CACHE,
    _copy_verified,
    _digest,
    _inventory,
    _write_manifest,
    cache_model,
)

VISUAL_PREFIX = "model.visual."
VISUAL_FILE = "trained-visual.safetensors"


def _snapshot_evidence(snapshot, base_export_source):
    manifest = json.loads((snapshot / "visual_snapshot.json").read_text())
    provenance = json.loads(Path(base_export_source).read_text())
    if (
        manifest.get("format") != "glm5_visual_eval_snapshot_v1"
        or manifest.get("freeze_language") is not True
        or manifest.get("resumable") is not False
        or manifest.get("vision_trainable") != "full"
    ):
        raise ValueError("Expected a final, audited full-visual frozen-language snapshot")
    reference = Path(manifest["ref_load"]) / f"iter_{manifest['ref_ckpt_step']:07d}"
    if (
        Path(provenance["checkpoint"]).absolute() != reference.absolute()
        or provenance["checkpoint_iteration"] != manifest["ref_ckpt_step"]
    ):
        raise ValueError("Baseline export and snapshot use different language checkpoints")
    world_size = manifest.get("world_size")
    if not isinstance(world_size, int) or world_size < 1:
        raise ValueError("The snapshot has no valid distributed world size")
    audit_paths = sorted(snapshot.glob("frozen_audit_rank*.json"))
    if len(audit_paths) != world_size:
        raise ValueError("The final snapshot does not cover every distributed rank")
    visual_ranks = 0
    visual_hashes = set()
    for rank, path in enumerate(audit_paths):
        audit = json.loads(path.read_text())
        if (
            audit["rank"] != rank
            or audit["iteration"] != manifest["iteration"]
            or audit.get("language_unchanged") is not True
            or audit["initial_language_sha256"] != audit["final_language_sha256"]
            or audit["successful_steps"] < 1
        ):
            raise ValueError(f"Invalid frozen-language audit: {path}")
        if audit["visual_parameters"]:
            visual_ranks += 1
            visual_hashes.add(audit["final_visual_sha256"])
            if (
                audit["initial_visual_sha256"] == audit["final_visual_sha256"]
                or audit["max_visual_gradient_abs_sum"] <= 0
            ):
                raise ValueError(f"The visual parameters did not update: {path}")
    if not visual_ranks:
        raise ValueError("No distributed rank trained visual parameters")
    if len(visual_hashes) != 1:
        raise ValueError("Visual replicas disagree on their final parameter hash")
    if _digest(snapshot / "visual.safetensors") != manifest["visual_sha256"]:
        raise ValueError("Visual snapshot bytes differ from the final manifest")
    return manifest


def _verified_base_cache(source, cache_root, workers):
    entries, fingerprint = _inventory(source)
    destination = cache_root / f"{source.name}-{fingerprint[:16]}"
    manifest_path = destination / ".cache_manifest.json"
    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text())
        verified_at = manifest_path.stat().st_mtime_ns
        valid = record.get("complete") is True and record.get("fingerprint") == fingerprint
        for name, entry in entries.items():
            path = destination / name
            previous = record.get("files", {}).get(name, {})
            if not path.is_file():
                valid = False
                break
            stat = path.stat()
            valid &= (
                previous.get("size") == entry["size"] == stat.st_size
                and previous.get("mtime_ns") == entry["mtime_ns"]
                and bool(previous.get("sha256"))
                and stat.st_mtime_ns <= verified_at
            )
        # Reuse a completed byte-verified cache only while its source and local
        # file identities remain stable. Do not rehash 321 GB for each visual view.
        if valid and _digest(destination / "model.safetensors.index.json") == _digest(
            source / "model.safetensors.index.json"
        ):
            return destination, fingerprint
    destination, _ = cache_model(source, cache_root=cache_root, workers=workers)
    return destination, fingerprint


def _validate_visual_schema(base, visual):
    # Safetensors is optional for ordinary orchestration; only overlays need it.
    from safetensors import safe_open

    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    visual_keys = {key for key in weight_map if key.startswith(VISUAL_PREFIX)}
    if not visual_keys:
        raise ValueError("The baseline contains no visual weights")
    visual_shards = {weight_map[key] for key in visual_keys}
    if any(not key.startswith(VISUAL_PREFIX) and shard in visual_shards for key, shard in weight_map.items()):
        raise ValueError("The baseline must have dedicated visual-only shards")
    schema = {}
    for name in sorted(visual_shards):
        with safe_open(base / name, framework="pt", device="cpu") as tensors:
            expected = {key for key in visual_keys if weight_map[key] == name}
            if set(tensors.keys()) != expected:
                raise ValueError("Baseline visual shard keys differ from its index")
            for key in tensors.keys():
                tensor = tensors.get_slice(key)
                schema[key] = (tensor.get_shape(), tensor.get_dtype())
    with safe_open(visual, framework="pt", device="cpu") as tensors:
        if set(tensors.keys()) != visual_keys:
            raise ValueError("Snapshot visual keys differ from the baseline")
        for key in tensors.keys():
            tensor = tensors.get_slice(key)
            if (tensor.get_shape(), tensor.get_dtype()) != schema[key]:
                raise ValueError(f"Snapshot shape or dtype differs: {key}")
    return index, visual_shards


def stage_visual_overlay(source, snapshot, base_export_source, cache_root=DEFAULT_CACHE, workers=12):
    started = time.monotonic()
    source, snapshot, cache_root = Path(source), Path(snapshot), Path(cache_root)
    manifest = _snapshot_evidence(snapshot, base_export_source)
    base, base_fingerprint = _verified_base_cache(source, cache_root, workers)
    index, visual_shards = _validate_visual_schema(base, snapshot / "visual.safetensors")
    identity = {
        "format": "glm5_visual_hf_overlay_v1",
        "base_fingerprint": base_fingerprint,
        "snapshot": manifest,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache_root / f"{source.name}-visual-{fingerprint[:16]}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".overlay.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _stage_view(base, snapshot, destination, index, visual_shards, identity)
    seconds = round(time.monotonic() - started, 3)
    print(
        json.dumps(
            {
                "stage": "visual_overlay",
                "base": str(base),
                "snapshot": str(snapshot),
                "destination": str(destination),
                "seconds": seconds,
            }
        ),
        flush=True,
    )
    return destination, seconds


def _stage_view(base, snapshot, destination, index, visual_shards, identity):
    language_shards = set(index["weight_map"].values()) - visual_shards
    for name in sorted(language_shards):
        path = destination / name
        if path.exists():
            if not os.path.samefile(base / name, path):
                raise ValueError(f"Overlay language shard is not the base file: {path}")
        else:
            os.link(base / name, path)
    for path in base.iterdir():
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.suffix != ".safetensors"
            and path.name not in ("model.safetensors.index.json", "vision_export_validation.json")
        ):
            shutil.copy2(path, destination / path.name)
    stat = (snapshot / "visual.safetensors").stat()
    record, _ = _copy_verified(
        snapshot / "visual.safetensors",
        destination / VISUAL_FILE,
        {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        {},
    )
    if record["sha256"] != identity["snapshot"]["visual_sha256"]:
        raise ValueError("The visual snapshot changed during overlay staging")
    remapped = {
        key: VISUAL_FILE if key.startswith(VISUAL_PREFIX) else name for key, name in index["weight_map"].items()
    }
    _write_manifest(destination / "model.safetensors.index.json", {**index, "weight_map": remapped})
    _write_manifest(
        destination / "visual_overlay.json",
        {
            **identity,
            "source_snapshot": str(snapshot.absolute()),
            "base_model": str(base),
            "complete": True,
            "visual_file": record,
            "language_shards_reused": len(language_shards),
        },
    )
