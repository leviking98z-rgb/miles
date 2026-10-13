#!/usr/bin/env python3
"""Copy a finalized HF export to verified node-local storage before inference."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

BLOCK_BYTES = 16 * 1024**2
DEFAULT_CACHE = Path.home() / ".cache/miles/glm53-vision-eval"


def prefetch_directory(source, cache_root=DEFAULT_CACHE):
    source = Path(source)
    namespace = hashlib.sha256(str(source.absolute()).encode()).hexdigest()[:16]
    return Path(cache_root) / ".prefetch" / f"{source.name}-{namespace}"


def file_identity(path):
    stat = Path(path).stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def prefetch_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _inventory(source):
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("The model index must contain a nonempty weight_map")
    shards = sorted(set(weight_map.values()))
    for name in shards:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Unsafe model shard path: {name!r}")
    files = [source / name for name in shards] + sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix != ".safetensors"
    )
    if not (source / "config.json").is_file():
        raise ValueError("The export has no config.json")
    entries = {}
    for path in files:
        stat = path.stat()
        entries[path.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    identity = {
        # The shared path resolves differently outside the runtime's bind mount.
        "source": str(source.absolute()),
        "index_sha256": _digest(index_path),
        "files": entries,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return entries, fingerprint


def _adopt_prefetched(source, destination, entry, directory):
    if directory is None or source.suffix != ".safetensors":
        return None
    identity = file_identity(source)
    if any(identity[key] != entry[key] for key in ("size", "mtime_ns")):
        return None
    key = prefetch_key(identity)
    cached = directory / f"{key}.safetensors"
    manifest = directory / f"{key}.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.adopt")
    try:
        record = json.loads(manifest.read_text())
        if (
            record.get("identity") != identity
            or cached.stat().st_size != entry["size"]
            or record.get("sha256") != _digest(cached)
        ):
            return None
        # Same-filesystem rename preserves the source inode through export finalization.
        # Reuse only that exact file identity, then verify the node-local bytes again.
        os.link(cached, temporary)
        if file_identity(source) != identity:
            return None
        os.replace(temporary, destination)
        return {**entry, "sha256": record["sha256"], "prefetched": True}
    except (OSError, ValueError, KeyError):
        return None
    finally:
        temporary.unlink(missing_ok=True)


def _copy_verified(source, destination, entry, previous, prefetch=None):
    if (
        previous.get("size") == entry["size"]
        and previous.get("mtime_ns") == entry["mtime_ns"]
        and destination.is_file()
        and destination.stat().st_size == entry["size"]
        and previous.get("sha256") == _digest(destination)
    ):
        return previous, 0
    adopted = _adopt_prefetched(source, destination, entry, prefetch)
    if adopted is not None:
        return adopted, 0
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            while block := reader.read(BLOCK_BYTES):
                writer.write(block)
                digest.update(block)
                copied += len(block)
            writer.flush()
            os.fsync(writer.fileno())
        after = source.stat()
        if copied != entry["size"] or after.st_size != entry["size"] or after.st_mtime_ns != entry["mtime_ns"]:
            raise RuntimeError(f"The source changed during cache staging: {source}")
        if _digest(temporary) != digest.hexdigest():
            raise RuntimeError(f"Cached bytes differ from their source: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {**entry, "sha256": digest.hexdigest()}, copied


def _write_manifest(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def cache_model(source, cache_root=DEFAULT_CACHE, workers=12):
    if workers < 1:
        raise ValueError("Cache worker count must be positive")
    source = Path(source)
    cache_root = Path(cache_root)
    entries, fingerprint = _inventory(source)
    destination = cache_root / f"{source.name}-{fingerprint[:16]}"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _stage(source, destination, entries, fingerprint, workers)


def _stage(source, destination, entries, fingerprint, workers):
    started = time.monotonic()
    manifest_path = destination / ".cache_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_files = previous.get("files", {})
    record = {
        "source": str(source.absolute()),
        "fingerprint": fingerprint,
        "complete": False,
        "files": {},
    }
    remaining = sum(entry["size"] for name, entry in entries.items() if not (destination / name).is_file())
    if shutil.disk_usage(destination).free < remaining + 2 * 1024**3:
        raise RuntimeError(f"Insufficient local disk space for {source}")
    copied_bytes = 0
    prefetched_bytes = 0
    total_bytes = sum(entry["size"] for entry in entries.values())
    last_log = started
    print(
        json.dumps(
            {
                "stage": "model_cache",
                "destination": str(destination),
                "files": len(entries),
                "bytes": total_bytes,
                "workers": workers,
            }
        ),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _copy_verified,
                source / name,
                destination / name,
                entry,
                previous_files.get(name, {}),
                prefetch_directory(source, destination.parent),
            ): name
            for name, entry in entries.items()
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            result, copied = future.result()
            record["files"][name] = result
            copied_bytes += copied
            if result.get("prefetched"):
                prefetched_bytes += result["size"]
            now = time.monotonic()
            if now - last_log >= 15 or len(record["files"]) == len(entries):
                _write_manifest(manifest_path, record)
                print(
                    json.dumps(
                        {
                            "stage": "model_cache_progress",
                            "completed": len(record["files"]),
                            "files": len(entries),
                            "copied_bytes": copied_bytes,
                            "prefetched_bytes": prefetched_bytes,
                            "seconds": round(now - started, 3),
                            "copy_mb_per_second": round(copied_bytes / max(now - started, 0.001) / 1e6, 2),
                        }
                    ),
                    flush=True,
                )
                last_log = now
    record["complete"] = True
    record["prefetched_bytes"] = prefetched_bytes
    record["staging_seconds"] = round(time.monotonic() - started, 3)
    _write_manifest(manifest_path, record)
    return destination, record["staging_seconds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    destination, seconds = cache_model(args.model, args.cache_root, args.workers)
    print(
        json.dumps({"cached_model": str(destination), "staging_seconds": seconds}),
        flush=True,
    )


if __name__ == "__main__":
    main()
