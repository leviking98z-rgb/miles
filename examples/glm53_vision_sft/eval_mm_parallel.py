#!/usr/bin/env python3
"""Evaluate ordered, disjoint image-QA slices on independent TP replicas."""

import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from examples.glm53_vision_sft.eval_mm import _rows
from miles.utils.external_utils.command_utils.common import run_process

WORKER_ENV_KEYS = (
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "HF_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCHINDUCTOR_COMPILE_THREADS",
    "OMP_NUM_THREADS",
    "NCCL_NVLS_ENABLE",
    "NCCL_IB_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES",
)


def shard_ranges(start, count, workers):
    if count <= 0 or workers <= 0 or start < 0:
        raise ValueError("Sharding needs a positive row count and worker count")
    workers = min(workers, count)
    return [(start + count * index // workers, start + count * (index + 1) // workers) for index in range(workers)]


def _command(args, output, start, end):
    command = [
        sys.executable,
        "-m",
        "examples.glm53_vision_sft.eval_mm",
        "--model",
        args.model,
        "--data",
        str(args.data),
        "--output",
        str(output),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--tp-size",
        str(args.tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cache-workers",
        str(args.cache_workers),
        "--cache-root",
        str(args.cache_root),
        "--parallel-workers",
        "1",
        "--start-index",
        str(start),
        "--end-index",
        str(end),
    ]
    if args.no_image:
        command.append("--no-image")
    if args.no_local_cache:
        command.append("--no-local-cache")
    if args.temperature != 0.0:
        command.extend(["--temperature", str(args.temperature)])
    if args.top_p != 1.0:
        command.extend(["--top-p", str(args.top_p)])
    if args.sampling_seed is not None:
        command.extend(["--sampling-seed", str(args.sampling_seed)])
    if getattr(args, "visual_snapshot", None) is not None:
        command.extend(
            [
                "--visual-snapshot",
                str(args.visual_snapshot),
                "--base-export-source",
                str(args.base_export_source),
            ]
        )
    return command


def _run_shard_child(command, log_path, environment):
    os.setsid()
    with Path(log_path).open("a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.environ.update(environment)
        try:
            result = run_process(command, capture_output=False, check=False)
        except BaseException:
            traceback.print_exc()
            raise
    raise SystemExit(result.returncode if result.returncode >= 0 else 1)


def _stop_shard(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            if not process.is_alive():
                break
            # Cancellation may arrive before the spawned child has called setsid.
            process.terminate() if sig == signal.SIGTERM else process.kill()
        process.join(timeout=30)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            if not process.is_alive():
                break
    if process.is_alive():
        raise RuntimeError(f"Evaluation subprocess {process.pid} did not exit")


def _run_shard(command, log_path, environment):
    started = time.monotonic()
    # Spawn avoids forking Ray's threaded worker. The private process group contains
    # only this eval, never Ray or another job on the allocation.
    process = multiprocessing.get_context("spawn").Process(
        target=_run_shard_child,
        args=(command, log_path, environment),
    )
    process.start()
    try:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"TP evaluation worker failed; log: {log_path}")
    finally:
        _stop_shard(process)
    return {"seconds": round(time.monotonic() - started, 3), "log": log_path}


def merge_predictions(shards, sources, output):
    rows = []
    for start, end, path in sorted(shards):
        predictions = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(predictions) != end - start:
            raise ValueError(f"Shard {path} has {len(predictions)} of {end - start} required predictions")
        rows.extend(predictions)
    if len(rows) != len(sources):
        raise ValueError("Merged predictions do not cover the selected input")
    for row, source in zip(rows, sources, strict=True):
        if (
            row["id"] != source["id"]
            or row.get("control", "image") != source.get("control", "image")
            or row["question"] != source["messages"][0]["content"]
            or row["answer"] != source["answer"]
        ):
            raise ValueError("Merged predictions differ from the input order")
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    return rows


def evaluate_parallel(args):
    # Optional Ray dependency is needed only for the multi-node orchestration path.
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    started = time.monotonic()
    sources = _rows(args.data, args.limit, args.start_index, args.end_index)
    ranges = shard_ranges(args.start_index, len(sources), args.parallel_workers)
    ray.init(address=os.environ["RAY_ADDRESS"], ignore_reinit_error=True)
    references = []
    completed = False
    try:
        nodes = sorted(
            (node for node in ray.nodes() if node.get("Alive") and node["Resources"].get("GPU", 0) >= args.tp_size),
            key=lambda node: node["NodeID"],
        )
        if len(nodes) < len(ranges):
            raise RuntimeError(f"Evaluation needs {len(ranges)} TP nodes, but only {len(nodes)} are alive")
        if ray.available_resources().get("GPU", 0) < len(ranges) * args.tp_size:
            raise RuntimeError("Training or conversion has not released enough GPUs for parallel evaluation")
        directory = args.output.with_suffix(".parts")
        directory.mkdir(parents=True, exist_ok=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        environment = {key: os.environ[key] for key in WORKER_ENV_KEYS if key in os.environ}
        worker = ray.remote(num_gpus=args.tp_size, num_cpus=16, max_retries=0)(_run_shard)
        shards, placement = [], []
        for index, ((start, end), node) in enumerate(zip(ranges, nodes, strict=False)):
            output = directory / f"part-{index:02d}.jsonl"
            log = directory / f"part-{index:02d}.log"
            reference = worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
            ).remote(_command(args, output, start, end), str(log), environment)
            references.append(reference)
            shards.append((start, end, output))
            placement.append({"node_id": node["NodeID"], "start": start, "end": end, "log": str(log)})
        print(
            json.dumps(
                {
                    "stage": "parallel_eval",
                    "workers": len(ranges),
                    "placement": placement,
                }
            ),
            flush=True,
        )
        workers = ray.get(references)
        rows = merge_predictions(shards, sources, args.output)
        timing = {
            "n": len(rows),
            "workers": len(ranges),
            "tp_size": args.tp_size,
            "total_seconds": round(time.monotonic() - started, 3),
            "worker_timings": [json.loads(path.with_suffix(".timing.json").read_text()) for _, _, path in shards],
            "worker_wall_seconds": workers,
            "placement": placement,
        }
        args.output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(json.dumps(timing), flush=True)
        completed = True
    finally:
        if references and not completed:
            # Cooperative cancellation lets each worker clean up its inference subprocesses.
            for reference in references:
                ray.cancel(reference, force=False)
            _, pending = ray.wait(references, num_returns=len(references), timeout=90)
            for reference in pending:
                ray.cancel(reference, force=True)
        ray.shutdown()
