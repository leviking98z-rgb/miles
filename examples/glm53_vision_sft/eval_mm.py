#!/usr/bin/env python3
"""Evaluate a GLM-5.3 checkpoint on image QA the way Miles itself serves it.

Three steps, and every earlier attempt here dropped one of them:

1. Render the conversation through the chat template, so the model answers rather than
   continuing the document.
2. Run the Miles processor over text + images. This is the step that matters: a single
   `<|image|>` in the text is expanded into one token per merged patch -- hundreds of
   them -- and those are the slots SGLang fills with image embeddings. Passing one
   placeholder leaves the model with nowhere to put the picture, and it says so.
3. Hand the processed tensors to SGLang under its `processor_output` input format,
   together with the expanded `input_ids`. Routing through SGLang's own GLM processor
   instead lets it re-derive the expansion, and on this checkpoint the images never
   arrive -- the model answers "I don't see an image attached to your message".
"""

import argparse
import json
import os
import time
from pathlib import Path

IMAGE_TOKEN = "<|image|>"


def _rows(path: Path, limit: int, start_index: int = 0, end_index: int | None = None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if end_index is not None and end_index > len(rows):
        raise ValueError(f"Requested end index {end_index} exceeds {len(rows)} source rows")
    selected = rows[start_index:end_index]
    return selected[:limit] if limit else selected


def _sampling_params(args, count):
    parameters = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.top_p != 1.0:
        parameters["top_p"] = args.top_p
    if args.sampling_seed is None:
        return parameters
    return [{**parameters, "sampling_seed": args.sampling_seed + args.start_index + index} for index in range(count)]


def _text_only_row(row, no_image=False):
    modality = row.get("input_modality", "image")
    if modality not in ("image", "text"):
        raise ValueError(f"Unsupported input modality: {modality}")
    if modality == "text" and row.get("images"):
        raise ValueError("Text controls cannot carry images")
    if not no_image and modality == "image" and not row.get("images"):
        raise ValueError("Image questions must carry images")
    return no_image or modality == "text"


def _generate_modalities(engine, input_ids, image_data, sampling_params):
    outputs = [None] * len(input_ids)
    for has_image in (False, True):
        indices = [index for index, payload in enumerate(image_data) if (payload is not None) == has_image]
        if not indices:
            continue
        parameters = (
            [sampling_params[index] for index in indices] if isinstance(sampling_params, list) else sampling_params
        )
        kwargs = {"image_data": [image_data[index] for index in indices]} if has_image else {}
        generated = engine.generate(
            input_ids=[input_ids[index] for index in indices],
            sampling_params=parameters,
            **kwargs,
        )
        if len(generated) != len(indices):
            raise RuntimeError("Engine did not cover the modality batch")
        for index, output in zip(indices, generated, strict=True):
            outputs[index] = output
    return outputs


def _prepare_model(args):
    if args.visual_snapshot is not None:
        from examples.glm53_vision_sft.visual_eval_overlay import stage_visual_overlay

        return stage_visual_overlay(
            args.model,
            args.visual_snapshot,
            args.base_export_source,
            cache_root=args.cache_root,
            workers=args.cache_workers,
        )
    if args.no_local_cache:
        return args.model, 0.0
    from examples.glm53_vision_sft.cache_eval_model import cache_model

    return cache_model(Path(args.model), cache_root=args.cache_root, workers=args.cache_workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=12)
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/miles/glm53-vision-eval")
    parser.add_argument("--no-local-cache", action="store_true")
    parser.add_argument("--visual-snapshot", type=Path)
    parser.add_argument("--base-export-source", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--no-image", action="store_true", help="ablation: same prompts, no chart")
    args = parser.parse_args()
    if args.limit < 0 or args.start_index < 0 or min(args.parallel_workers, args.cache_workers) < 1:
        parser.error("Row limits and offsets must be nonnegative; worker count must be positive")
    if args.end_index is not None and args.end_index <= args.start_index:
        parser.error("--end-index must exceed --start-index")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Temperature must be nonnegative and top-p must be in (0, 1]")
    if args.output.exists():
        raise FileExistsError("Inspect existing predictions before rerunning evaluation")
    if args.visual_snapshot is not None and (args.base_export_source is None or args.no_local_cache):
        parser.error("Visual overlays require --base-export-source and local caching")
    if args.base_export_source is not None and args.visual_snapshot is None:
        parser.error("--base-export-source is only used with --visual-snapshot")
    if args.parallel_workers > 1:
        # The driver is CPU-only; each worker reserves all TP GPUs on a separate owned Ray node.
        from examples.glm53_vision_sft.eval_mm_parallel import evaluate_parallel

        evaluate_parallel(args)
        return

    started = time.monotonic()
    model, cache_seconds = _prepare_model(args)
    model = str(model)
    preprocessing_started = time.monotonic()
    rows = _rows(args.data, args.limit, args.start_index, args.end_index)
    if not rows:
        raise ValueError("The requested evaluation slice is empty")
    print(f"loaded {len(rows)} rows from {args.data}", flush=True)

    from PIL import Image
    from transformers import AutoTokenizer

    from miles.utils.processing_utils import load_processor

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    processor = load_processor(model)
    print(
        f"processor: {type(processor).__name__}  image_token={processor.image_token!r}",
        flush=True,
    )

    all_input_ids, all_images, expanded_ids = [], [], []
    for row in rows:
        question = row["messages"][0]["content"].replace("<image>", "")
        text_only = _text_only_row(row, args.no_image)
        text = question if text_only else f"{IMAGE_TOKEN}{question}"
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        all_input_ids.append(rendered)
        if text_only:
            all_images.append(None)
            if not args.no_image:
                expanded_ids.append(tokenizer.encode(rendered, add_special_tokens=False))
            continue
        # Hand SGLang the Miles processor's own output rather than a placeholder and a
        # file. SGLang ships a GLM multimodal processor, but routing through it means it
        # re-derives the expansion, and on this checkpoint the images never arrive -- the
        # model answers "I don't see an image attached". The `processor_output` input
        # format exists for exactly this: pass the processed tensors and the already
        # expanded ids, and its own processor steps aside. Exactly one item per request,
        # which `validate_mm_data` enforces.
        images = [Image.open(path).convert("RGB") for path in row["images"]]
        processed = processor(text=[rendered], images=images)
        ids = processed["input_ids"][0]
        expanded_ids.append(ids.tolist() if hasattr(ids, "tolist") else list(ids))
        payload = {"format": "processor_output", "modality": "IMAGE"}
        payload.update({key: value for key, value in processed.items() if key != "input_ids"})
        all_images.append([payload])

    lengths = [len(ids) for ids in (expanded_ids or all_input_ids)]
    print(
        f"prompt length: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}",
        flush=True,
    )
    preprocessing_seconds = time.monotonic() - preprocessing_started

    import sglang as sgl

    load_started = time.monotonic()
    engine = sgl.Engine(
        model_path=model,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        context_length=8192,
        disable_cuda_graph=True,
        trust_remote_code=True,
    )
    load_seconds = time.monotonic() - load_started
    try:
        inference_started = time.monotonic()
        if args.no_image:
            outputs = engine.generate(prompt=all_input_ids, sampling_params=_sampling_params(args, len(rows)))
        else:
            # SGLang infers multimodal payload format from the first batch element.
            # Keep text-only controls in a separate call, then restore source order.
            outputs = _generate_modalities(engine, expanded_ids, all_images, _sampling_params(args, len(rows)))
        inference_seconds = time.monotonic() - inference_started
    finally:
        engine.shutdown()

    if len(outputs) != len(rows):
        raise RuntimeError(f"Engine returned {len(outputs)} outputs for {len(rows)} inputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for row, output in zip(rows, outputs, strict=True):
            text = output["text"] if isinstance(output, dict) else str(output)
            out.write(
                json.dumps(
                    {
                        **{
                            key: row[key]
                            for key in ("id", "task", "control", "dataset_role", "sample_seed")
                            if key in row
                        },
                        "question": row["messages"][0]["content"],
                        "answer": row["answer"],
                        "prediction": text,
                        "finish_reason": (
                            output.get("meta_info", {}).get("finish_reason") if isinstance(output, dict) else None
                        ),
                        "completion_tokens": (
                            output.get("meta_info", {}).get("completion_tokens") if isinstance(output, dict) else None
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    os.replace(temporary, args.output)
    timing = {
        "n": len(rows),
        "start_index": args.start_index,
        "end_index": args.start_index + len(rows),
        "source_model": args.model,
        "loaded_model": model,
        "visual_snapshot": str(args.visual_snapshot) if args.visual_snapshot is not None else None,
        "base_export_source": str(args.base_export_source) if args.base_export_source is not None else None,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_seed": args.sampling_seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "cache_staging_seconds": round(cache_seconds, 3),
        "preprocessing_seconds": round(preprocessing_seconds, 3),
        "model_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(time.monotonic() - started, 3),
    }
    args.output.with_suffix(".timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(f"wrote {len(rows)} predictions to {args.output}", flush=True)
    print(json.dumps(timing), flush=True)


if __name__ == "__main__":
    main()
