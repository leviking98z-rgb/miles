---
title: GLM-5.3-Flash
description: RL and BF16 image SFT recipes for GLM-5.3-Flash, a KDA + DSA hybrid MoE with mHC hyper-connections and NoPE MLA.
---

Implementation: [`radixark/miles#2786`](https://github.com/radixark/miles/pull/2786). It goes
with the SGLang
[`sglang-miles-glm53next`](https://github.com/sgl-project/sglang/tree/sglang-miles-glm53next)
branch and [`radixark/Megatron-LM#89`](https://github.com/radixark/Megatron-LM/pull/89); the
image in section 3 pins all three.

## 1. Model Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`model_type: glm5_next`) is a
**45-layer KDA + DSA hybrid MoE**. It is a different architecture from the 744 B GLM5 and
GLM5.2 flagships, not a smaller cut of them.

- **45 layers, hybrid**: 34 KDA linear-attention layers + 11 DSA sparse-attention layers.
- **288-expert MoE**, sigmoid routing at top-8; the first 3 layers are dense.
- **mHC hyper-connections** at every block.
- **NoPE MLA** — multi-latent attention with the positional half of the QK head empty.
- **kpool-compressed lightning indexer** picks which keys the DSA layers attend.
- Hidden 4096, FFN 12288, 64 attention heads, vocab 154880, rotary base 800000.
- MTP is dropped for training.

## 2. Supported Variants

| Variant | `--model-name` | Layers |
|---|---|---|
| Full | `GLM-5.3-Flash` | 45 |
| 4-layer slice | `GLM-5.3-Flash-4layer` | 4 (launcher default) |

## 3. Environment Setup

Use `docker.io/radixark/miles:glm53next` — the rolling `radixark/miles:dev` image with the
three moving parts checked out at the versions this recipe was built against, multi-arch so
the same tag serves GB300 and x86 nodes.

| Component | Pinned at |
|---|---|
| miles | [`#2786`](https://github.com/radixark/miles/pull/2786) `1cd14c00` |
| SGLang | `sglang-miles-glm53next` `9a26e749` |
| Megatron-LM | [`#89`](https://github.com/radixark/Megatron-LM/pull/89) `e8f57451` |

```bash
hf download zai-org/GLM-5.3-Flash --local-dir /root/models/GLM-5.3-Flash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

The reference checkpoint has to be converted first — `--ref-load` resolves to
`<--ckpt-dir>/<megatron_model_type>_torch_dist`:

```bash
cd /root/miles
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py glm5.3-flash)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
CONVERT_KEEP_PP1=1 CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/root/Megatron-LM \
  torchrun --nproc-per-node 8 \
    tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/models/GLM-5.3-Flash \
    --save          /root/ckpt/glm5.3-flash_torch_dist
```

## 4. Launch

Bring up a ray cluster across the nodes, `export MILES_SCRIPT_EXTERNAL_RAY=1`, then on the
head node:

```bash
cd /root/miles
python scripts/run_glm5_3_flash.py train \
  --model-name GLM-5.3-Flash \
  --num-nodes 16 --num-gpus-per-node 4 \
  --num-rollout 20 --rollout-max-response-len 4096
```

Smoke slice on one node:

```bash
python scripts/run_glm5_3_flash.py train --num-nodes 1 --num-gpus-per-node 8
```

| Shape | TP | PP | EP | Rollout engine |
|---|---|---|---|---|
| 16 × 4 (full) | 8 | 4 | 16 | 8 GPUs, SGLang TP 8 / EP 8 |
| 8 × 4 (full) | 8 | 4 | 8 | 8 GPUs, SGLang TP 8 / EP 8 |
| 2 × 4 or 1 × 8 (4-layer slice) | 2 | 2 | 2 | 4 GPUs, SGLang TP 4 / EP 4 |

The PP-4 shapes run 11 / 11 / 11 / 12 layers per stage, since 45 does not divide by 4.
GRPO on DAPO-Math-17k, Adam at `lr 1e-6`, `max_tokens_per_gpu 8192`, full uniform recompute.
Rollout is colocated, with the trainer offloaded to disk; both DSA paths run on tilelang and
the KV cache is BF16. Routing replay turns on with `--enable-r3`.

## 5. What a Healthy Run Looks Like

From the validation run in [#2786](https://github.com/radixark/miles/pull/2786) — 16 nodes ×
4 GB300, DAPO on DAPO-Math-17k:

| Metric | Observed |
|---|---|
| `train/train_rollout_logprob_abs_diff` | 0.0068 – 0.0106 across the first 11 rollouts |
| `rollout/raw_reward` | 0.5 → 0.94 within 10 rollouts |
| `train/ppo_kl` | ~2.6e-4 |
| `train/grad_norm` | 0.31 – 0.49 |

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up: it
covers the KDA, DSA and hyper-connection paths at once.

## 6. BF16 Image SFT

`scripts/run_glm5_3_flash_vision_sft.py` trains GLM-5.3-Flash from image-text
conversations. `--vision-trainable` selects how much of the visual tower trains
alongside the language model; it defaults to `projector`.
Measured runs, loss curves and their limits:
[glm5-3-flash-vision-experiments.md](glm5-3-flash-vision-experiments.md).
It runs `train_async.py --debug-train-only`: no generation, SGLang engine, reward
model, or evaluation is started.

| `--vision-trainable` | Trains | Visual parameters |
|---|---|---|
| `none` | language only | 0 of 563,627,008 |
| `projector` (default) | `post_layernorm`, `downsample`, `merger` | 159,396,864 |
| `full` | also `patch_embed` and the 24 ViT blocks | 563,627,008 |

`full` also needs `--glm5-vision-recompute`, which the launcher sets for that scope
(pass `--vision-recompute` / `--no-vision-recompute` to override):
at GEO3K's image sizes the 24 ViT blocks hold about 5 GiB of activations per rank,
which overflows a 95 GiB H20, and recomputing them costs about 1.3 GiB instead.
(Transformers' own `gradient_checkpointing_enable()` does reach this tower, but it
gates on training mode alone and so also wraps forwards no backward pass consumes.
Miles selects a recomputing block class instead, which ties the cost to the
trainable scope and the autograd state; the two produce identical numbers.)

With `none` the tower stays outside the optimizer and the distributed checkpoint.
Any other scope registers it as the `visual` submodule, so its trainable
parameters join the optimizer, the DDP gradient buckets and the checkpoint, and
`visual.*` keys appear in the saved checkpoint.

Use a BF16 HF checkpoint containing `model.visual.*`, `vision_config`, tokenizer
files, `chat_template.jinja`, and `processor_config.json`. The converted Megatron
checkpoint contains only language parameters and uses the existing GLM-5.3
conversion recipe. The visual weights are loaded strictly from the HF checkpoint
on the first pipeline stage. The compatibility processor supports the environment's
fixed `transformers==5.12.1`; upgrading Transformers is not required.

### Dataset

Each JSONL row, or Parquet record, contains a `messages` conversation and an
`images` **list**. Use one `<image>` placeholder per image, in the same order.
Image paths must be accessible on the training nodes.

```json
{
  "messages": [
    {"role": "user", "content": "<image>What number is written in the image?"},
    {"role": "assistant", "content": "42"}
  ],
  "images": ["/absolute/path/to/image.png"]
}
```

The launcher sets `--input-key messages --multimodal-keys '{"image":"images"}'`.
Do **not** pass `--apply-chat-template`: the SFT rollout keeps raw messages,
renders the complete GLM template, expands image tokens through the processor,
and then aligns the assistant-only loss mask. Images retain their original
dimensions until GLM's own dynamic resize/patchify. Multiple images and multi-turn
conversations are supported; `step_loss_mask: 0` disables supervision for an
assistant turn. Image embedding tokens are never supervised.

### Launch

Eight-GPU 4-layer smoke:

```bash
python scripts/run_glm5_3_flash_vision_sft.py \
  --num-nodes 1 --num-gpus-per-node 8 \
  --hf-checkpoint /root/models/GLM-5.3-Flash-4layer-BF16 \
  --ref-load /root/models/glm5.3-flash-4layer_torch_dist \
  --prompt-data /root/datasets/glm53_image_sft.jsonl \
  --output-dir /root/runs/glm53_image_sft \
  --num-rollout 2 --save-interval 1
```

Full-model recipe on an externally joined 32-GPU Ray cluster:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_glm5_3_flash_vision_sft.py \
  --model-name GLM-5.3-Flash --num-nodes 4 --num-gpus-per-node 8 \
  --hf-checkpoint /root/models/GLM-5.3-Flash-BF16 \
  --ref-load /root/models/glm5.3-flash_torch_dist \
  --prompt-data /root/datasets/glm53_image_sft.jsonl \
  --output-dir /root/runs/glm53_image_sft_full
```

| Shape | TP | PP | EP | CP |
|---|---|---|---|---|
| 1 x 8, 4-layer | 2 | 2 | 2 | 1 |
| 4 x 8, full | 8 | 4 | 8 | 1 |
| 8 x 8, full | 8 | 4 | 16 | 1 |

All shapes use sequence parallelism. Defaults are 16 conversations per optimizer
step, BF16, Adam at `lr 1e-5`, full recompute, dynamic batching with
`max_tokens_per_gpu=8192`, and CPU-offloaded Adam states. Adjust batch sizes together
using `--rollout-batch-size` and `--global-batch-size`. The launcher reads the local
checkpoint's `text_config.rope_theta` so older checkpoint metadata remains consistent
with startup validation; `--rotary-base` can override it explicitly.

The image path was validated on 8 H20 GPUs with a 4-layer BF16 slice at
TP2/PP2/EP2/SP/CP1: two fresh optimizer steps and one resumed step at
`--vision-trainable projector`, including single-image, multi-image, and masked
multi-turn samples. Loss moved `13.7438 -> 13.7231 -> 13.5854`. Every first-stage
rank produced finite, nonzero gradients for both the language and the trainable
visual parameters, the frozen visual parameters kept no gradient, and all ranks
completed their optimizer updates. The distributed checkpoint held Adam state, RNG
state and 347 `visual.*` entries, and `visual.merger.post_projection_norm.bias`
changed across both the fresh and the resumed update. `--vision-trainable full`
also completed a step with all 563,627,008 visual parameters trainable.

The full 45-layer model was then validated on 32 H20 GPUs at TP8/PP4/EP8/SP/CP1
with `--vision-trainable projector`: one optimizer step, loss 5.0549, grad norm
189.3484 identical on all 32 ranks. The eight first-pipeline-stage ranks each
carried the tower with the same 159,396,864 trainable and 404,230,144 frozen
parameters as the 4-layer slice, all with finite nonzero visual gradients, and
all 32 ranks completed their optimizer update. The checkpoint holds 347
`visual.*` entries alongside the optimizer and RNG state.

The full model also trained an epoch per scope on 2101 GEO3K image-text samples
(65 steps at a global batch of 32, same seed and data in both runs, 32 H20 GPUs).
Loss fell from 3.768 to 0.264 with `--vision-trainable none` and from 3.696 to
0.285 with `full`, both saving their checkpoint. That confirms each scope trains
without diverging; it does not rank them, because the scopes differ by 563,627,008
trainable parameters and the GEO3K targets are a bare `Answer: \boxed{...}` line,
which is easy to fit and says nothing about whether the geometry reasoning improved.

These are functional
smoke tests, not an image-task accuracy or convergence benchmark.

### Limits And Resume

- Image input only; video and audio inputs are rejected explicitly.
- Context parallelism must be 1, and training uses packed `--qkv-format thd`.
- The visual tower is replicated on first-stage TP ranks. Under sequence
  parallelism its trainable gradients are partial per rank, so they are marked for
  Megatron's TP sum-reduction in `finalize_model_grads`.
- With `--vision-trainable none` the tower is excluded from the optimizer and the
  Megatron checkpoint, and its weights are reloaded from the HF checkpoint on every
  start. This mode is not visual-tower finetuning.
- Any other scope writes `visual.*` into the checkpoint, so an existing run cannot
  change `--vision-trainable` on resume: the saved optimizer state no longer matches
  the parameter set and the load fails rather than training a different model.
- A fresh run loads the language-only conversion, which has no `visual.*` entries;
  the tower leaves the module tree for that load and keeps its HF-loaded weights.
- Reuse the same `--output-dir` and HF checkpoint to resume. Language weights,
  trainable visual weights, optimizer state, and RNG are saved and restored
  normally.
- The full-model topology reuses the text-training recipe. Both the 4-layer slice
  and the full 45-layer model are validated for startup, gradients and
  checkpointing only; neither measures convergence, image-task accuracy or
  throughput.
