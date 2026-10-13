# GLM-5.3-Flash Image SFT: Experiment Record

What has and has not been measured for `--vision-trainable`, with the commands to
reproduce each number. The feature itself is documented in
[glm5-3-flash.md](glm5-3-flash.md#6-bf16-image-sft); this page is the evidence
behind it.

**Current status, 2026-10-13 10:52 Beijing:** full-scale training and validated
multimodal export work. The corrected ChartQA baseline is 65.60%; self-distilled
SFT with a frozen tower scores 67.40%, while the trainable-tower arm scores
65.04%. This first paired run does not meet the goal: training the tower reduces
accuracy by 2.36 percentage points under the unchanged scoring contract. A
separate improved-data experiment is running: its frozen arm has finished 96
steps and is exporting; its full arm has finished 59 of 96 steps. Neither has a
completed second-round evaluation. Sections 1-4 preserve the earlier GEO3K
evidence and withdrawn evaluation.

## 1. Full-model training run

45-layer GLM-5.3-Flash on 32xH20 (TP8/PP4/EP8/SP, CP1), GEO3K's 2101 image-text
train rows, one epoch = 65 steps at a global batch of 32. Both arms used the same
seed, data order and hyper-parameters, and differ only in `--vision-trainable`.

![Training loss, none vs full](assets/loss_curves.svg)

| | `none` | `full` |
|---|---|---|
| trainable visual params | 0 | 563,627,008 |
| step 1 | 3.768 | 3.696 |
| step 65 | **0.264** | **0.285** |
| minimum | 0.141 | 0.195 |
| checkpoint saved | yes | yes |

Per-step values: [`assets/loss_curves.csv`](assets/loss_curves.csv).

Both arms converge without diverging or producing NaNs, every rank reports the
same gradient norm, and the checkpoints load back. That is the claim this run
supports.

**It does not rank the two arms.** The scopes differ by 563,627,008 trainable
parameters, so equal-step loss is not a like-for-like comparison, and the GEO3K
targets are a bare `Answer: \boxed{...}` line (median 18 characters) that is easy
to fit either way.

### Reproducing

```bash
# 32 GPUs, external Ray. One command per arm; they are independent.
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_glm5_3_flash_vision_sft.py \
  --model-name GLM-5.3-Flash --num-nodes 4 --num-gpus-per-node 8 \
  --hf-checkpoint /root/models/GLM-5.3-Flash-BF16 \
  --ref-load /root/models/glm5.3-flash_torch_dist \
  --prompt-data /root/datasets/geo3k/train.jsonl \
  --output-dir /root/runs/geo3k_none \
  --vision-trainable none \
  --num-rollout 65 --rollout-batch-size 32 --global-batch-size 32 \
  --save-interval 65 --max-tokens-per-gpu 2048 --lr 1e-5
```

Swap `--vision-trainable full` and the output directory for the second arm; the
launcher adds `--glm5-vision-recompute` for that scope on its own.

What makes the two arms comparable is `--rollout-seed`, which seeds the shuffle in
`DataSource` and so fixes the order the samples arrive in. The launcher pins it at
42 rather than inheriting the parser default, so the arms share a data order
without the caller having to arrange it.

The runs behind the numbers were launched with two extra flags that this command
drops, because neither reaches the arithmetic:

- `--seed 42`, which replaces Megatron's default 1234. It seeds weight init and
  dropout; this recipe loads every weight from a checkpoint and sets both dropout
  rates to 0, so nothing consumes it. It is not what orders the data.
- `--seq-length 2048`, which only sizes `max_position_embeddings`. Training is
  packed `thd` with `variable_seq_lengths`, and rope derives its table length from
  the batch at runtime, so the value is a placeholder.

They also used `--spec glm5_next_optimizer_cache_compat`, an out-of-tree wrapper
that released the CUDA cache before the optimizer was built. That release now
lives in `setup_model_and_optimizer`, so the repository covers it and the command
above needs no custom spec.

Only the visual shard of the HF checkpoint is read (all 347 `model.visual.*`
tensors live in `model-00062-of-00062.safetensors`), so `--hf-checkpoint` does not
need the full 306 GB export. The language weights come from `--ref-load`.

### The data these runs used

`train.jsonl` is GEO3K in the format
[the feature page specifies](glm5-3-flash.md#dataset): one row per sample, with a
`messages` conversation and an `images` list of readable paths.

```json
{
  "messages": [
    {"role": "user", "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n<image>Find x."},
    {"role": "assistant", "content": "Answer: \\boxed{48}"}
  ],
  "images": ["/root/datasets/geo3k/images_train/54b363a493f4b5a9.png"]
}
```

Built from [`chenhegu/geo3k_imgurl`](https://huggingface.co/datasets/chenhegu/geo3k_imgurl)
(2101 train / 601 test rows, one image each). Two things about that dataset are
easy to get wrong: the decoded images are in `preprocessed_images`, not `images`
(which holds base64 data URIs), and `problem` already carries its own `<image>`
placeholder and `\boxed{}` instruction, so neither needs to be added. The
assistant target is `f"Answer: \\boxed{{{row['answer']}}}"`.

Any dataset in this shape reproduces the run; the numbers above are specific to
this one.

### How closely this reproduces

The commands and the repository are enough to re-run both arms and get the same
shape of result: a fall from ~3.7 to ~0.2-0.3 over the epoch, no divergence, a
loadable checkpoint. The comparison between arms holds because they share a seed.

Exact per-step values will not match. BF16 reductions are not associative, so a
different GPU count, micro-batch split or container build reorders them; the runs
above were on 32xH20 under one pinned image. Treat the CSV as the trace of one
run, not a target to diff against.

## 2. The zero-shot baseline is withdrawn

This page used to report a GEO3K zero-shot accuracy for the 45-layer model. **Those
numbers are withdrawn: the evaluation never delivered the image to the model**, so they
describe its text-only behaviour on geometry questions and say nothing about the visual
tower.

What gave it away, in the stored predictions from that run:

- 155 of the 601 responses contain a `</image>` tag. The harness put the training
  convention `<image>` into the prompt, which is a placeholder only the Miles processor
  understands; reaching the model as literal text, it answered by closing the tag.
- The responses narrate guessing rather than reading. One opens "There is image with
  likely some lengths", another emits `[asy]` source inventing a diagram.

Four separate defects in that harness, each confirmed against the checkpoint:

1. The model's image token is `<|image|>` (id 154854). `<image>` is the processor's
   input convention, not the model's.
2. No chat template was applied, so the model continued the document instead of
   answering it -- echoing the question back, starting a "Question 2.".
3. The processor expands one `<|image|>` into one token per merged patch -- 682 of them
   for a 1176x746 chart. Those are the slots the image embeddings fill. A prompt with a
   single placeholder leaves the model nowhere to put the picture, and it says so: "I
   don't see any image attached to this conversation."
4. The chat template ends in `<think>`, so a short token budget is spent reasoning and
   never reaches an answer.

Note that defects 1 and 3 pull in opposite directions from SGLang's side: it ships its
own GLM multimodal processor and expands the placeholder itself, so pre-expanding with
the Miles processor *and* passing the images makes its loader reject the request.
`miles/rollout/sglang_rollout.py` is the working reference -- processor-expanded
`input_ids` plus base64 data URIs, posted to a server endpoint rather than driven
through the Engine API.

**No corrected GEO3K number is given.** The old predictions are not reused.
Section 5 records a separate, corrected ChartQA evaluation. The training results
in section 1 are unaffected: they come from the training path, which uses the
processor directly and is covered by the e2e test.

## 3. What is not measured, and why GEO3K cannot measure it

At the time of these GEO3K runs, there was **no evidence about whether training
the visual tower improves image-task quality**. Their bare-answer training
targets are not the reasoning data used for the later ChartQA experiment.

GEO3K's prompt asks the model to "Solve the following math problem step by step",
but its SFT target contains no steps - just `Answer: \boxed{48}`, a median of 18
characters. The prompt and the target disagree about what the answer is, and the
target wins: training on it teaches the model to skip the reasoning the prompt asks
for. That reading comes from the dataset alone, so unlike the withdrawn baseline above
it does not depend on anything the evaluation harness measured.

That matters for this comparison specifically: if the task collapses to emitting a
short string, the image contributes little and the arms converge regardless of
whether the tower trains. Any "no difference" result would be a property of the
data, not a finding about the visual tower.

Measuring the effect needs targets that contain the derivation - distilled from
correct model-generated solutions, or a geometry set that ships rationales.

## 4. Smaller-scale checks

The 4-layer slice (`--model-name GLM-5.3-Flash-4layer`) is a startup fixture: it
exercises parallelism, gradients and checkpointing cheaply, and all three scopes
run 195 steps on it.

**Do not use it for anything that depends on model behaviour.** It cannot generate
text - the untrained slice emits `Unload unload unload...` and a trained one emits
`}}}}`, so task accuracy on it is identically zero and cannot separate any two
configurations. That one survives the withdrawal above: a model emitting `}}}}` is
broken whether or not an image reached it, so the conclusion rests on the output text
rather than on the evaluation path.

## 5. Corrected ChartQA Self-Distillation Experiment

This is a separate 45-layer experiment, not a reinterpretation of the withdrawn
GEO3K evaluation. Training inputs are ChartQA **training** images and questions;
assistant targets are the teacher's generated reasoning and answer, not the
dataset's bare label. The first-round teacher is the original GLM-5.3-Flash.
Only predictions with complete reasoning and a relaxed-correct final answer were
retained: 1480 of 2101 training questions. An audit verifies image/question
alignment and verbatim generated targets. These 2101 source questions were all
human-authored; that coverage limitation motivates the independent second round.

Both arms train the language model for 184 optimizer steps at global batch 32,
learning rate `1e-5`, and training/data-shuffle seed 42. `none` freezes the entire
visual tower; `full` trains it and enables visual recomputation. Each arm uses
its own 32-H20 TP8/PP4/EP8 cluster. `full` omits optimizer-state **saving** to
reduce checkpoint I/O; optimizer computation is unchanged.

The corrected evaluation applies the native chat template and multimodal
processor, provides the actual images with their expanded token positions, and
uses deterministic decoding with a 1024-new-token limit. Four independent TP8
replicas each evaluate 625 questions. Predictions are merged in original order
and checked against all 2500 test questions before scoring. Test answers are used
only for evaluation, never teacher generation, filtering, or training.

| Model | Correct / 2500 | Relaxed Accuracy | Human / 1250 | Machine / 1250 |
|---|---:|---:|---:|---:|
| Original model | 1640 | 65.60% | 77.44% | 53.76% |
| Self-distilled `none`, seed 42 | 1685 | 67.40% | 80.08% | 54.72% |
| Self-distilled `full`, seed 42 | 1626 | 65.04% | 79.28% | 50.80% |

The frozen-tower arm improves by **1.80 percentage points** over the original
model. The trainable-tower arm is **2.36 points below `none`** and **0.56 points
below the original model**. In the paired predictions, both arms answer 1545
questions correctly; `full` fixes 81 that `none` misses, but misses 140 that
`none` answers correctly. Both miss 734. The largest subgroup regression is
machine-authored questions (50.80% versus 54.72%); human accuracy also decreases
(79.28% versus 80.08%). This is one seed and one data/configuration choice, not
proof that visual training cannot help other configurations.

The full export has all 45 language layers,
74001 language tensors, and all 347 visual tensors; 260 visual tensors changed,
with maximum absolute update `0.0009765625`, and the exported visual weights are
finite. These checks establish an actual cumulative update, not improved accuracy.

### First-Round Diagnosis

A paired audit of all 2500 ordered predictions separates observed errors from
unverified explanations. The net loss is 59 correct answers: 49 in the machine
half and 10 in the human half. Numeric-answer questions account for a net loss
of 49, and text-answer questions for 10; these are overlapping breakdowns, not
additive causes. Both arms close their reasoning block in 2494 outputs and fail
to close it in six, so extra output truncation does not explain the gap.

Answer presentation is an important confound. The unchanged scorer evaluates
the first nonempty line after `</think>` as a whole. It strips bold markers but
does not accept appended units or explanations. For example, the full arm
answers `**524** (thousand persons)` where the label is `524`, and sometimes
emits `Yes, ...` where the label is `Yes`.

An additional diagnostic extracts only an anchored leading numeric or yes/no
token. It does not search the output for the gold value and does not change the
official score:

| Paired Change | Count | With a Matching Leading Token in the Rejected Answer |
|---|---:|---:|
| `none` correct, `full` rejected | 140 | 95 |
| `full` correct, `none` rejected | 81 | 26 |

Matching uses the existing relaxed numeric tolerance. A matching prefix does
not establish correct units, a unique answer, full-response semantic correctness,
or image-grounded reasoning. This table therefore cannot turn the negative
result into a successful training result. It does show that the 2.36-point gap
cannot be equated directly with a 2.36-point loss in visual recognition.
Genuine value errors also occur: one machine question with label `315` receives
`315` from `none` and `2616` from `full`.

The training audit establishes three limitations that need separate controlled
experiments before claiming a cause:

- The 1480 retained traces are all human-authored questions, covering 923 unique
  chart contents. The 184 steps at batch 32 expose about 3.98 dataset passes.
- Visual and language parameters both use `1e-5`. The visual learning rate was
  not independently tuned, and there is no training-derived held-out selection
  result establishing that this update strength or duration generalizes.
- The teacher's complete reasoning and final answer were filtered, but
  intermediate reasoning was not checked step by step against the image.
  Final-answer correctness alone does not verify those steps.

Data coverage, overly strong adaptation, and reasoning quality are hypotheses,
not proven causes. A follow-up validation set must come from training-source
charts absent from both preceding training inputs and teacher training exposure.
Simply holding out some of the already used 3017 rows would not establish
independence. No test questions or answers are used for data generation,
filtering, or configuration selection.

### Measured Wall Time

Times below come from supervisor records, rank-zero training timers, checkpoint
timestamps, and completed evaluation timing files, not estimated GPU FLOPs.

| Component | `none` Seconds | `full` Seconds |
|---|---:|---:|
| Training wall time | 13179.4 | 8539.2 |
| Initialization and weight loading | 3336.3 | 2578.2 |
| First train step, including cold kernel work | 907.7 | 908.8 |
| Remaining 183 rank-zero train timers | 2948.6 | 3353.6 |
| Checkpoint I/O | 4806.5 | 648.0 |
| Between-step and teardown residual | 1180.3 | 1050.6 |
| Export prepare / convert / finalize | 22.5 / 1807.0 / 293.3 | 18.3 / 2261.1 / 249.3 |
| Successful four-replica evaluation wall time | 2017.6 | 2887.8 |

`none`'s completed successful-stage sum is 17319.8 seconds, about 4 hours
49 minutes; `full`'s is 13955.7 seconds, about 3 hours 53 minutes. These exclude
failed attempts and idle gaps. The runs overlap on two
clusters, so adding their wall times does not measure total project elapsed
time. The lower full training wall time is mostly a checkpoint-saving
difference; its steady rank-zero train timer is **18.33 s** versus **16.11 s**
for `none`, not evidence of faster visual computation.

An initial shared-filesystem mmap evaluation stalled in page waits and was
preserved as a failed attempt. The successful path stages the exported model to
each node's local cache with source/readback SHA256 checks and atomic writes.
For `none`, per-worker staging took 1274-1542 s, preprocessing 52-60 s, model
loading 108-110 s, and inference 217-528 s. Reusing the verified teacher cache
took about 18 s with zero model bytes copied. The full evaluation's first cache
staging took 2366-2403 s per node, preprocessing 63-73 s, model loading 106-110 s,
and inference 275-304 s. Its pipeline supervisor finished successfully at
09:32:43 Beijing on October 13.

### Independent Second Round

The original 1480-row dataset remains unchanged. A separate training-only
candidate set contains 1500 novel machine questions, 1000 novel human questions,
and 621 previously rejected questions, deduplicated by chart content and
question. The first-round `none` checkpoint generated a new sampled reasoning
trace for each candidate (`temperature=0.6`, `top_p=0.9`, 1536 new tokens, stable
seed `42 + global_row_index`).

The stricter filter accepts exact finite numeric equality or normalized text
equality, and one complete reasoning block with at least 32 characters. It keeps
1449 original exact-answer traces plus 1568 new traces, for **3017** conversations:
2298 human and 719 machine. New retained rows include 733 novel human questions,
719 novel machine questions, and 116 recovered prior failures. Final-answer
correctness is not proof that every intermediate reasoning step is grounded.

Both second-round arms start from the same first-round `none` language checkpoint
at iteration 183 and the original visual tower. They use the same 3017-row
dataset hash, seed 42, and 96 optimizer steps, saving only the final model in
both arms. `none` started on bj47 at 09:11 Beijing; `full` started on bj48 at
09:32:56, after the first full pipeline finished successfully. These are follow-up experiments with
different data and initialization, not directly interchangeable with the
first-round results.

At 10:52 Beijing, `none` has completed iteration 95 and is converting its final
checkpoint; `full` has completed iteration 58. Both have finite logged losses
and gradient norms. This is training progress, not evidence of improved eval.
Each node now runs a bounded two-worker CPU prefetch task which watches complete
converter shards and verifies source and local-copy SHA256. The evaluation cache
may adopt these exact bytes by hardlink, with source-identity checks and normal
copy fallback. This overlaps export with local staging without changing model
weights, precision, predictions, or scoring. Real-shard structural validation
passed in the owned runtime; end-to-end staging time saved is not yet measured.

Operational scripts and evidence are under
`/root/sync/zwfy/miles-glm53-vision-sft/`: `scripts/compare_chartqa_results.py`,
`scripts/summarize_chartqa_timings.py`, `logs/chartqa_none_seed42.evaluation.json`,
`logs/distilled_training_audit.json`, and
`data/chartqa/distill_v2/sft_exact.manifest.json`. The second-round comparison
rejects mismatched data hashes, initial checkpoint steps, step counts, or partial
evaluation results. The paired diagnostic is
`logs/chartqa_seed42_error_analysis.json`, generated by
`scripts/analyze_chartqa_errors.py`; its prefix classification is diagnostic
only and is covered by CPU tests.
