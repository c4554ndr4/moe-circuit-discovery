# From the original GPT-OSS experiments

Source: Cassandra's `moe_cna_experiments` work from June 2026, especially `gptoss_intervention_eval.py`, `gptoss_multi_cna_eval.py`, and `interim_gptoss_neuron_circuits.py`.

This is a standalone cleaned successor; the original experiment folder is unchanged. It does not depend on those scripts, their absolute paths, or their external refusal datasets.

| Original | Cleaned workflow |
|---|---|
| Hard-coded adult/refusal/benign buckets | User-provided labeled JSONL |
| Regex refusal detector | Explicit behavior and quality labels on fresh generations |
| Generate, then trace refusal-labeled outputs | Teacher-force user-supplied final responses |
| Sparse top-k activation trace bank | Stream full routed-neuron means; save aggregate statistics |
| Token-weighted class averages | Equal weight per response, then per label |
| Positive signed-delta targets | Absolute signed-delta ranking by default; positive-only option |
| Unpinned expert implementations | Pinned routing layout with checkpoint-level no-ablation check |
| Permanent forward monkey-patches | Context-managed adapter restored on exit |
| Baseline and target comparisons | Baseline, target, and per-expert count-matched random controls |
| Optional text saving | Always save generated text and completion status |
| Row-count cache reuse | New output directories; no implicit resume |

These measurement changes mean the new tool is **not an exact reproduction** of the GPU experiment recorded in the June log (43/60 → 0–1/60 measured refusals). The original workflow was run successfully; the outstanding check is regression validation of this modified package. Original target artifacts and raw results were not recovered or bundled. See [VALIDATION.md](VALIDATION.md) for prior results and current test coverage.

The adapter follows GPT-OSS's gated expert computation used in the original experiments, with a distinct routing-weight layout for Transformers 4.57.6. See the [upstream implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gpt_oss/modeling_gpt_oss.py). The full checkpoint is loaded using [MXFP4 dequantization](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/mxfp4).

No reuse license has been assigned to Cassandra's code in this release. Public availability alone does not grant a general redistribution or modification license. Upstream model and dependency licenses continue to apply.
