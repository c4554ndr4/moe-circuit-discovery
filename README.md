# MoE circuit discovery

Bring labeled examples of a behavior. Discover candidate expert neurons in GPT-OSS, ablate them, and measure the change on held-out prompts.

This is a research toolkit extracted and generalized from the June 2026 GPT-OSS CNA experiments. It does **not** contain a pretrained circuit, generate your dataset, or promise that a behavior has a sparse removable circuit. Only the GPT-OSS adapter is implemented. Dataset validation and evaluation reporting run without a GPU.

**Status:** 24 local tests pass, including an end-to-end tiny GPT-OSS checkpoint test. Full GPT-OSS-20B GPU validation is still outstanding. The supported response mode is final-only; reasoning/analysis-channel behavior is outside this version's scope.

The workflow is: **your labeled responses → activation contrasts → candidate neurons → held-out ablation comparisons → behavior and quality scores**. Discovery finds associations; the intervention evaluation is what tests whether those neurons causally affect the behavior.

Download the source ZIP from [Releases](https://github.com/c4554ndr4/moe-circuit-discovery/releases), or clone the repository below.

## What you provide

1. An operational definition of the behavior. `label: 1` means present; `label: 0` means absent. These do not mean good/bad.
2. Discovery examples: a prompt, a complete assistant response, and its label.
3. Separate validation prompts where you want to measure the intervention, and control prompts for unrelated capabilities.

One JSON object per line:

```json
{"id":"train-1","split":"discovery","prompt":"How many days are in a leap year?","response":"Sorry, there are 366 days in a leap year.","label":1}
{"id":"train-2","split":"discovery","prompt":"How many days are in a leap year?","response":"There are 366 days in a leap year.","label":0}
{"id":"heldout-1","split":"validation","prompt":"What is the capital of Portugal?"}
{"id":"control-1","split":"control","prompt":"Calculate 17 multiplied by 23."}
```

This illustrates unnecessary apologies. The bundled [example dataset](examples/apology.jsonl) is handcrafted **format/demo data**, not GPT-OSS outputs or evidence that this circuit exists. Use actual observed responses where possible. Contrasting authored responses are accepted, but they measure model representations of supplied text; behavioral validation is essential.

Read [DATASETS.md](DATASETS.md) for labeling, matching, split rules, sizes, and limitations.

## Install

Use Python 3.11–3.13. Python 3.12 is the tested development version.

```bash
git clone https://github.com/c4554ndr4/moe-circuit-discovery.git
cd moe-circuit-discovery
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
moe-circuits validate examples/apology.jsonl
```

For model commands:

```bash
pip install -e '.[model]'
```

Core model dependencies are deliberately pinned: PyTorch 2.8.0, Transformers 4.57.6, Accelerate 1.10.1, NumPy 2.2.6. The adapter depends on that Transformers routing layout. To change versions, first port the adapter and run its equivalence tests. See `requirements-tested.txt` for the full locally tested dependency snapshot; platform-specific GPU wheel resolution may differ.

**Hardware:** real GPT-OSS-20B commands load dequantized bf16 expert weights, not the small MXFP4 inference footprint. Budget roughly 40+ GB for weights plus activations and loading overhead; an 80 GB CUDA GPU is a practical starting configuration, not a tested minimum. No model weights are bundled. Model commands download the selected Hugging Face checkpoint. Use `--revision` with a commit SHA for reproducibility. CPU execution is for tiny-model tests, not practical 20B discovery. Fused expert kernels, disk offloading, and training are unsupported.

## 1. Validate your data

```bash
moe-circuits validate my_behavior.jsonl
```

Checks unique IDs, nonempty text, binary labels, both discovery classes, exact/whitespace-normalized prompt leakage, duplicate examples, and `group` leakage. It cannot judge label correctness or detect semantic paraphrases automatically.

## 2. Discover candidate neurons

```bash
moe-circuits discover \
  --dataset my_behavior.jsonl \
  --behavior 'The response apologizes although no apology is warranted.' \
  --model openai/gpt-oss-20b \
  --top-k 200 \
  --out runs/apology-discovery
```

Before discovery, the tool compares native-model logits with its no-ablation adapter on a real dataset example. It stops on a mismatch or missing expert execution.

The tool feeds each supplied response to the model, captures expert activations on response-token positions, weights them by routing, and ranks the difference between labels. It measures every neuron of each routed expert; unselected experts contribute zero. Each response receives equal weight regardless of length. No model training occurs.

`--ranking absolute` (default) ranks the magnitude of the signed difference. `--ranking positive` keeps neurons with higher mean signed activation in positive examples. Neither score guarantees that zeroing a neuron will suppress the behavior. `--min-positive-examples 3` requires the expert to be routed during at least three positive examples; this is an exposure filter, not a neuron significance test.

Outputs:

- `circuit.json`: candidate coordinates, scores, dataset and checkpoint provenance, and equivalence results.
- `activation_stats.npz`: aggregate class sums and expert-exposure counts for inspection, without a large per-token trace bank.
- `discovery_manifest.jsonl`: per-example hashes and measured token counts.

## 3. Evaluate on fresh prompts

```bash
moe-circuits evaluate \
  --dataset my_behavior.jsonl \
  --circuit runs/apology-discovery/circuit.json \
  --random-controls 3 \
  --max-new-tokens 512 \
  --out runs/apology-evaluation
```

Only `validation` and `control` rows are evaluated. This produces baseline responses, candidate-circuit ablations, and three random ablations matched for neuron count **within each selected layer/expert**, excluding the candidate neurons. Random controls do not match activation magnitudes and are not a complete significance analysis.

The same no-ablation-checked adapter runs every variant. Zero ablation acts on the gated expert activation before its output projection, from prompt processing through generation. Weights are never edited. Forward methods are restored when the context exits, including on an exception. Runs do not silently resume or overwrite existing files.

## 4. Grade and summarize

`review.jsonl` contains shuffled outputs without variant names. In `labels.jsonl`, fill in:

- `behavior_present`: `1` if your target behavior occurs, `0` if absent, `null` if uncertain.
- `acceptable`: `1` if the answer remains correct/helpful for the prompt, `0` if damaged or unacceptable, `null` if uncertain.

Grade the actual generated response, not the input prompt or the discovery label. Inspect `hit_token_limit`; unfinished text must not count as successful behavior removal. Do not change generation IDs or hashes. A custom scorer can fill this same file; no external judge or API is required by the toolkit.

```bash
moe-circuits summarize \
  --run runs/apology-evaluation \
  --labels runs/apology-evaluation/labels.jsonl
```

`report.md` and `summary.json` show behavior rates, answer acceptability, scoring coverage, truncation counts, and paired changes relative to baseline. Before grading, the report explicitly says `partially_scored`, with null rates. Missing judgments never become automatic successes. If baseline rarely exhibits the behavior, the experiment has little room to establish suppression.

## GPT-OSS response mode

**This first version studies final-only responses.** It obtains the assistant-final prefix from the checkpoint's chat template, then feeds a supplied final answer or generates directly from that prefix. It does not trace or generate an analysis/reasoning channel. This keeps discovery and validation consistent, but conclusions apply to this final-only condition—not necessarily GPT-OSS's normal reasoning-then-final behavior. Responses must be plain final-answer text, not Harmony dumps or tool transcripts.

The template date is frozen for a run and reused for evaluation. There is no silent input truncation. Long examples fail with an actionable error. A/B datasets can be converted to this schema, but answer-letter features can dominate; balance/reverse answer order and validate on open-ended tasks.

## Development and evidence

```bash
pip install -e '.[model,dev]'
pytest
ruff check src tests
```

Tests use tiny randomly initialized **real Transformers GPT-OSS modules**, plus synthetic examples, to check numerical equivalence, surgical intervention, restoration, ranking, split integrity, reporting, and command orchestration without downloading a 20B model. See [VALIDATION.md](VALIDATION.md) for exactly what was run. Passing these checks is not a GPU/full-checkpoint reproduction or a behavioral result.

Other model architectures belong in separate adapters. The data schema, ranking, matched controls, and reporting are reusable; neuron coordinates and discovered circuits are checkpoint-specific.

See [MIGRATION.md](MIGRATION.md) for differences from the original experiments.
