# Validation performed

## Original full-model GPU experiments

The original workflow was already exercised on GPT-OSS-20B using an A100 80GB GPU with bf16 dequantized experts. The June 27 experiment record reports the following results on 60 held-out adult prompts:

| Condition | Measured refusals |
|---|---:|
| Baseline | 43/60 |
| Earlier 200-neuron target set | 10/60 |
| Full-bank adult-refusal versus benign-nonrefusal targets | 0/60 |
| Five size-matched target sets | 0/60, 1/60, 1/60, 1/60, 0/60 |

The five size-matched sets each used 23 positive and 13 negative traces to select 200 neurons. Their pairwise overlap was 180–187 neurons. These counts are transcribed from the contemporaneous `GPT_OSS_MOE_CNA_SIDE_THREAD_TEST_PLAN.md` experiment log, not newly reproduced results. The raw run files and original target artifacts are not bundled in this repository.

This is prior empirical evidence for the original discovery-and-ablation workflow on that behavior and checkpoint. It does not establish arbitrary-behavior generalization or the efficacy of every subsequent implementation change.

## Local checks on this refactor

Environment: macOS arm64, Python 3.12.9, PyTorch 2.8.0, Transformers 4.57.6, Accelerate 1.10.1, NumPy 2.2.6. Full dependency snapshot: `requirements-tested.txt`.

- **24 tests passed** using pytest, with no full model download.
- **Ruff checks passed** for package code and tests.
- **Bundled dataset validation passed**, reporting the expected small-sample warning.
- **Actual `openai/gpt-oss-20b` tokenizer checked:** a real prompt plus supplied final response encoded with an assistant-final prefix. The answer round-tripped exactly; only its 10 response tokens were inside the response window. Only tokenizer assets were downloaded for this check.

## What the tests establish

- No-ablation adapter logits match a tiny randomly initialized Transformers GPT-OSS model at float32 tolerances of `atol=1e-6`, `rtol=1e-5`.
- Native and adapter cached greedy generation agree on that test model.
- A selected-neuron ablation matches an independent reference that zeros the neuron's outgoing projection row; tokens not routed to that expert are unaffected in the direct-module test.
- Original forward methods are restored after normal use and exceptions.
- The statistics exclude prompt tokens, apply route weights, and weight examples equally despite different response lengths.
- Random controls preserve target counts within layer/expert and exclude the candidate neurons.
- Dataset validation rejects prompt/group leakage, malformed labels, duplicate examples, and mislabeled evaluation inputs.
- Discovery-to-evaluation provenance catches reuse of a discovery prompt even in a new evaluation file.
- Human scoring rejects mismatched output hashes and reports missing labels instead of treating them as successes.
- A full subprocess CLI test saves a tiny real GPT-OSS checkpoint and tokenizer, reloads it through the production runtime in **bf16**, discovers candidates, generates baseline/candidate/random outputs, and writes an unscored report. This test runs with Hugging Face networking disabled.

## Not established for this modified package

- A full GPT-OSS-20B CUDA regression run of this refactor, including checkpoint loading with its pinned dependency stack.
- Replication of the original behavior-suppression results after this package's changes to response mode, activation capture, and averaging; capability preservation also requires evaluation.
- Support for other Transformers versions, other model architectures, fused expert kernels, or GPT-OSS analysis-channel behavior.
- A minimal, sufficient, unique, or complete causal circuit. Discovered neurons remain candidates pending held-out intervention evidence.

The production commands run their own checkpoint-level no-ablation equivalence check before collecting activations or generating evaluation outputs. That is a numerical smoke check on one example, not a substitute for complete evaluation.
