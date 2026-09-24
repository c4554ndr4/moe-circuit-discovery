# Validation performed

## Local checks

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

## Not established

- A full GPT-OSS-20B run on CUDA, or MXFP4 checkpoint dequantization on the deployment GPU.
- Behavior suppression, capability preservation, or replication of the June experiment's numerical results.
- Support for other Transformers versions, other model architectures, fused expert kernels, or GPT-OSS analysis-channel behavior.
- A minimal, sufficient, unique, or complete causal circuit. Discovered neurons remain candidates pending held-out intervention evidence.

The production commands run their own checkpoint-level no-ablation equivalence check before collecting activations or generating evaluation outputs. That is a numerical smoke check on one example, not a substitute for complete evaluation.
