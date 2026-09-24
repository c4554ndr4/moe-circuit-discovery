"""Small, explicit CLI. Dataset validation and scoring require no model installation."""

import argparse
import json
from pathlib import Path

from .data import digest, new_run, prompt_key, read_jsonl, validate, write_json, write_jsonl


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def discover(args):
    import numpy as np
    import torch
    from .adapter import ExpertAdapter, check_equivalence
    from .discovery import ActivationStats
    from .runtime import encode, input_device, load_runtime

    rows = read_jsonl(args.dataset)
    dataset_info = validate(rows)
    for warning in dataset_info["warnings"]:
        print(f"Note: {warning}", flush=True)
    selected = [row for row in rows if row["split"] == "discovery"]
    out = new_run(args.out)
    model, tokenizer, runtime = load_runtime(args.model, args.revision, args.device)
    adapter = ExpertAdapter(model)
    device = input_device(model)
    # Fail on bad/overlong examples before any activation collection.
    encoded = [(row, *encode(tokenizer, row, device, args.max_tokens, True)) for row in selected]
    equivalence = check_equivalence(model, adapter, encoded[0][1])
    stats = ActivationStats(adapter.dims)
    manifest = []
    with torch.inference_mode(), adapter:
        adapter.observe = stats.observe
        for index, (row, inputs, start) in enumerate(encoded, 1):
            stop = inputs["input_ids"].shape[1]
            stats.begin(start, stop)
            model(**inputs, use_cache=False)
            stats.finish(row["label"])
            manifest.append({"id": row["id"], "label": row["label"], "prompt_sha256": prompt_key(row),
                             "response_sha256": digest(row["response"]), "response_tokens": stop - start,
                             "input_ids_sha256": digest(inputs["input_ids"][0].tolist())})
            print(f"Collected {index}/{len(encoded)}: {row['id']}", flush=True)
    targets = stats.rank(args.top_k, args.min_positive_examples, args.ranking)
    config = {"behavior": args.behavior, "ranking": args.ranking, "top_k_requested": args.top_k,
              "min_positive_examples": args.min_positive_examples, "max_tokens": args.max_tokens,
              "normalization": "mean_per_response_then_mean_per_label",
              "activation": "signed_route_weighted_post_gate_pre_down_projection",
              "token_scope": "supplied_final_response_tokens"}
    circuit = {"schema_version": 1, "kind": "candidate_expert_neurons", "runtime": runtime,
               "dims": list(adapter.dims), "dataset": dataset_info, "config": config,
               "equivalence": equivalence, "targets": targets,
               "discovery_prompt_hashes": sorted({prompt_key(row) for row in selected}),
               "discovery_groups": sorted({row['group'] for row in selected if 'group' in row})}
    circuit["circuit_sha256"] = digest(circuit)
    write_json(out / "circuit.json", circuit)
    write_jsonl(out / "discovery_manifest.jsonl", manifest)
    np.savez_compressed(out / "activation_stats.npz", sums=stats.sums,
                        positive_example_hits=stats.positive_example_hits, counts=stats.counts)
    (out / "README.md").write_text(
        f"# Candidate circuit\n\nBehavior: {args.behavior}\n\n"
        f"Selected {len(targets)} / requested {args.top_k} neurons using {stats.counts[1]} positive "
        f"and {stats.counts[0]} negative responses. These are association-ranked candidates, "
        "not a validated causal circuit. Run evaluate on held-out prompts and score the outputs.\n"
    )
    print(f"Wrote {out / 'circuit.json'}; ablation validation is still required.")


def load_circuit(path):
    circuit = json.loads(Path(path).read_text())
    saved_hash = circuit.pop("circuit_sha256", None)
    if saved_hash != digest(circuit):
        raise ValueError("Circuit integrity check failed; use an unedited discovery artifact")
    circuit["circuit_sha256"] = saved_hash
    if circuit.get("schema_version") != 1 or circuit.get("kind") != "candidate_expert_neurons":
        raise ValueError("Unsupported circuit format")
    if not circuit.get("targets"):
        raise ValueError("Circuit has no neuron targets")
    return circuit


def evaluate(args):
    import torch
    from .adapter import ExpertAdapter, check_equivalence
    from .discovery import matched_random
    from .runtime import encode, generate, input_device, load_runtime

    rows = read_jsonl(args.dataset)
    validate(rows, require_discovery=False)
    circuit = load_circuit(args.circuit)
    selected = [row for row in rows if row["split"] in {"validation", "control"}]
    if not any(row["split"] == "validation" for row in selected):
        raise ValueError("Evaluation requires held-out validation prompts")
    for row in selected:
        if prompt_key(row) in circuit["discovery_prompt_hashes"]:
            raise ValueError(f"{row['id']}: evaluation prompt was used for discovery")
        if row.get("group") in circuit["discovery_groups"]:
            raise ValueError(f"{row['id']}: evaluation group was used for discovery")
    out = new_run(args.out)
    source = circuit["runtime"]
    model, tokenizer, runtime = load_runtime(source["model_id"], source["revision"], args.device,
                                             source["template_date"])
    for key in ("model_id", "revision", "template_sha256", "model_config_sha256", "local_weights_sha256",
                "response_mode", "dtype"):
        if runtime[key] != source[key]:
            raise ValueError(f"Model/tokenizer provenance mismatch: {key}")
    adapter = ExpertAdapter(model)
    if list(adapter.dims) != circuit["dims"]:
        raise ValueError("Circuit dimensions do not match the loaded model")
    device = input_device(model)
    encoded = [(row, encode(tokenizer, row, device, args.max_tokens)[0]) for row in selected]
    if any(inputs["input_ids"].shape[1] + args.max_new_tokens > model.config.max_position_embeddings
           for _, inputs in encoded):
        raise ValueError("Prompt plus generation budget exceeds the model context length")
    equivalence = check_equivalence(model, adapter, encoded[0][1])
    variants = {"baseline": [], "candidate": circuit["targets"]}
    for index in range(args.random_controls):
        variants[f"random_{index:02d}"] = matched_random(circuit["targets"], adapter.dims, args.seed + index)
    write_json(out / "run.json", {"runtime": runtime, "equivalence": equivalence,
                                 "circuit_sha256": circuit["circuit_sha256"],
                                 "behavior": circuit["config"]["behavior"],
                                 "dataset_sha256": digest(rows), "seed": args.seed,
                                 "max_new_tokens": args.max_new_tokens, "max_tokens": args.max_tokens,
                                 "variants": variants, "ablation_scope": "all_prompt_and_generated_tokens",
                                 "decoding": "greedy_final_only"})
    results = []
    # Stream completed generations, but never silently resume or overwrite a run.
    with torch.inference_mode(), adapter, (out / "generations.jsonl").open("w") as handle:
        for variant, targets in variants.items():
            adapter.set_targets(targets)
            for row, inputs in encoded:
                adapter.events = 0
                result = generate(model, tokenizer, inputs, args.max_new_tokens)
                result.update({"id": digest([row["id"], variant]), "example_id": row["id"],
                               "split": row["split"], "variant": variant, "prompt": row["prompt"],
                               "system": row.get("system", ""), "ablation_events": adapter.events})
                result["output_sha256"] = digest(result["response"])
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"Generated {variant}: {row['id']}", flush=True)
    labels = [{"id": row["id"], "output_sha256": row["output_sha256"],
               "behavior_present": None, "acceptable": None} for row in results]
    write_jsonl(out / "labels.jsonl", labels)
    # A shuffled review sheet hides variant names to reduce grading bias.
    import random
    review = [{key: row[key] for key in ("id", "prompt", "system", "response", "hit_token_limit")}
              for row in results]
    random.Random(args.seed).shuffle(review)
    write_jsonl(out / "review.jsonl", review)
    from .report import summarize
    summarize(out, out / "labels.jsonl")
    print(f"Wrote {out}. Grade review.jsonl, fill labels.jsonl, then run summarize.")


def parser():
    root = argparse.ArgumentParser(description="Discover candidate GPT-OSS neuron circuits from your dataset")
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Validate JSONL and check split leakage; no GPU required")
    check.add_argument("dataset", type=Path)
    collect = commands.add_parser("discover", help="Collect labeled activations and rank candidate neurons")
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--behavior", required=True, help="Operational definition of label 1")
    collect.add_argument("--out", type=Path, required=True)
    collect.add_argument("--model", default="openai/gpt-oss-20b")
    collect.add_argument("--revision", default="main", help="Prefer a checkpoint commit SHA")
    collect.add_argument("--device", default="auto")
    collect.add_argument("--top-k", type=positive_int, default=200)
    collect.add_argument("--min-positive-examples", type=positive_int, default=3)
    collect.add_argument("--ranking", choices=["absolute", "positive"], default="absolute")
    collect.add_argument("--max-tokens", type=positive_int, default=2048)
    run = commands.add_parser("evaluate", help="Generate baseline, candidate, and matched random ablations")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--circuit", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--random-controls", type=positive_int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-tokens", type=positive_int, default=2048)
    run.add_argument("--max-new-tokens", type=positive_int, default=512)
    summary = commands.add_parser("summarize", help="Compute paired effects from human/custom scorer labels")
    summary.add_argument("--run", type=Path, required=True)
    summary.add_argument("--labels", type=Path, required=True)
    return root


def main():
    root = parser()
    args = root.parse_args()
    try:
        if args.command == "validate":
            print(json.dumps(validate(read_jsonl(args.dataset)), indent=2))
        elif args.command == "discover":
            discover(args)
        elif args.command == "evaluate":
            evaluate(args)
        else:
            from .report import summarize
            print(json.dumps(summarize(args.run, args.labels), indent=2))
    except (ValueError, RuntimeError, OSError, AssertionError) as error:
        root.exit(1, f"Error: {error}\n")
    except ModuleNotFoundError as error:
        root.exit(1, f"Missing dependency: {error.name}. Install with: pip install -e '.[model]'\n")


if __name__ == "__main__":
    main()
