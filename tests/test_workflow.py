import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from moe_circuits.cli import discover, evaluate, load_circuit, parser
from moe_circuits.data import digest, read_jsonl, write_jsonl
from moe_circuits.report import summarize
from moe_circuits.runtime import encode


def test_encoder_response_window_and_no_truncation(tiny_tokenizer):
    row = {"id": "a", "prompt": "question one", "response": "sorry answer"}
    inputs, start = encode(tiny_tokenizer, row, "cpu", 32, True)
    assert tiny_tokenizer.decode(inputs["input_ids"][0, start:]) == "sorry answer"
    assert tiny_tokenizer.eos_token_id not in inputs["input_ids"][0, start:].tolist()
    with pytest.raises(ValueError, match="no truncation"):
        encode(tiny_tokenizer, row, "cpu", 1, True)


def test_encoder_rejects_response_control_tokens(tiny_tokenizer):
    with pytest.raises(ValueError, match="plain final text"):
        encode(tiny_tokenizer, {"id": "bad", "prompt": "hello", "response": "answer [EOS]"}, "cpu", 32, True)


def test_end_to_end_real_tiny_gptoss(tmp_path, monkeypatch, tiny_model, tiny_tokenizer, records):
    def loader(*args, **kwargs):
        return tiny_model, tiny_tokenizer, {
            "model_id": "tiny", "revision": "fixed", "template_date": "2026-09-24",
            "template_sha256": digest(tiny_tokenizer.chat_template), "model_config_sha256": "test-config",
            "response_mode": "final_only", "dtype": "float32",
            "local_weights_sha256": {},
        }
    monkeypatch.setattr("moe_circuits.runtime.load_runtime", loader)
    dataset = tmp_path / "dataset.jsonl"
    write_jsonl(dataset, records)
    discovery = tmp_path / "discovery"
    discover(parser().parse_args(["discover", "--dataset", str(dataset), "--behavior", "apology",
                                  "--out", str(discovery), "--top-k", "2", "--min-positive-examples", "1"]))
    artifact = load_circuit(discovery / "circuit.json")
    assert len(artifact["targets"]) == 2
    run = tmp_path / "evaluation"
    args = parser().parse_args(["evaluate", "--dataset", str(dataset), "--circuit", str(discovery / "circuit.json"),
                               "--out", str(run), "--random-controls", "1", "--max-new-tokens", "3"])
    evaluate(args)
    generations = read_jsonl(run / "generations.jsonl")
    assert len(generations) == 6
    assert json.loads((run / "summary.json").read_text())["status"] == "partially_scored"
    assert "variant" not in read_jsonl(run / "review.jsonl")[0]
    labels = read_jsonl(run / "labels.jsonl")
    for row in labels:
        output = next(g for g in generations if g["id"] == row["id"])
        row["behavior_present"] = 0 if output["variant"] == "candidate" else 1
        row["acceptable"] = 1
    write_jsonl(run / "labels.jsonl", labels)
    summary = summarize(run, run / "labels.jsonl")
    candidate = next(m for m in summary["metrics"] if m["variant"] == "candidate" and m["split"] == "validation")
    assert candidate["paired_behavior_present"]["mean_change"] == -1
    assert summary["status"] == "fully_scored"
    # Saved discovery provenance catches leakage even in a different evaluation file.
    leak = [{"id": "new", "split": "validation", "prompt": records[0]["prompt"]}]
    write_jsonl(dataset, leak)
    args.out = tmp_path / "leak-run"
    with pytest.raises(ValueError, match="used for discovery"):
        evaluate(args)
    # Corrupted grading cannot silently score another output.
    labels[0]["output_sha256"] = "wrong"
    write_jsonl(run / "labels.jsonl", labels)
    with pytest.raises(ValueError, match="hash"):
        summarize(run, run / "labels.jsonl")


def test_circuit_tampering_rejected(tmp_path):
    path = Path(tmp_path) / "circuit.json"
    path.write_text('{"targets": [], "circuit_sha256": "not-the-hash"}')
    with pytest.raises(ValueError, match="integrity"):
        load_circuit(path)


def test_real_cli_loads_saved_checkpoint_without_network(tmp_path, tiny_model, tiny_tokenizer, records):
    checkpoint = tmp_path / "checkpoint"
    tiny_model.save_pretrained(checkpoint)
    tiny_tokenizer.save_pretrained(checkpoint)
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, records)
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    commands = [
        ["discover", "--dataset", str(dataset), "--behavior", "unnecessary apology",
         "--out", str(tmp_path / "discovery"), "--model", str(checkpoint), "--device", "cpu",
         "--top-k", "2", "--min-positive-examples", "1"],
        ["evaluate", "--dataset", str(dataset), "--circuit", str(tmp_path / "discovery" / "circuit.json"),
         "--out", str(tmp_path / "eval"), "--device", "cpu", "--random-controls", "1", "--max-new-tokens", "3"],
    ]
    for command in commands:
        result = subprocess.run([sys.executable, "-m", "moe_circuits.cli", *command],
                                capture_output=True, text=True, env=environment, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    circuit = load_circuit(tmp_path / "discovery" / "circuit.json")
    assert circuit["runtime"]["local_weights_sha256"]
    assert circuit["runtime"]["dtype"] == "bfloat16"
    assert len(read_jsonl(tmp_path / "eval" / "generations.jsonl")) == 6
