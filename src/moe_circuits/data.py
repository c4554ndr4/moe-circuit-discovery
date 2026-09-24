"""Strict, GPU-independent dataset and artifact handling."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def new_run(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"{path} is not empty. Use a new output directory; existing runs are never reused.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prompt_key(row):
    # Collapse superficial whitespace to catch accidental cross-split duplication.
    return digest(" ".join(row["prompt"].split()))


def validate(rows, require_discovery=True):
    ids, prompts, groups, examples = set(), {}, {}, set()
    counts = Counter()
    allowed = {"id", "split", "prompt", "response", "label", "system", "group", "metadata"}
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in ids:
            raise ValueError(f"Every row needs a unique, nonempty string id: {ident!r}")
        ids.add(ident)
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"{ident}: unknown fields {sorted(unknown)}; put extra information in metadata")
        split = row.get("split")
        if split not in {"discovery", "validation", "control"}:
            raise ValueError(f"{ident}: split must be discovery, validation, or control")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{ident}: prompt must be nonempty text")
        if "system" in row and not isinstance(row["system"], str):
            raise ValueError(f"{ident}: system must be text")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"{ident}: metadata must be an object")
        if split == "discovery":
            if not isinstance(row.get("response"), str) or not row["response"].strip():
                raise ValueError(f"{ident}: discovery requires a complete labeled response")
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError(f"{ident}: label must be integer 1 (behavior present) or 0 (absent)")
            counts[f"discovery_label_{row['label']}"] += 1
            example = digest([prompt_key(row), row.get("system", ""), row["response"].strip()])
            if example in examples:
                raise ValueError(f"{ident}: duplicate prompt-response example (possibly conflicting labels)")
            examples.add(example)
        elif "response" in row or "label" in row:
            raise ValueError(f"{ident}: validation/control rows contain prompts only, not response or label")
        key = prompt_key(row)
        if key in prompts and prompts[key] != split:
            raise ValueError(f"{ident}: prompt leaks between {prompts[key]} and {split}")
        if key in prompts and split != "discovery":
            raise ValueError(f"{ident}: duplicate evaluation prompt")
        prompts[key] = split
        if "group" in row:
            group = row["group"]
            if not isinstance(group, str) or not group.strip():
                raise ValueError(f"{ident}: group must be nonempty text")
            if group in groups and groups[group] != split:
                raise ValueError(f"{ident}: group {group!r} leaks across splits")
            groups[group] = split
        counts[split] += 1
    if require_discovery and any(counts[f"discovery_label_{label}"] == 0 for label in (0, 1)):
        raise ValueError("Discovery needs at least one positive and one negative response")
    warnings = []
    if require_discovery and min(counts[f"discovery_label_{label}"] for label in (0, 1)) < 50:
        warnings.append("Fewer than 50 responses per label: suitable for a pilot, not a reliability guarantee.")
    if not counts["validation"]:
        warnings.append("No held-out validation prompts provided.")
    if not counts["control"]:
        warnings.append("No control prompts provided to inspect unrelated capabilities.")
    return {"counts": dict(counts), "warnings": warnings, "dataset_sha256": digest(rows)}
