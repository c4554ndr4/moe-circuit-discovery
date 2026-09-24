"""Human labels tied to immutable generation hashes; no refusal-specific scoring."""

from collections import defaultdict
from pathlib import Path

from .data import read_jsonl, write_json


def summarize(run, labels_path):
    run = Path(run)
    generations = read_jsonl(run / "generations.jsonl")
    labels = read_jsonl(labels_path)
    expected = {row["id"]: row for row in generations}
    if len(expected) != len(generations):
        raise ValueError("Duplicate generation ids")
    seen = set()
    for label in labels:
        ident = label.get("id")
        if ident not in expected or ident in seen:
            raise ValueError(f"Unknown or duplicate scored generation: {ident}")
        seen.add(ident)
        if label.get("output_sha256") != expected[ident]["output_sha256"]:
            raise ValueError(f"{ident}: label does not match the saved output hash")
        for field in ("behavior_present", "acceptable"):
            value = label.get(field)
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError(f"{ident}: {field} must be 0, 1, or null")
            expected[ident][field] = value
    if seen != set(expected):
        raise ValueError("Labels must include every generation id; use null for judgments not yet available")
    groups = defaultdict(list)
    indexed = {}
    for row in expected.values():
        groups[(row["split"], row["variant"])].append(row)
        indexed[(row["example_id"], row["variant"])] = row
    metrics = []
    for (split, variant), rows in sorted(groups.items()):
        item = {"split": split, "variant": variant, "total": len(rows),
                "hit_token_limit": sum(row["hit_token_limit"] for row in rows)}
        for field in ("behavior_present", "acceptable"):
            scored = [row[field] for row in rows if row.get(field) is not None]
            item[field] = {"scored": len(scored), "rate": sum(scored) / len(scored) if scored else None}
        if variant != "baseline":
            pairs = [(indexed[(row["example_id"], "baseline")], row) for row in rows]
            for field in ("behavior_present", "acceptable"):
                judged = [(a[field], b[field]) for a, b in pairs
                          if a.get(field) is not None and b.get(field) is not None]
                item[f"paired_{field}"] = {
                    "scored_pairs": len(judged),
                    "one_to_zero": sum(a == 1 and b == 0 for a, b in judged),
                    "zero_to_one": sum(a == 0 and b == 1 for a, b in judged),
                    "mean_change": sum(b - a for a, b in judged) / len(judged) if judged else None,
                }
        metrics.append(item)
    complete = all(row.get("behavior_present") is not None and row.get("acceptable") is not None
                   for row in expected.values())
    summary = {"status": "fully_scored" if complete else "partially_scored",
               "interpretation": "Descriptive paired effects, not a proof of a complete circuit. "
                                 "Compare candidate with random controls and inspect quality/truncation.",
               "metrics": metrics}
    write_json(run / "summary.json", summary)
    lines = ["# Ablation evaluation", "", f"Scoring status: **{summary['status']}**", "",
             summary["interpretation"], "",
             "| Split | Variant | N | Behavior rate (scored) | Acceptable rate (scored) | Token limit |",
             "|---|---|---:|---|---|---:|"]
    for item in metrics:
        values = []
        for field in ("behavior_present", "acceptable"):
            value = item[field]
            rate = f"{value['rate']:.1%}" if value["rate"] is not None else "unscored"
            values.append(f"{rate} ({value['scored']}/{item['total']})")
        lines.append(f"| {item['split']} | {item['variant']} | {item['total']} | "
                     f"{values[0]} | {values[1]} | {item['hit_token_limit']} |")
    lines += ["", "Rates with different scoring coverage are not directly comparable. "
              "Paired changes are in summary.json. Negative behavior change means suppression; "
              "negative acceptable change means quality loss. A random control matches neuron counts "
              "per expert, not activation strength. Repeated tuning on this validation set requires "
              "a new final test set.", ""]
    (run / "report.md").write_text("\n".join(lines))
    return summary
