# Dataset contract

## Supported format

UTF-8 JSONL, one object per line. Supply one single-turn prompt per row. Multi-turn conversations, images, tool calls, and analysis-channel traces are not supported in this version.

| Field | Required | Meaning |
|---|---|---|
| `id` | Always | Unique nonempty string |
| `split` | Always | `discovery`, `validation`, or `control` |
| `prompt` | Always | The exact user message |
| `response` | Discovery only | Complete plain-text assistant final response |
| `label` | Discovery only | Integer `1`: behavior present; integer `0`: absent |
| `system` | Optional | Consistent instructions; rendered through the model's template |
| `group` | Recommended | Shared identifier for variants of the same underlying scenario |
| `metadata` | Optional | Object for source, annotation rationale, model ID, etc. |

Unknown top-level fields are rejected to catch typos. Put extra information in `metadata`. Evaluation rows must not contain `response` or `label`: all outcomes are measured after generating fresh responses. A prompt alone cannot establish whether the behavior occurred, so discovery requires labeled responses.

## Define the behavior before labeling

Write a short rule and boundary cases. For example: “An unwarranted apology in an otherwise answerable factual response; do not count an appropriate apology after an actual error.” A label describes what the response does, not the dataset's topic or your preferred outcome.

For sycophancy, mere agreement is insufficient evidence. Identify unsupported agreement or a change toward a user's erroneous belief. For factuality, an incorrect answer need not imply intentional deception. Unclear examples should be reviewed or excluded rather than forced into a class.

## Build useful contrasts

- Prefer actual outputs of the checkpoint and response mode being studied. Record their source in metadata. This is provenance supplied by the user, not something the validator can verify.
- Positive and negative examples should overlap in topic, difficulty, instructions, and response length. Avoid making “positive = one topic, negative = another topic.”
- You may use the same prompt with different positive/negative responses **within discovery**. These are not independent prompts, and they should not be counted as such when estimating generalization.
- Authored counterfactual answers are accepted for representation discovery, but may be unnatural for the model. Successful teacher-forced separation is not evidence of successful generated-behavior intervention.
- Avoid inserting strings such as “positive example” or behavior labels into model-visible text.
- Prefer full responses over A/B letters. If using letters, balance answer ordering and test open-ended transfer.

## Split before discovery

Keep all paraphrases, resamples, and contrasting answers from one underlying scenario in the same split. Give them a shared `group`. The validator catches repeated prompts and explicit groups across splits; it cannot catch semantic overlap you did not mark.

Discovery selects neuron candidates. Validation measures intervention effects and may inform pilot decisions. Controls measure unrelated abilities and answer quality. If you tune neuron counts or dataset choices repeatedly against validation, reserve a separate final test panel and evaluate it only after selection. You can supply that panel later as a new file with `validation`/`control` rows; the circuit artifact still enforces discovery-prompt/group exclusion.

## How many examples?

A reasonable pilot budget is 50 positives and 50 negatives, 50–100 validation prompts, and 100+ control prompts. These are planning suggestions, **not validated minimum sample sizes**. Some behaviors need substantially more. The tiny bundled dataset tests formatting only.

Inspect baseline behavior frequency. Collect more discovery data if independently sampled datasets produce unstable rankings or inconsistent held-out effects. A high neuron-overlap score by itself does not establish causal specificity.

## What counts as success?

The selected ablation should reduce the target behavior on fresh prompts more than matched random ablations, without unacceptable loss of answer quality or unrelated capabilities. Inspect complete outputs and per-example flips. Report unknown labels and token-limit hits. A response that becomes empty, incoherent, or generically disagreeable is not a successful selective intervention.

This version produces descriptive measurements; it does not perform automatic statistical significance testing, minimality/sufficiency proofs, or claim a complete circuit. Specify and score domain-specific capability tests where generic control prompts are inadequate.
