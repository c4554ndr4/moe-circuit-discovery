from copy import deepcopy

import pytest

from moe_circuits.data import validate, new_run


def test_paired_discovery_is_allowed(records):
    assert validate(records)["counts"]["discovery_label_1"] == 2


@pytest.mark.parametrize("field,value", [("label", True), ("label", "1"), ("label", None),
                                        ("response", " "), ("prompt", "")])
def test_bad_discovery_rejected(records, field, value):
    records[0][field] = value
    with pytest.raises(ValueError):
        validate(records)


def test_normalized_prompt_leakage(records):
    records[-1]["prompt"] = "question   one "
    with pytest.raises(ValueError, match="leaks"):
        validate(records)


def test_semantic_group_leakage(records):
    records[0]["group"] = "same scenario"
    records[-1]["group"] = "same scenario"
    with pytest.raises(ValueError, match="group.*leaks"):
        validate(records)


def test_system_change_cannot_hide_prompt_leakage(records):
    records[-1]["prompt"] = records[0]["prompt"]
    records[-1]["system"] = "A different instruction"
    with pytest.raises(ValueError, match="leaks"):
        validate(records)


def test_duplicate_and_conflicting_response(records):
    duplicate = deepcopy(records[0])
    duplicate.update(id="different-id", label=0)
    with pytest.raises(ValueError, match="duplicate prompt-response"):
        validate(records + [duplicate])


def test_evaluation_cannot_supply_answer_labels(records):
    records[-1]["label"] = 0
    with pytest.raises(ValueError, match="prompts only"):
        validate(records)


def test_no_overwrite(tmp_path):
    new_run(tmp_path / "run")
    (tmp_path / "run" / "artifact").write_text("keep me")
    with pytest.raises(ValueError, match="not empty"):
        new_run(tmp_path / "run")
