import pytest
import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

from moe_circuits.adapter import ExpertAdapter, check_equivalence


def test_native_logits_and_cached_decode_equivalence(tiny_model):
    inputs = {"input_ids": torch.tensor([[5, 6, 7, 8]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    adapter = ExpertAdapter(tiny_model)
    result = check_equivalence(tiny_model, adapter, inputs, atol=1e-6, rtol=1e-5)
    assert result["passed"] and result["max_absolute_logit_error"] < 1e-6
    with torch.inference_mode():
        native = tiny_model.generate(**inputs, max_new_tokens=3, do_sample=False)
        with adapter:
            patched = tiny_model.generate(**inputs, max_new_tokens=3, do_sample=False)
    assert torch.equal(native, patched)


def test_surgical_ablation_matches_zeroed_down_projection(tiny_model):
    # Independent reference: zero the selected neuron's outgoing weight row.
    adapter = ExpertAdapter(tiny_model)
    module = tiny_model.model.layers[0].mlp.experts
    hidden = torch.randn(1, 3, 16)
    indices = torch.tensor([[0, 1], [1, 2], [2, 3]])
    weights = torch.tensor([[.6, .4, 0., 0.], [0., .3, .7, 0.], [0., 0., .8, .2]])
    original_forward = module.forward
    with torch.inference_mode():
        unmodified = module(hidden, indices, weights)
        saved = module.down_proj[1, 3].clone()
        module.down_proj[1, 3] = 0
        reference = module(hidden, indices, weights)
        module.down_proj[1, 3] = saved
        adapter.set_targets([{"layer": 0, "expert": 1, "neuron": 3}])
        with adapter:
            actual = module(hidden, indices, weights)
    torch.testing.assert_close(actual, reference)
    assert not torch.equal(actual, unmodified)
    torch.testing.assert_close(actual[:, 2], unmodified[:, 2])  # Expert 1 was not selected.
    assert adapter.events == 2
    assert module.forward == original_forward


def test_restores_after_exception(tiny_model):
    module = tiny_model.model.layers[0].mlp.experts
    original = module.forward
    with pytest.raises(RuntimeError, match="intentional"):
        with ExpertAdapter(tiny_model):
            raise RuntimeError("intentional")
    assert module.forward == original
    assert "forward" not in module.__dict__
    assert module.forward.__func__ is GptOssExperts.forward


def test_rejects_bad_targets(tiny_model):
    with pytest.raises(ValueError, match="Invalid"):
        ExpertAdapter(tiny_model).set_targets([{"layer": 0, "expert": 0, "neuron": 999}])
