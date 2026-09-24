"""Reversible GPT-OSS expert forward adapter, based on the original CNA experiment.

Only the pinned Transformers 4.57.6 dense routing-weight layout is supported.
All-neuron capture avoids the original top-k trace censoring. No router edits.
"""

from contextlib import AbstractContextManager
import types

import torch


class ExpertAdapter(AbstractContextManager):
    def __init__(self, model):
        if model.config.model_type != "gpt_oss":
            raise ValueError("Only GPT-OSS is supported")
        self.model = model
        self.targets = {}
        self.observe = None
        self.events = 0
        self.calls = set()
        self.saved = []
        self.modules = {i: layer.mlp.experts for i, layer in enumerate(model.model.layers)}
        self.dims = (len(self.modules), model.config.num_local_experts, model.config.intermediate_size)
        for module in self.modules.values():
            expected = (self.dims[1], model.config.hidden_size, 2 * self.dims[2])
            if tuple(module.gate_up_proj.shape) != expected or not module.gate_up_proj.is_floating_point():
                raise ValueError("Unsupported expert weights. Load dequantized bf16 experts, not MXFP4 kernels.")

    def set_targets(self, coordinates):
        targets = {}
        seen = set()
        for row in coordinates:
            coord = tuple(row[key] for key in ("layer", "expert", "neuron"))
            if any(type(value) is not int or not 0 <= value < dim for value, dim in zip(coord, self.dims)):
                raise ValueError(f"Invalid circuit coordinate {coord}, model dimensions {self.dims}")
            if coord in seen:
                raise ValueError(f"Duplicate circuit coordinate {coord}")
            seen.add(coord)
            layer, expert, neuron = coord
            targets.setdefault(layer, {}).setdefault(expert, []).append(neuron)
        self.targets = targets
        self.events = 0

    def __enter__(self):
        if self.saved:
            raise RuntimeError("Adapter is already installed")
        try:
            for layer, module in self.modules.items():
                # Remember whether an instance override existed, not just its bound value.
                self.saved.append((module, module.__dict__.get("forward"), "forward" in module.__dict__))
                module.forward = types.MethodType(self._forward(layer), module)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for module, previous, had_override in reversed(self.saved):
            if had_override:
                module.forward = previous
            else:
                del module.forward
        self.saved.clear()
        self.observe = None
        return False

    def _forward(self, layer):
        def forward(module, hidden_states, router_indices=None, routing_weights=None):
            if module.training:
                raise RuntimeError("Circuit discovery is inference-only: call model.eval()")
            shape = hidden_states.shape
            hidden = hidden_states.reshape(-1, module.hidden_size)
            if routing_weights is None or tuple(routing_weights.shape) != (len(hidden), module.num_experts):
                raise RuntimeError("Unsupported routing layout; use transformers==4.57.6")
            if router_indices is None or router_indices.shape[0] != len(hidden):
                raise RuntimeError("Missing or malformed routed expert indices")
            self.calls.add(layer)
            result = torch.zeros_like(hidden)
            for expert_tensor in torch.unique(router_indices):
                expert = int(expert_tensor)
                if not 0 <= expert < module.num_experts:
                    raise RuntimeError("Unexpected expert index")
                token_ids = torch.where((router_indices == expert).any(dim=1))[0]
                gate_up = hidden[token_ids] @ module.gate_up_proj[expert] + module.gate_up_proj_bias[expert]
                gate = gate_up[..., ::2].clamp(max=module.limit)
                up = gate_up[..., 1::2].clamp(-module.limit, module.limit)
                gated = (up + 1) * (gate * torch.sigmoid(gate * module.alpha))
                weight = routing_weights[token_ids, expert, None]
                if self.observe is not None:
                    self.observe(layer, expert, token_ids, gated, weight)
                neurons = self.targets.get(layer, {}).get(expert, [])
                if neurons:
                    gated[:, neurons] = 0
                    self.events += len(token_ids) * len(neurons)
                projected = gated @ module.down_proj[expert] + module.down_proj_bias[expert]
                result.index_add_(0, token_ids, (projected * weight).to(result.dtype))
            return result.reshape(shape)

        return forward


@torch.inference_mode()
def check_equivalence(model, adapter, inputs, atol=0.05, rtol=0.01):
    """Compare native and adapter logits on the user's actual checkpoint before running."""
    native = model(**inputs, use_cache=False).logits.float()
    adapter.set_targets([])
    adapter.calls.clear()
    with adapter:
        patched = model(**inputs, use_cache=False).logits.float()
    if adapter.calls != set(adapter.modules):
        raise RuntimeError("Not every expert module executed; a fused kernel may bypass the adapter")
    torch.testing.assert_close(patched, native, atol=atol, rtol=rtol)
    return {"passed": True, "atol": atol, "rtol": rtol,
            "max_absolute_logit_error": float((patched - native).abs().max()),
            "tokens_checked": int(inputs["input_ids"].numel())}
