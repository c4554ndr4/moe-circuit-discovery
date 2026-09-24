"""Pinned model loading and explicit final-channel prompt/response encoding."""

from datetime import datetime, timezone
import importlib.metadata
import hashlib
from pathlib import Path
import re

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

from .data import digest


def load_runtime(model_id, revision="main", device="auto", template_date=None):
    if transformers.__version__ != "4.57.6":
        raise RuntimeError("This adapter requires transformers==4.57.6; install the model extra.")
    template_date = template_date or datetime.now(timezone.utc).date().isoformat()
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if config.model_type != "gpt_oss":
        raise ValueError("Only GPT-OSS is supported; refusing to load another architecture's weights")
    revision = getattr(config, "_commit_hash", None) or revision
    local_hashes = {}
    if Path(model_id).is_dir():
        for path in sorted(Path(model_id).glob("*.safetensors")):
            with path.open("rb") as handle:
                local_hashes[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        if not local_hashes:
            raise ValueError("Local checkpoints must have safetensors weight files for provenance")
    quantization = getattr(config, "quantization_config", None)
    load_options = {}
    if quantization:
        if quantization.get("quant_method") != "mxfp4":
            raise ValueError("Only unquantized or MXFP4-to-bf16 GPT-OSS checkpoints are supported")
        load_options["quantization_config"] = Mxfp4Config(dequantize=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=False, dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager",
        **load_options,
    ).eval()
    if getattr(model, "hf_device_map", {}).values() and "disk" in model.hf_device_map.values():
        raise ValueError("Disk offloading is not supported for expert inspection")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    template = tokenizer.get_chat_template()
    # GPT-OSS inserts the current date. Freeze it in saved run provenance.
    template = re.sub(r"strftime_now\([\"']%Y-%m-%d[\"']\)", repr(template_date), template)
    tokenizer.chat_template = template
    metadata = {
        "model_id": model_id,
        "revision": getattr(model.config, "_commit_hash", None) or revision,
        "template_date": template_date,
        "template_sha256": digest(template),
        "model_config_sha256": digest(model.config.to_dict()),
        "local_weights_sha256": local_hashes,
        "dtype": "bfloat16", "attention": "eager", "experts": "dequantized",
        "response_mode": "final_only",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "accelerate", "numpy")},
    }
    return model, tokenizer, metadata


def encode(tokenizer, row, device, max_tokens, include_response=False):
    if include_response and any(token and token in row["response"] for token in tokenizer.all_special_tokens):
        raise ValueError(f"{row['id']}: response must be plain final text, without special/chat control tokens")
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": row["system"]})
    messages.append({"role": "user", "content": row["prompt"]})
    marker = "__MOE_CIRCUIT_RESPONSE_BOUNDARY_713fde__"
    messages.append({"role": "assistant", "content": marker})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if rendered.count(marker) != 1:
        raise ValueError("Chat template did not preserve an unambiguous response boundary")
    prefix, _suffix = rendered.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    text = prefix + row["response"] if include_response else prefix
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not prefix_ids or ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Tokenizer merges across the response boundary; unsupported template")
    if include_response and len(ids) <= len(prefix_ids):
        raise ValueError(f"{row['id']}: response contains no tokens")
    if len(ids) > max_tokens:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds --max-tokens={max_tokens}; no truncation applied")
    tensor = torch.tensor([ids], device=device, dtype=torch.long)
    return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}, len(prefix_ids)


def input_device(model):
    return model.get_input_embeddings().weight.device


@torch.inference_mode()
def generate(model, tokenizer, inputs, max_new_tokens):
    ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )[0, inputs["input_ids"].shape[1]:].tolist()
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    return {
        "response": tokenizer.decode(ids, skip_special_tokens=True).strip(),
        "generated_tokens": len(ids),
        "ended_with_eos": bool(ids and ids[-1] in eos),
        "hit_token_limit": len(ids) >= max_new_tokens and not (ids and ids[-1] in eos),
    }
