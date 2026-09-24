import pytest
import torch
from transformers import GptOssConfig, GptOssForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers


@pytest.fixture
def tiny_model():
    torch.manual_seed(17)
    config = GptOssConfig(
        vocab_size=24, hidden_size=16, intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4, num_local_experts=4, num_experts_per_tok=2,
        max_position_embeddings=128, sliding_window=32,
        rope_scaling={"rope_type": "default"},
        pad_token_id=0, eos_token_id=2, bos_token_id=1,
    )
    config._attn_implementation = "eager"
    model = GptOssForCausalLM(config).eval()
    # Ensure router diversity, rather than relying on tied initialized logits.
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.router.weight.normal_(0, 0.2)
            layer.mlp.experts.gate_up_proj.normal_(0, 0.1)
            layer.mlp.experts.down_proj.normal_(0, 0.1)
    return model


@pytest.fixture
def tiny_tokenizer():
    vocab = {word: i for i, word in enumerate([
        "[PAD]", "[UNK]", "[EOS]", "[USER]", "[FINAL]", "hello", "sorry", "answer",
        "one", "two", "three", "four", "five", "six", "seven", "eight", "yes", "no",
        "how", "what", "is", "a", "test", "question",
    ])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                  pad_token="[PAD]", eos_token="[EOS]",
                                  additional_special_tokens=["[USER]", "[FINAL]"])
    fast.chat_template = ("{% for m in messages %}{% if m.role == 'user' %}[USER] {{m.content}} "
                          "{% elif m.role == 'assistant' %}[FINAL] {{m.content}} [EOS]"
                          "{% else %}{{m.content}} {% endif %}{% endfor %}")
    return fast


@pytest.fixture
def records():
    return [
        {"id": "p1", "split": "discovery", "prompt": "question one", "response": "sorry answer", "label": 1},
        {"id": "n1", "split": "discovery", "prompt": "question one", "response": "answer", "label": 0},
        {"id": "p2", "split": "discovery", "prompt": "question two", "response": "sorry yes", "label": 1},
        {"id": "n2", "split": "discovery", "prompt": "question two", "response": "yes", "label": 0},
        {"id": "v", "split": "validation", "prompt": "question three"},
        {"id": "c", "split": "control", "prompt": "question four"},
    ]
