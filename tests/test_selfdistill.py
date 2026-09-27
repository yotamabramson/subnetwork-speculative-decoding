"""Batched, left-padded generation must equal per-prompt generation."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ssd.data.generate_selfdistill import generate_batch
from ssd.engine.speculative import TargetRunner, autoregressive_generate


def test_batched_padded_greedy_matches_single():
    torch.manual_seed(0)
    base = LlamaForCausalLM(
        LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                    num_attention_heads=4, num_key_value_heads=2)
    ).eval()
    target = TargetRunner(base)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, 120, (n,), generator=g).tolist() for n in (3, 9, 6, 1)]
    eos = {125}
    rows = generate_batch(target, prompts, 20, eos, temperature=0.0, top_p=1.0, pad_id=127)
    for p, row in zip(prompts, rows):
        ref = autoregressive_generate(target, torch.tensor([p]), 20, eos).tokens
        assert row == ref
