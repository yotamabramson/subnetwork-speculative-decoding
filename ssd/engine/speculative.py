"""Speculative decoding loop and the autoregressive baseline.

The target runs through ``SubnetworkDraftModel`` with *all* layers selected
(no bridges). That is the same code path as the draft, which gives us tree
masks and cache compaction, and it matches HF ``LlamaForCausalLM`` exactly
(see tests). The baseline uses the same path, so speedups compare like with like.

One KV cache, shared by target and draft (batch size 1). Between rounds it
holds the target's real KV, for every layer, of all committed tokens except the
last one, the root. Each round:

1. Draft: the draft layers process the root, then the tree level by level. They
   attend to the target's real KV for the committed prefix and append entries
   (at their own layers only) for the speculative tokens.
2. Crop the cache back to the committed prefix, which discards the draft's
   speculative entries.
3. Verify: the target processes the root plus the tree in one forward, writing
   real KV for all of them. Keep the root and the accepted path; the bonus token
   becomes the next root.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from ssd.config import DraftingConfig
from ssd.engine.kv_cache import KVCache
from ssd.engine.tree_drafter import TreeDrafter, warp_probs
from ssd.engine.verify import greedy_verify, sample_verify, verify_inputs
from ssd.models.subnetwork_draft import SubnetworkDraftModel


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class _Timer:
    def __init__(self, device):
        self.device = device
        self.totals: dict[str, float] = {}

    def __call__(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                _sync(timer.device)
                self.t0 = time.perf_counter()

            def __exit__(self, *exc):
                _sync(timer.device)
                timer.totals[name] = timer.totals.get(name, 0.0) + time.perf_counter() - self.t0

        return _Ctx()


@dataclass
class GenerationResult:
    tokens: list[int]
    times: dict[str, float]  # seconds per phase
    accepted_per_round: list[int] = field(default_factory=list)  # tokens committed per round (accepted + bonus)
    tree_sizes: list[int] = field(default_factory=list)

    @property
    def decode_time(self) -> float:
        return sum(v for k, v in self.times.items() if k != "prefill")

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted_per_round) / max(1, len(self.accepted_per_round))


def _first_token(logits: torch.Tensor, temperature: float, top_p: float, generator) -> int:
    if temperature == 0:
        return int(logits.argmax())
    return int(torch.multinomial(warp_probs(logits, temperature, top_p).cpu(), 1, generator=generator))


def _finish(tokens: list[int], eos_ids: set[int], max_new: int) -> tuple[list[int], bool]:
    for i, t in enumerate(tokens[:max_new]):
        if t in eos_ids:
            return tokens[: i + 1], True
    return tokens[:max_new], len(tokens) >= max_new


class TargetRunner(SubnetworkDraftModel):
    """The full target through the draft's layer implementation."""

    def __init__(self, base: nn.Module):
        super().__init__(base, list(range(base.config.num_hidden_layers)))


@torch.no_grad()
def autoregressive_generate(
    target: TargetRunner,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_ids: set[int],
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> GenerationResult:
    dev = input_ids.device
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    timer = _Timer(dev)
    cache = target.new_cache(input_ids.shape[1] + max_new_tokens + 1)
    with timer("prefill"):
        logits = target(input_ids, past_key_values=cache, logits_to_keep=1).logits[0, -1]
        out = [_first_token(logits, temperature, top_p, gen)]
    with timer("decode"):
        while not _finish(out, eos_ids, max_new_tokens)[1]:
            logits = target(torch.tensor([[out[-1]]], device=dev), past_key_values=cache).logits[0, -1]
            out.append(_first_token(logits, temperature, top_p, gen))
    return GenerationResult(tokens=_finish(out, eos_ids, max_new_tokens)[0], times=timer.totals)


class SpeculativeGenerator:
    def __init__(self, base: nn.Module, draft: SubnetworkDraftModel, cfg: DraftingConfig, target: Optional[TargetRunner] = None):
        self.target = target or TargetRunner(base)
        self.draft = draft
        self.cfg = cfg
        self.drafter = TreeDrafter(draft, cfg)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_ids: set[int],
        seed: Optional[int] = None,
    ) -> GenerationResult:
        cfg = self.cfg
        dev = input_ids.device
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        timer = _Timer(dev)
        n_prompt = input_ids.shape[1]
        # Slack: one round can overshoot max_new_tokens by up to depth+1 tokens.
        extra = max(self.drafter.max_verify_nodes(), self.drafter.max_tree_cache()) + 1
        cache = self.target.new_cache(n_prompt + max_new_tokens + cfg.depth + extra + 1)

        with timer("prefill"):
            logits = self.target(input_ids, past_key_values=cache, logits_to_keep=1).logits[0, -1]
            out = [_first_token(logits, cfg.temperature, cfg.top_p, gen)]
        result = GenerationResult(tokens=[], times=timer.totals)

        while not _finish(out, eos_ids, max_new_tokens)[1]:
            committed = cache.get_seq_length()  # == position of the root
            with timer("draft"):
                tree = self.drafter.build(cache, torch.tensor([[out[-1]]], device=dev), committed, gen)
                cache.crop(committed)
            with timer("verify"):
                ids, pos, mask = verify_inputs(tree, out[-1], committed, committed, dev)
                logits = self.target(ids, position_ids=pos, past_key_values=cache, tree_attention_mask=mask).logits[0]
                if cfg.temperature == 0:
                    accepted, bonus = greedy_verify(tree, logits)
                else:
                    accepted, bonus = sample_verify(tree, logits, cfg.temperature, cfg.top_p, gen)
            with timer("bookkeeping"):
                cache.keep_positions(committed, torch.tensor([committed] + [committed + 1 + i for i in accepted]))
                new = [tree.tokens[i] for i in accepted] + [bonus]
                out.extend(new)
                result.accepted_per_round.append(len(new))
                result.tree_sizes.append(len(tree))

        result.tokens = _finish(out, eos_ids, max_new_tokens)[0]
        return result
