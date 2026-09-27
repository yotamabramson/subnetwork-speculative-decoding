"""Dynamic tree drafting with the sub-network draft.

One round. The cache holds the target's real KV for the committed prefix, and
``pending`` is the last committed token (the tree root):

1. Feed ``pending``. Its logits give the root's next-token distribution.
2. For depth d = 1..K: every node kept at depth d-1 proposes ``branch[d]``
   children. With greedy decoding these are the top tokens; with sampling they
   are drawn without replacement (Gumbel top-k), in draw order. Across the
   whole level, the ``width[d]`` children with the highest cumulative draft
   log-prob are kept. Unless d == K, the kept children go through the draft in
   ONE batched forward, where each attends only to the committed prefix and
   its own ancestors, and that forward yields their children's distributions.
3. The best ``max_nodes`` nodes by cumulative log-prob form the tree sent to
   verification. The set is closed under ancestors, because a child's score
   is never above its parent's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ssd.config import DraftingConfig
from ssd.engine.kv_cache import KVCache
from ssd.models.subnetwork_draft import SubnetworkDraftModel

ROOT = -1


def warp_probs(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Temperature + nucleus filtering -> probabilities (fp32). Same warp is
    applied to draft and target so speculative sampling targets the warped p."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(dim=-1, descending=True)
        cum = sorted_p.cumsum(-1)
        drop = cum - sorted_p > top_p  # keep the smallest prefix with mass >= top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


@dataclass
class DraftTree:
    tokens: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)  # ROOT (-1) or node index
    depths: list[int] = field(default_factory=list)  # 1..K
    scores: list[float] = field(default_factory=list)  # cumulative draft log-prob
    ranks: list[int] = field(default_factory=list)  # proposal order among siblings
    draft_pos: list[int] = field(default_factory=list)  # draft-cache position, -1 if never fed
    # Sampling only: the warped draft distribution each parent's children were drawn from.
    child_q: dict[int, torch.Tensor] = field(default_factory=dict)
    cache_start: int = 0  # first draft-cache slot written by tree nodes this round

    def __len__(self) -> int:
        return len(self.tokens)

    def children(self) -> dict[int, list[int]]:
        """parent -> child node indices in proposal order."""
        out: dict[int, list[int]] = {}
        for i, p in enumerate(self.parents):
            out.setdefault(p, []).append(i)
        for kids in out.values():
            kids.sort(key=lambda i: self.ranks[i])
        return out

    def ancestors(self, i: int) -> list[int]:
        """Node indices on the path root -> i, inclusive."""
        path = []
        while i != ROOT:
            path.append(i)
            i = self.parents[i]
        return path[::-1]

    def subset(self, keep: list[int]) -> "DraftTree":
        keep = sorted(keep)
        remap = {old: new for new, old in enumerate(keep)}
        remap[ROOT] = ROOT
        t = DraftTree(cache_start=self.cache_start)
        for old in keep:
            t.tokens.append(self.tokens[old])
            t.parents.append(remap[self.parents[old]])
            t.depths.append(self.depths[old])
            t.scores.append(self.scores[old])
            t.ranks.append(self.ranks[old])
            t.draft_pos.append(self.draft_pos[old])
        for p, q in self.child_q.items():
            if p in remap:
                t.child_q[remap[p]] = q
        return t


class TreeDrafter:
    def __init__(self, draft: SubnetworkDraftModel, cfg: DraftingConfig):
        cfg.validate()
        self.draft = draft
        self.cfg = cfg
        self.branch = cfg.per_depth(cfg.branch, "branch")
        self.width = cfg.per_depth(cfg.width, "width")
        self.greedy = cfg.temperature == 0

    def max_nodes_per_level(self) -> list[int]:
        n, out = 1, []
        for d in range(self.cfg.depth):
            n *= self.branch[d]
            if self.width[d] is not None:
                n = min(n, self.width[d])
            out.append(n)
        return out

    def max_tree_cache(self) -> int:
        """Upper bound on draft-cache slots one round's tree uses (the last level is never fed)."""
        return sum(self.max_nodes_per_level()[:-1])

    def max_verify_nodes(self) -> int:
        total = sum(self.max_nodes_per_level())
        return min(total, self.cfg.max_nodes) if self.cfg.max_nodes else total

    def _propose(self, logits: torch.Tensor, k: int, generator: Optional[torch.Generator]):
        """logits [n, V'] for n parents -> per parent (tokens, log-probs, warped q or None).
        With a draft vocab subset, V' < V and columns map through ``draft.vocab_ids``;
        q is scattered back to the full vocab (zero outside the subset)."""
        vocab_ids = self.draft.vocab_ids
        if self.greedy:
            lp, tok = F.log_softmax(logits.float(), -1).topk(k, dim=-1)
            if vocab_ids is not None:
                tok = vocab_ids[tok]
            return [(t, l, None) for t, l in zip(tok.tolist(), lp.tolist())]
        probs = warp_probs(logits, self.cfg.temperature, self.cfg.top_p)
        if vocab_ids is not None:
            full = torch.zeros(probs.shape[0], self.draft.base.lm_head.weight.shape[0], device=probs.device)
            probs = full.index_copy_(1, vocab_ids.to(probs.device), probs)
        out = []
        for q in probs.cpu():
            support = int((q > 0).sum())
            gumbel = -torch.empty_like(q).exponential_(generator=generator).log()
            keys = q.log() + gumbel  # -inf outside the support
            tok = keys.topk(min(k, support)).indices  # descending keys == draw order
            out.append((tok.tolist(), q[tok].log().tolist(), q))
        return out

    @torch.no_grad()
    def build(
        self,
        cache: KVCache,
        pending: torch.Tensor,
        root_pos: int,
        generator: Optional[torch.Generator] = None,
    ) -> DraftTree:
        """``pending``: [1, n] committed tokens not yet in ``cache``; the last one
        is the tree root at absolute position ``root_pos``."""
        dev = pending.device
        n_pend = pending.shape[1]
        pos = torch.arange(root_pos - n_pend + 1, root_pos + 1, device=dev).unsqueeze(0)
        logits = self.draft(pending, position_ids=pos, past_key_values=cache, logits_to_keep=1).logits[0]
        tree_start = cache.get_seq_length()  # first cache slot used by tree nodes

        tree = DraftTree(cache_start=tree_start)
        frontier, frontier_logits = [ROOT], logits[-1:]
        for d in range(self.cfg.depth):
            cands = []  # (score, parent, token, rank)
            for parent, (toks, lps, q) in zip(frontier, self._propose(frontier_logits, self.branch[d], generator)):
                if q is not None:
                    tree.child_q[parent] = q
                base = tree.scores[parent] if parent != ROOT else 0.0
                cands += [(base + lp, parent, t, r) for r, (t, lp) in enumerate(zip(toks, lps))]
            cands.sort(key=lambda c: -c[0])
            if self.width[d] is not None:
                cands = cands[: self.width[d]]
            level = []
            for score, parent, tok, rank in cands:
                level.append(len(tree))
                tree.tokens.append(tok)
                tree.parents.append(parent)
                tree.depths.append(d + 1)
                tree.scores.append(score)
                tree.ranks.append(rank)
                tree.draft_pos.append(-1)
            if d == self.cfg.depth - 1 or not level:
                break

            # Feed this level through the draft in one batched forward. Nodes attend
            # to the committed prefix (the target's KV) plus their own ancestors.
            past = cache.get_seq_length()
            for j, i in enumerate(level):
                tree.draft_pos[i] = past + j
            mask = torch.zeros(len(level), past + len(level), dtype=torch.bool)
            mask[:, :tree_start] = True
            for j, i in enumerate(level):
                for a in tree.ancestors(i):
                    mask[j, tree.draft_pos[a]] = True
            ids = torch.tensor([[tree.tokens[i] for i in level]], device=dev)
            pos = torch.full((1, len(level)), root_pos + d + 1, device=dev)
            frontier = level
            frontier_logits = self.draft(ids, position_ids=pos, past_key_values=cache, tree_attention_mask=mask.to(dev)).logits[0]

        if self.cfg.max_nodes is not None and len(tree) > self.cfg.max_nodes:
            order = sorted(range(len(tree)), key=lambda i: (-tree.scores[i], tree.depths[i]))
            keep = set(order[: self.cfg.max_nodes])
            for i in list(keep):  # guard against float ties breaking ancestor-closure
                keep.update(tree.ancestors(i))
            tree = tree.subset(sorted(keep))
        return tree
