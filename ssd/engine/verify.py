"""Tree verification.

The target scores ``[root] + tree nodes`` in one forward. Each node attends to
the committed prefix, the root, and its own ancestors (``verify_inputs``
builds the ids / positions / mask). Then, starting at the root:

* Greedy: move to the child whose token equals the target's argmax. When no
  child matches, the argmax becomes the bonus token. The output is identical
  to greedy decoding with the target alone.
* Sampling: multi-candidate speculative sampling without replacement. Each
  child x, in draw order, is accepted with prob min(1, p(x)/q(x)). On
  rejection, p <- norm(max(p - q, 0)) and q <- q with x removed,
  renormalised. If every child is rejected, the bonus token is sampled from
  the final p. When the children are an unpruned draw from q, this reproduces
  sampling from the target exactly (Yang et al. 2024, "Multi-Candidate
  Speculative Decoding"). Width / max_nodes pruning keeps sibling subsets
  chosen by score, which makes it near-exact, the same trade-off EAGLE-2
  makes.

Every node gains one bonus token, so a round commits accepted + 1 tokens.
"""

from __future__ import annotations

from typing import Optional

import torch

from ssd.engine.tree_drafter import ROOT, DraftTree, warp_probs


def verify_inputs(tree: DraftTree, root_token: int, root_pos: int, past_len: int, device) -> tuple:
    """ids [1, n+1], positions [1, n+1], bool mask [n+1, past+n+1]. Row 0 is the root."""
    n = len(tree)
    ids = torch.tensor([[root_token] + tree.tokens], device=device)
    pos = torch.tensor([[root_pos] + [root_pos + d for d in tree.depths]], device=device)
    mask = torch.zeros(n + 1, past_len + n + 1, dtype=torch.bool)
    mask[:, : past_len + 1] = True  # prefix + root
    for i in range(n):
        for a in tree.ancestors(i):
            mask[i + 1, past_len + 1 + a] = True
    return ids, pos, mask.to(device)


def greedy_verify(tree: DraftTree, target_logits: torch.Tensor) -> tuple[list[int], int]:
    """target_logits [n+1, V] (row 0 = root). -> (accepted node indices, bonus token)."""
    argmax = target_logits.argmax(-1).tolist()
    kids = tree.children()
    accepted: list[int] = []
    cur = ROOT
    while True:
        want = argmax[cur + 1]
        nxt = next((c for c in kids.get(cur, []) if tree.tokens[c] == want), None)
        if nxt is None:
            return accepted, want
        accepted.append(nxt)
        cur = nxt


def sample_verify(
    tree: DraftTree,
    target_logits: torch.Tensor,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[list[int], int]:
    kids = tree.children()
    accepted: list[int] = []
    cur = ROOT
    while True:
        p = warp_probs(target_logits[cur + 1], temperature, top_p).cpu()
        children = kids.get(cur, [])
        if children:
            q = tree.child_q[cur].clone()
            chosen = None
            for c in children:
                x = tree.tokens[c]
                if q[x] <= 0:
                    continue
                if torch.rand((), generator=generator).item() < min(1.0, (p[x] / q[x]).item()):
                    chosen = c
                    break
                residual = (p - q).clamp_min(0)
                if residual.sum() > 0:
                    p = residual / residual.sum()
                q[x] = 0
                if q.sum() <= 0:
                    break
                q = q / q.sum()
            if chosen is not None:
                accepted.append(chosen)
                cur = chosen
                continue
        bonus = torch.multinomial(p, 1, generator=generator).item()
        return accepted, bonus
