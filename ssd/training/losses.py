from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def feature_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """alpha * MSE + beta * (1 - cos), per token, averaged over ``mask``.

    MSE is divided by the target's per-token mean square. Residual-stream
    norms vary by orders of magnitude across tokens, so raw MSE would be
    dominated by a handful of positions.
    """
    pred, target = pred.float(), target.float()
    mse = (pred - target).pow(2).mean(-1) / target.pow(2).mean(-1).clamp_min(1e-6)
    cos = F.cosine_similarity(pred, target, dim=-1)
    per_tok = alpha * mse + beta * (1.0 - cos)
    if mask is None:
        mask = torch.ones_like(per_tok, dtype=torch.bool)
    n = mask.sum().clamp_min(1)
    loss = (per_tok * mask).sum() / n
    stats = {"rel_mse": ((mse * mask).sum() / n).item(), "cos": ((cos * mask).sum() / n).item()}
    return loss, stats


def distill_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    temperature: float = 2.0,
    kd_weight: float = 1.0,
    ce_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """kd_weight * T^2 * KL(target_T || draft_T) + ce_weight * CE(draft, labels).

    ``labels`` are aligned with the logits (label[t] is the token after
    position t); -100 is ignored. Also reports top-1 agreement with the target,
    which is the greedy acceptance rate at depth 1.
    """
    V = draft_logits.shape[-1]
    s = draft_logits.float().reshape(-1, V)
    t = target_logits.float().reshape(-1, V)
    T = temperature
    kd = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="batchmean") * T * T
    loss = kd_weight * kd
    stats = {"kd": kd.item()}
    if labels is not None and ce_weight > 0:
        ce = F.cross_entropy(s, labels.reshape(-1), ignore_index=-100)
        loss = loss + ce_weight * ce
        stats["ce"] = ce.item()
    stats["top1_agree"] = (s.argmax(-1) == t.argmax(-1)).float().mean().item()
    return loss, stats


def chunked_distill_loss(
    lm_head: torch.nn.Module,
    draft_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    labels: Optional[torch.Tensor],
    temperature: float = 2.0,
    kd_weight: float = 1.0,
    ce_weight: float = 0.1,
    chunk_tokens: int = 1024,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``distill_loss`` computed from final (post-norm) hidden states, applying
    ``lm_head`` one token chunk at a time under activation checkpointing.
    Only one chunk of [tokens, vocab] logits is alive at once, which matters
    with Llama-3's 128k vocabulary."""
    from torch.utils.checkpoint import checkpoint

    d = draft_hidden.reshape(-1, draft_hidden.shape[-1])
    t = target_hidden.reshape(-1, target_hidden.shape[-1]).to(d.device)
    y = labels.reshape(-1).to(d.device) if labels is not None else None
    T = temperature

    def chunk_fn(dc, tc, yc):
        s = lm_head(dc).float()
        with torch.no_grad():
            tl = lm_head(tc).float()
        kd = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(tl / T, -1), log_target=True, reduction="sum") * T * T
        ce = F.cross_entropy(s, yc, ignore_index=-100, reduction="sum") if yc is not None else s.new_zeros(())
        agree = (s.argmax(-1) == tl.argmax(-1)).sum()
        return kd, ce, agree

    n = d.shape[0]
    n_ce = (y != -100).sum().clamp_min(1) if y is not None else 1
    kd_sum = ce_sum = 0.0
    agree_sum = 0
    for i in range(0, n, chunk_tokens):
        sl = slice(i, i + chunk_tokens)
        kd, ce, agree = checkpoint(chunk_fn, d[sl], t[sl], y[sl] if y is not None else None, use_reentrant=False)
        kd_sum = kd_sum + kd
        ce_sum = ce_sum + ce
        agree_sum += agree.item()
    kd_mean = kd_sum / n
    loss = kd_weight * kd_mean
    stats = {"kd": kd_mean.item()}
    if y is not None and ce_weight > 0:
        ce_mean = ce_sum / n_ce
        loss = loss + ce_weight * ce_mean
        stats["ce"] = ce_mean.item()
    stats["top1_agree"] = agree_sum / n
    return loss, stats
