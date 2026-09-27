"""Draft model assembled from a frozen subset of the target model's layers.

    embed -> [bridge] -> layer a0 -> [bridge] -> layer a1 -> ... -> [bridge] -> norm -> lm_head

All base modules are *referenced*, never copied: they are held outside the
nn.Module registry, so ``parameters()``/``state_dict()`` contain only the
bridges and ``.to()`` on the draft never moves base weights.

Bridge placement uses "boundary" indices: boundary ``i`` is the residual
stream entering base layer ``i``; boundary ``L`` enters the final norm. A
bridge is inserted wherever the draft skips base layers, mapping the boundary
it has (``a_prev + 1``) to the one the next module expects. With the default
first/middle/last selection that is exactly one bridge per consecutive pair of
selected layers. These boundary indices are also the regression targets that
``TargetWrapper`` taps for Stage 1.

The decoder-layer forward is re-implemented on top of the layer's own
submodules (q/k/v/o_proj, norms, mlp) rather than calling
``LlamaDecoderLayer.forward``. That gives us arbitrary tree attention masks,
our own KV cache, and independence from HF's cache/mask API churn.

KV caching is keyed by *base* layer index, so at inference the draft shares
the target's cache. Its layers attend to the target's real keys/values for
every committed token and only add entries for the speculative tokens they
process. ``true_layer_inputs`` reproduces exactly that situation in training:
position t attends to the target's KV for positions < t and to its own KV at t.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.config import BridgeConfig, validate_layer_indices
from ssd.engine.kv_cache import KVCache
from ssd.models.bridges import TransitionBridge


@dataclass
class BridgeSpec:
    name: str  # key in SubnetworkDraftModel.bridges
    src_boundary: int  # boundary the bridge consumes (in the base model's teacher-forced run)
    tgt_boundary: int  # boundary the bridge must reproduce


@dataclass
class DraftOutput:
    logits: Optional[torch.Tensor]
    hidden_states: torch.Tensor  # final-norm output, [B, T, d]
    past_key_values: Optional[KVCache]
    bridge_states: dict[str, torch.Tensor] = field(default_factory=dict)  # bridge name -> output
    vocab_ids: Optional[torch.Tensor] = None  # set when logits cover a vocab subset: column j is token vocab_ids[j]
    layer_inputs: dict[int, torch.Tensor] = field(default_factory=dict)  # base layer -> its input (after any bridge)


def plan_bridges(layer_indices: list[int], num_base_layers: int) -> list[BridgeSpec]:
    """A bridge goes wherever the draft skips base layers: before each selected
    layer that doesn't directly follow the previous one, and before the final
    norm if the last base layer isn't selected."""
    specs: list[BridgeSpec] = []
    have = 0  # boundary produced so far (embeddings = boundary 0)
    for idx in layer_indices:
        if have != idx:
            specs.append(BridgeSpec(f"into_{idx}", have, idx))
        have = idx + 1
    if have != num_base_layers:
        specs.append(BridgeSpec("into_norm", have, num_base_layers))
    return specs


def required_boundaries(layer_indices: list[int], num_base_layers: int) -> set[int]:
    return {b for s in plan_bridges(layer_indices, num_base_layers) for b in (s.src_boundary, s.tgt_boundary)}


class _BaseRefs:
    """Plain holder so base modules are not registered as draft submodules."""

    def __init__(self, base_model: nn.Module, layer_indices: list[int]):
        inner = base_model.model
        self.embed_tokens = inner.embed_tokens
        self.layers = [inner.layers[i] for i in layer_indices]
        self.norm = inner.norm
        self.rotary_emb = inner.rotary_emb
        self.lm_head = base_model.lm_head


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _module_device(m: nn.Module) -> torch.device:
    return next(m.parameters()).device


class SubnetworkDraftModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        layer_indices: list[int],
        bridge_config: Optional[BridgeConfig] = None,
        freeze_base: bool = True,
    ):
        super().__init__()
        cfg = base_model.config
        self.num_base_layers = cfg.num_hidden_layers
        self.layer_indices = validate_layer_indices(layer_indices, self.num_base_layers)
        self.d_model = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or self.d_model // self.num_heads
        self.bridge_config = bridge_config or BridgeConfig()

        if freeze_base:
            base_model.requires_grad_(False)
        self.base = _BaseRefs(base_model, self.layer_indices)

        self.bridge_specs = plan_bridges(self.layer_indices, self.num_base_layers)
        by_target = {s.tgt_boundary: s.name for s in self.bridge_specs}
        # Bridge (if any) that runs right before each selected layer / the final norm.
        self._pre_layer_bridge = [by_target.get(idx) for idx in self.layer_indices]
        self._pre_norm_bridge = by_target.get(self.num_base_layers)

        self.bridges = nn.ModuleDict(
            {s.name: TransitionBridge.from_config(self.d_model, self.bridge_config) for s in self.bridge_specs}
        )
        self.place_bridges()
        self.vocab_ids: Optional[torch.Tensor] = None
        self._sub_head: Optional[torch.Tensor] = None

    def set_vocab_subset(self, token_ids: Optional[torch.Tensor]) -> None:
        """Score only ``token_ids`` (e.g. the most frequent tokens) in the draft's
        lm_head. Keeps a sliced copy of the lm_head rows (|subset| x d)."""
        if token_ids is None:
            self.vocab_ids = self._sub_head = None
            return
        w = self.base.lm_head.weight
        self.vocab_ids = token_ids.to(w.device)
        self._sub_head = w.detach().index_select(0, self.vocab_ids).contiguous()

    def place_bridges(self) -> None:
        """Put each bridge on the device of the base module that consumes its output."""
        for spec in self.bridge_specs:
            if spec.tgt_boundary == self.num_base_layers:
                consumer = self.base.norm
            else:
                consumer = self.base.layers[self.layer_indices.index(spec.tgt_boundary)]
            self.bridges[spec.name].to(_module_device(consumer))

    # ------------------------------------------------------------------ utils
    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    def new_cache(self, max_length: int) -> KVCache:
        """Cache with a slot per *base* layer (only the selected ones get allocated),
        so the same cache can be shared with the full target."""
        return KVCache(self.num_base_layers, max_length)

    @staticmethod
    def _build_attn_mask(
        bsz: int,
        q_len: int,
        past_len: int,
        tree_attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Boolean SDPA mask [B, 1, q, past+q] (True = attend), or None for pure causal."""
        if tree_attention_mask is not None and tree_attention_mask.shape[-1] == past_len + q_len and past_len > 0:
            mask = tree_attention_mask.to(device=device, dtype=torch.bool)
            if mask.dim() == 2:
                mask = mask.expand(bsz, q_len, past_len + q_len)
            return mask.unsqueeze(1)
        if tree_attention_mask is None:
            if q_len == 1:
                return None
            causal = torch.ones(q_len, q_len, dtype=torch.bool, device=device).tril()
        else:
            causal = tree_attention_mask.to(device=device, dtype=torch.bool)
            if causal.shape[-2:] != (q_len, q_len):
                raise ValueError(f"tree_attention_mask must be [.., {q_len}, {q_len}], got {tuple(causal.shape)}")
        prefix = torch.ones(*causal.shape[:-1], past_len, dtype=torch.bool, device=device)
        mask = torch.cat([prefix, causal], dim=-1)
        if mask.dim() == 2:
            mask = mask.expand(bsz, q_len, past_len + q_len)
        return mask.unsqueeze(1)

    def _layer_forward(
        self,
        slot: int,
        h: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        cache: Optional[KVCache],
        true_input: Optional[torch.Tensor] = None,
        true_rope: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        layer = self.base.layers[slot]
        attn = layer.self_attn
        bsz, q_len, _ = h.shape
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

        def kv(x, cos=cos, sin=sin):
            n = x.shape[1]
            k = attn.k_proj(x).view(bsz, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(x).view(bsz, n, self.num_kv_heads, self.head_dim).transpose(1, 2)
            return k * cos + _rotate_half(k) * sin, v

        residual = h
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k, v = kv(x)

        if cache is not None:
            k, v = cache.update(self.layer_indices[slot], k, v)
        elif true_input is not None:
            # Keys: target's KV for positions < t, then the draft's own KV at t. The
            # target inputs may cover a longer window than the queries (context-only
            # prefix): queries are its last q_len positions.
            t_len = true_input.shape[1]
            t_cos, t_sin = (c.unsqueeze(1) for c in true_rope) if true_rope is not None else (cos, sin)
            k_t, v_t = kv(layer.input_layernorm(true_input.to(h.device, h.dtype)), t_cos, t_sin)
            k, v = torch.cat([k_t, k], dim=2), torch.cat([v_t, v], dim=2)
            q_abs = torch.arange(t_len - q_len, t_len, device=h.device)
            attn_mask = torch.cat(
                [torch.arange(t_len, device=h.device)[None, :] < q_abs[:, None],
                 torch.eye(q_len, dtype=torch.bool, device=h.device)], 1)

        # GQA without materialising repeated K/V (which would copy the whole cache per call).
        gqa = self.num_heads != self.num_kv_heads
        if attn_mask is None and q_len > 1:  # no cache, no tree: plain causal
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=gqa)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        h = residual + attn.o_proj(out)

        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
        compute_logits: bool = True,
        output_bridge_states: bool = False,
        true_layer_inputs: Optional[dict[int, torch.Tensor]] = None,
        output_layer_inputs: Optional[set[int]] = None,
    ) -> DraftOutput:
        """Run the draft sub-network.

        Args:
            input_ids: [B, q] new tokens (or pass ``inputs_embeds``).
            position_ids: [B, q] absolute positions. Defaults to
                ``past_len + arange(q)``; tree drafting must pass depth-based positions.
            past_key_values: a ``KVCache`` (see ``new_cache``). Updated in place
                and advanced by ``q``.
            tree_attention_mask: bool, True = query may attend to key. Either
                [.., q, q] over the new tokens only (every cached token visible), or
                [.., q, past+q] over cache + new tokens (lets tree nodes see only
                their own ancestors among earlier cached nodes). None -> causal.
            logits_to_keep: if > 0, only compute logits for the last N positions
                (the 128k-vocab lm_head is a large share of draft cost).
            compute_logits: skip the lm_head entirely (e.g. Stage 1 feature losses).
            output_bridge_states: return each bridge's output (for Stage 1/2 losses).
            true_layer_inputs: training only (no cache). ``{base_layer: [B, T, d]}``,
                the target's own input to each selected layer (``TargetWrapper``
                boundary taps). Position t then attends to keys/values computed from
                these for positions < t, exactly as at inference with a shared cache.
                T may exceed the number of new tokens: the extra leading positions are
                context only (keys/values, no draft computation or outputs).
            output_layer_inputs: base layer indices whose input hidden state to return
                in ``layer_inputs`` (with all layers selected, these are the target's
                own boundary activations, i.e. what ``true_layer_inputs`` expects).
        """
        if true_layer_inputs is not None and past_key_values is not None:
            raise ValueError("true_layer_inputs is for training without a cache")
        if inputs_embeds is None:
            inputs_embeds = self.base.embed_tokens(input_ids)
        h = inputs_embeds
        bsz, q_len, _ = h.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + q_len, device=h.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.base.rotary_emb(h, position_ids)
        true_cos = true_sin = None
        if true_layer_inputs:
            t_len = next(iter(true_layer_inputs.values())).shape[1]
            if t_len < q_len:
                raise ValueError("true_layer_inputs must cover at least the query positions")
            if t_len > q_len:  # context-only prefix: rope for its key positions too
                t_pos = position_ids[:, :1] - (t_len - q_len) + torch.arange(t_len, device=h.device)
                true_cos, true_sin = self.base.rotary_emb(h, t_pos)

        if past_key_values is None and tree_attention_mask is None:
            attn_mask = None  # SDPA is_causal fast path
        else:
            attn_mask = self._build_attn_mask(bsz, q_len, past_len, tree_attention_mask, h.device)

        bridge_states: dict[str, torch.Tensor] = {}
        layer_inputs: dict[int, torch.Tensor] = {}
        for slot, layer in enumerate(self.base.layers):
            name = self._pre_layer_bridge[slot]
            if name is not None:
                h = self.bridges[name](h)
                if output_bridge_states:
                    bridge_states[name] = h
            dev = _module_device(layer)
            h = h.to(dev)
            if output_layer_inputs and self.layer_indices[slot] in output_layer_inputs:
                layer_inputs[self.layer_indices[slot]] = h
            h = self._layer_forward(
                slot,
                h,
                cos.to(dev),
                sin.to(dev),
                attn_mask.to(dev) if attn_mask is not None else None,
                past_key_values,
                true_layer_inputs[self.layer_indices[slot]] if true_layer_inputs is not None else None,
                (true_cos.to(dev), true_sin.to(dev)) if true_cos is not None else None,
            )

        if self._pre_norm_bridge is not None:
            h = self.bridges[self._pre_norm_bridge](h)
            if output_bridge_states:
                bridge_states[self._pre_norm_bridge] = h

        if past_key_values is not None:
            past_key_values.advance(q_len)

        h = self.base.norm(h.to(_module_device(self.base.norm)))
        logits = None
        if compute_logits:
            hs = h[:, -logits_to_keep:] if logits_to_keep > 0 else h
            hs = hs.to(_module_device(self.base.lm_head))
            logits = F.linear(hs, self._sub_head) if self._sub_head is not None else self.base.lm_head(hs)
        return DraftOutput(
            logits=logits,
            hidden_states=h,
            past_key_values=past_key_values,
            bridge_states=bridge_states,
            vocab_ids=self.vocab_ids if compute_logits else None,
            layer_inputs=layer_inputs,
        )

    # ------------------------------------------------------------ persistence
    def save_bridges(self, path: str, **meta) -> None:
        """Save bridge weights plus ``meta`` (e.g. stage, step) to ``path``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "layer_indices": self.layer_indices,
                "bridge_config": vars(self.bridge_config),
                "state_dict": self.bridges.state_dict(),
                "meta": meta,
            },
            path,
        )

    def load_bridges(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if list(ckpt["layer_indices"]) != self.layer_indices:
            raise ValueError(f"checkpoint layer_indices {ckpt['layer_indices']} != {self.layer_indices}")
        self.bridges.load_state_dict(ckpt["state_dict"], strict=strict)
