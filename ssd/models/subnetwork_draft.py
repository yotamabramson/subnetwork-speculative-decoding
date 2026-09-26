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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.config import BridgeConfig, SSDConfig, validate_layer_indices
from ssd.engine.kv_cache import DraftKVCache
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
    past_key_values: Optional[DraftKVCache]
    bridge_states: dict[str, torch.Tensor] = field(default_factory=dict)  # bridge name -> output


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

        # Bridge placement: slot k's bridge (if any) runs right before layer
        # layer_indices[k]; "into_norm" runs before the final norm.
        self.bridge_specs: list[BridgeSpec] = []
        self._pre_layer_bridge: list[Optional[str]] = []
        have = 0  # boundary produced so far (embeddings = boundary 0)
        for idx in self.layer_indices:
            name = None
            if have != idx:
                name = f"into_{idx}"
                self.bridge_specs.append(BridgeSpec(name, have, idx))
            self._pre_layer_bridge.append(name)
            have = idx + 1
        self._pre_norm_bridge = None
        if have != self.num_base_layers:
            self._pre_norm_bridge = "into_norm"
            self.bridge_specs.append(BridgeSpec("into_norm", have, self.num_base_layers))

        self.bridges = nn.ModuleDict(
            {s.name: TransitionBridge.from_config(self.d_model, self.bridge_config) for s in self.bridge_specs}
        )
        self.place_bridges()

    @classmethod
    def from_config(cls, cfg: SSDConfig) -> tuple[nn.Module, "SubnetworkDraftModel"]:
        """Load the target model named in ``cfg`` and build its draft. Returns (base, draft)."""
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.base_model,
            dtype=getattr(torch, cfg.model.torch_dtype),
            device_map=cfg.model.device_map,
            attn_implementation=cfg.model.attn_implementation,
        )
        base.eval()
        return base, cls(base, cfg.model.layer_indices, cfg.bridge)

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

    def new_cache(self, max_length: int) -> DraftKVCache:
        return DraftKVCache(self.num_layers, max_length)

    @staticmethod
    def _build_attn_mask(
        bsz: int,
        q_len: int,
        past_len: int,
        tree_attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Boolean SDPA mask [B, 1, q, past+q] (True = attend), or None for pure causal."""
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
        cache: Optional[DraftKVCache],
    ) -> torch.Tensor:
        layer = self.base.layers[slot]
        attn = layer.self_attn
        bsz, q_len, _ = h.shape

        residual = h
        x = layer.input_layernorm(h)
        q = attn.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        if cache is not None:
            k, v = cache.update(slot, k, v)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        if attn_mask is None and q_len > 1:  # no cache, no tree: plain causal
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.head_dim)
        h = residual + attn.o_proj(out)

        h = h + layer.mlp(layer.post_attention_layernorm(h))
        return h

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[DraftKVCache] = None,
        tree_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        logits_to_keep: int = 0,
        compute_logits: bool = True,
        output_bridge_states: bool = False,
    ) -> DraftOutput:
        """Run the draft sub-network.

        Args:
            input_ids: [B, q] new tokens (or pass ``inputs_embeds``).
            position_ids: [B, q] absolute positions. Defaults to
                ``past_len + arange(q)``; tree drafting must pass depth-based positions.
            past_key_values: a ``DraftKVCache`` (see ``new_cache``). Updated in place
                and advanced by ``q``.
            tree_attention_mask: bool [q, q] or [B, q, q], True = query may attend to
                key, among the ``q`` new tokens. All cached tokens are always visible.
                None -> causal.
            logits_to_keep: if > 0, only compute logits for the last N positions
                (the 128k-vocab lm_head is a large share of draft cost).
            compute_logits: skip the lm_head entirely (e.g. Stage 1 feature losses).
            output_bridge_states: return each bridge's output (for Stage 1/2 losses).
        """
        if inputs_embeds is None:
            inputs_embeds = self.base.embed_tokens(input_ids)
        h = inputs_embeds
        bsz, q_len, _ = h.shape
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + q_len, device=h.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.base.rotary_emb(h, position_ids)

        if past_key_values is None and tree_attention_mask is None:
            attn_mask = None  # SDPA is_causal fast path
        else:
            attn_mask = self._build_attn_mask(bsz, q_len, past_len, tree_attention_mask, h.device)

        bridge_states: dict[str, torch.Tensor] = {}
        for slot, layer in enumerate(self.base.layers):
            name = self._pre_layer_bridge[slot]
            if name is not None:
                h = self.bridges[name](h)
                if output_bridge_states:
                    bridge_states[name] = h
            dev = _module_device(layer)
            h = h.to(dev)
            h = self._layer_forward(
                slot,
                h,
                cos.to(dev),
                sin.to(dev),
                attn_mask.to(dev) if attn_mask is not None else None,
                past_key_values,
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
            logits = self.base.lm_head(hs.to(_module_device(self.base.lm_head)))
        return DraftOutput(logits=logits, hidden_states=h, past_key_values=past_key_values, bridge_states=bridge_states)

    # ------------------------------------------------------------ persistence
    def save_bridges(self, path: str) -> None:
        torch.save(
            {
                "layer_indices": self.layer_indices,
                "bridge_config": vars(self.bridge_config),
                "state_dict": self.bridges.state_dict(),
            },
            path,
        )

    def load_bridges(self, path: str, strict: bool = True) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if list(ckpt["layer_indices"]) != self.layer_indices:
            raise ValueError(f"checkpoint layer_indices {ckpt['layer_indices']} != {self.layer_indices}")
        self.bridges.load_state_dict(ckpt["state_dict"], strict=strict)
