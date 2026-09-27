"""Online self-distillation: activation capture during cached generation, and a short run."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ssd.config import SSDConfig
from ssd.engine.speculative import TargetRunner
from ssd.models.target_wrapper import TargetWrapper
from ssd.training.train_online import _Streams, train_online


def tiny():
    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                    num_attention_heads=4, num_key_value_heads=2)
    ).eval()


class ChatTok:
    def apply_chat_template(self, msgs, add_generation_prompt=True, return_dict=True):
        return {"input_ids": [ord(c) % 128 for c in msgs[0]["content"]]}

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) % 128 for c in text[-12:]]}

    def convert_tokens_to_ids(self, token):
        return 2

    def decode(self, ids):
        return "".join(chr(33 + i % 90) for i in ids)


def test_stepwise_capture_matches_full_forward():
    base = tiny()
    layers = [0, 3, 5]
    target = TargetRunner(base)
    streams = _Streams(2, 32, 64, layers, torch.float32, "cpu")
    cache = target.new_cache(32)
    with torch.no_grad():
        ids = torch.randint(0, 128, (2, 5), generator=torch.Generator().manual_seed(0))
        out = target(ids, past_key_values=cache, output_layer_inputs=set(layers))
        streams.write(ids, out, torch.zeros_like(ids, dtype=torch.bool))
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(10):
            out = target(tok, past_key_values=cache, output_layer_inputs=set(layers))
            streams.write(tok, out, torch.ones_like(tok, dtype=torch.bool))
            tok = out.logits[:, -1].argmax(-1, keepdim=True)
        seq = streams.tokens[:, : streams.len]
        taps = TargetWrapper(base)(seq, boundaries=[*layers, 6], compute_logits=False).boundaries
    for i in layers:
        torch.testing.assert_close(streams.taps[i][:, : streams.len], taps[i], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(streams.final[:, : streams.len], base.model.norm(taps[6]), atol=1e-5, rtol=1e-4)


def test_online_run_slides_and_saves(tmp_path):
    base = tiny()
    cfg = SSDConfig()
    cfg.draft_layers = [0, 3, 5]
    cfg.training.output_dir = str(tmp_path)
    cfg.training.log_every = 2
    oc = cfg.training.online
    oc.prompt, oc.streams, oc.chunk, oc.context = "hello there", 4, 8, 8
    oc.max_context, oc.keep_on_slide, oc.micro_batch, oc.save_every = 40, 16, 2, 5
    path = train_online(cfg, base, ChatTok(), max_steps=24)  # 12 chunks x 2 micro-batches: slides twice
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["meta"]["step"] == 24 and ckpt["meta"]["stage"] == "online"


def test_online_reprompts_on_end_of_turn(tmp_path, caplog):
    import logging

    base = tiny()
    base.generation_config.eos_token_id = list(range(0, 128, 3))  # a third of the vocab ends the turn
    cfg = SSDConfig()
    cfg.draft_layers = [0, 3, 5]
    cfg.training.output_dir = str(tmp_path)
    cfg.training.log_every = 4
    oc = cfg.training.online
    oc.prompt, oc.streams, oc.chunk, oc.context = "hello there", 4, 8, 8
    oc.max_context, oc.keep_on_slide, oc.micro_batch = 40, 16, 2
    with caplog.at_level(logging.INFO, logger="ssd"):
        train_online(cfg, base, ChatTok(), max_steps=8)
    line = [r.getMessage() for r in caplog.records if "reprompts=" in r.getMessage()][-1]
    reprompts = float(line.split("reprompts=")[1].split()[0])
    masked = float(line.split("masked=")[1].split()[0])
    assert reprompts > 0 and 0 < masked < 1  # inserted prompt tokens are excluded from the loss
