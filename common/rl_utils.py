"""RL helpers shared by the PPO (Task 2) and GRPO (Task 3) continuations."""
from __future__ import annotations

import random

import torch


def zero_dropout(*models):
    """Disable LoRA dropout so the PPO ratio is exactly 1 at the first epoch of every update."""
    for model in models:
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0


def fixed_prompt_schedule(cfg, tokenizer, rows, n):
    """Deterministic prompt order shared by every run and fork (a fork uses a prefix of it).

    Prompts longer than max_prompt_length are skipped: generation truncates from the right, which
    would cut off the assistant turn marker.
    """
    limit = int(cfg["max_prompt_length"])
    ok = [
        r for r in rows
        if len(tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True)) <= limit
    ]
    random.Random(int(cfg["seed"])).shuffle(ok)
    print(f"prompt pool: {len(ok)}/{len(rows)} fit in {limit} tokens; using first {n}")
    return ok[:n]


@torch.no_grad()
def full_entropy(logits, mask, chunk=128):
    """Mean token entropy of the full next-token distribution (not the sampled-token estimate)."""
    logits, mask = logits[0], mask[0]
    total = torch.zeros((), device=logits.device)
    for i in range(0, logits.shape[0], chunk):
        lp = torch.log_softmax(logits[i:i + chunk].float(), dim=-1)
        total += ((-(lp.exp() * lp).sum(-1)) * mask[i:i + chunk]).sum()
    return (total / mask.sum().clamp_min(1.0)).item()
