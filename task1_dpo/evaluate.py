from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import filter_long_prompts, make_collate


def load_evaluation_bundle(config_path: str, adapter: str, eval_path: str | None = None):
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = read_jsonl(eval_path or cfg["paths"]["dpo_standard_eval"])
    rows = filter_long_prompts(rows, tokenizer, int(cfg["max_sequence_length"]))
    # adapter == "none" evaluates the untouched base model (the SFT baseline).
    adapter_path = None if adapter.lower() in ("none", "base") else adapter
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "policy": load_policy(cfg, adapter_path=adapter_path, trainable=False),
    }


@torch.no_grad()
def preference_eval(policy, rows, tokenizer, cfg, beta):
    """Held-out DPO loss and preference accuracy (teacher-forced, no generation)."""
    loader = DataLoader(rows, batch_size=1, shuffle=False,
                        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])))
    device = next(policy.parameters()).device
    margins, losses = [], []
    for chosen, rejected in loader:
        chosen = {k: v.to(device) for k, v in chosen.items()}
        rejected = {k: v.to(device) for k, v in rejected.items()}
        with reference_mode(policy):
            ref_c, _, _ = response_sequence_logprobs(policy, chosen)
            ref_r, _, _ = response_sequence_logprobs(policy, rejected)
        pol_c, _, _ = response_sequence_logprobs(policy, chosen)
        pol_r, _, _ = response_sequence_logprobs(policy, rejected)
        loss, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
        m = (pol_c - ref_c) - (pol_r - ref_r)  # manual's preference margin m_theta
        margins.append(m.item())
        losses.append(loss.item())
    margins = np.array(margins)
    return {
        "dpo_loss": float(np.mean(losses)),
        "preference_accuracy": float((margins > 0).mean()),  # m_theta > 0
        "margin_mean": float(margins.mean()),
        "n_pairs": len(margins),
    }, margins


@torch.no_grad()
def generate_and_kl(policy, tokenizer, prompts, cfg, batch_size):
    """Sample one response per prompt; also record per-token log-probs under policy and reference."""
    set_seed(int(cfg["seed"]))
    texts, lengths, truncated = [], [], []
    pol_all, ref_all, mask_all, seq_kl = [], [], [], []
    for i in range(0, len(prompts), batch_size):
        out = batch_generate(
            policy, tokenizer, prompts[i:i + batch_size],
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            **cfg["generation"],
        )
        args = (out["sequences"], out["attention_mask"], out["prompt_width"], out["response_ids"])
        pol, _ = response_token_logprobs(policy, *args)
        with reference_mode(policy):
            ref, _ = response_token_logprobs(policy, *args)
        mask = out["response_mask"].to(pol.device)
        seq_kl.extend(((pol - ref) * mask).sum(-1).tolist())  # per-sequence summed KL estimate
        pol_all.append(pol[mask.bool()].cpu())
        ref_all.append(ref[mask.bool()].cpu())
        mask_all.append(torch.ones(int(mask.sum().item())))
        texts += out["responses"]
        lengths += out["response_lengths"]
        truncated += out["truncated"]
        print(f"generated {len(texts)}/{len(prompts)}", flush=True)
    kl_token = sampled_kl(torch.cat(pol_all), torch.cat(ref_all), torch.cat(mask_all))
    return texts, lengths, truncated, float(kl_token), seq_kl


def score_with_reward_model(cfg, prompts, texts, batch_size=4):
    rm, rm_tok = load_reward_model(cfg)
    scores = []
    for i in range(0, len(texts), batch_size):
        s = score_reward_pairs(rm, rm_tok, prompts[i:i + batch_size], texts[i:i + batch_size])
        scores += s.cpu().tolist()
    return scores


def evaluate(config_path, adapter, name, beta=None, eval_path=None, gen_examples=None, gen_batch_size=2):
    bundle = load_evaluation_bundle(config_path, adapter, eval_path)
    cfg, rows, tokenizer, policy = bundle["cfg"], bundle["rows"], bundle["tokenizer"], bundle["policy"]
    beta = float(cfg["beta"] if beta is None else beta)
    out_dir = Path(cfg["results_dir"])

    pref, margins = preference_eval(policy, rows, tokenizer, cfg, beta)
    print(pref, flush=True)

    gen_rows = rows[: int(gen_examples)] if gen_examples else rows
    prompts = [prompt_messages_from_preference(r) for r in gen_rows]
    texts, lengths, truncated, kl_token, seq_kl = generate_and_kl(policy, tokenizer, prompts, cfg, gen_batch_size)

    # Free the policy before loading the reward model (6 GB GPU).
    del policy, bundle
    gc.collect()
    torch.cuda.empty_cache()
    rewards = score_with_reward_model(cfg, prompts, texts)

    L = np.array(lengths, dtype=float)
    summary = {
        "name": name, "adapter": adapter, "beta": beta, **pref,
        "n_generated": len(texts),
        "kl_token_mean": kl_token,                # course sampled_kl, pooled over all response tokens
        "kl_sequence_sum_mean": float(np.mean(seq_kl)),  # same estimator, summed per response then averaged
        "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
        "length_mean": float(L.mean()), "length_std": float(L.std()),
        "length_iqr": float(np.percentile(L, 75) - np.percentile(L, 25)),
        "frac_hit_cap_no_eos": float(np.mean(truncated)),
    }
    save_json(out_dir / f"eval_{name}.json", summary)

    gen_path = repo_path(out_dir / f"eval_{name}_generations.jsonl")
    gen_path.unlink(missing_ok=True)
    for i, (row, text, n, r, k) in enumerate(zip(gen_rows, texts, lengths, rewards, seq_kl)):
        append_jsonl(gen_path, {
            "idx": i, "prompt_id": row.get("prompt_id"), "prompt": prompt_messages_from_preference(row),
            "response": text, "length": n, "reward": r, "seq_kl": k,
        })
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base model")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the reported DPO loss (default: config)")
    ap.add_argument("--eval-path", help="override held-out pair file")
    ap.add_argument("--gen-examples", type=int, help="generate for only the first N held-out prompts")
    ap.add_argument("--gen-batch-size", type=int, default=2)
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.beta, args.eval_path, args.gen_examples, args.gen_batch_size)


if __name__ == "__main__":
    main()
