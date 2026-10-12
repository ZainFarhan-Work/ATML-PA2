from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model_cpu, load_tokenizer, reference_mode
from common.rl_utils import full_entropy

# The held-out prompts include some long ones (up to ~1.9k tokens); the training-time 256-token
# prompt cap is not applied here so that the fixed course evaluation set is used unchanged.
EVAL_MAX_PROMPT_TOKENS = 2048


def load_evaluation_bundle(config_path: str, adapter: str, max_prompts: int | None = None, overrides: dict | None = None):
    cfg = load_yaml(config_path)
    cfg.update(overrides or {})
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_prompts:
        rows = rows[: int(max_prompts)]
    # adapter == "none" evaluates the untouched base model.
    adapter_path = None if adapter.lower() in ("none", "base") else adapter
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter_path, trainable=False),
    }


def generate_all(policy, tokenizer, rows, cfg, batch_size, sample):
    """One response per prompt. Prompts are batched by length to limit padding; output keeps input order."""
    lengths = [len(tokenizer.apply_chat_template(r["messages"], tokenize=True, add_generation_prompt=True)) for r in rows]
    order = np.argsort(lengths)
    results = [None] * len(rows)
    gen_kwargs = dict(cfg["generation"]) if sample else {"do_sample": False}
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        gen = batch_generate(
            policy, tokenizer, [rows[i]["messages"] for i in idx],
            max_prompt_length=EVAL_MAX_PROMPT_TOKENS,
            max_new_tokens=int(cfg["eval_max_response_length"]),
            **gen_kwargs,
        )
        pw = gen["prompt_width"]
        for j, i in enumerate(idx):
            n = int(gen["response_lengths"][j])
            results[i] = {
                "text": gen["responses"][j], "length": n,
                "eos": bool(gen["terminated_with_eos"][j]),
                # sequence trimmed to prompt + n response tokens (left padding is kept) for teacher forcing
                "seq": gen["sequences"][j:j + 1, : pw + n].clone(),
                "attn": gen["attention_mask"][j:j + 1, : pw + n].clone(),
                "resp": gen["response_ids"][j:j + 1, :n].clone(),
                "pw": pw,
            }
        print(f"generated {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return results


@torch.no_grad()
def drift_and_entropy(policy, results):
    """Teacher-forced policy/reference log-probs on the generated responses -> KL estimate and entropy."""
    pol_all, ref_all, seq_kl, ent_sum, ent_n = [], [], [], 0.0, 0
    for r in results:
        args = (r["seq"], r["attn"], r["pw"], r["resp"])
        policy.eval()
        pol, logits = response_token_logprobs(policy, *args)
        mask = torch.ones_like(pol)
        ent = full_entropy(logits, mask)
        del logits
        with reference_mode(policy):
            ref, _ = response_token_logprobs(policy, *args)
        pol_all.append(pol[0].cpu())
        ref_all.append(ref[0].cpu())
        seq_kl.append((pol - ref).sum().item())
        r["entropy"] = ent
        ent_sum += ent * r["length"]
        ent_n += r["length"]
    kl_token = sampled_kl(torch.cat(pol_all), torch.cat(ref_all), torch.ones(sum(len(x) for x in pol_all)))
    return float(kl_token), seq_kl, ent_sum / max(ent_n, 1)


def evaluate(config_path, adapter, name, max_prompts=None, gen_batch_size=4, sample=False, overrides=None):
    """`overrides` lets another task reuse this exact protocol with its own results_dir / config keys."""
    t0 = time.perf_counter()
    bundle = load_evaluation_bundle(config_path, adapter, max_prompts, overrides)
    cfg, rows, tokenizer, policy = bundle["cfg"], bundle["rows"], bundle["tokenizer"], bundle["policy"]
    set_seed(int(cfg["seed"]))
    results = generate_all(policy, tokenizer, rows, cfg, gen_batch_size, sample)
    t_gen = time.perf_counter() - t0
    kl_token, seq_kl, entropy = drift_and_entropy(policy, results)

    del policy, bundle
    gc.collect()
    torch.cuda.empty_cache()

    rm, rm_tok = load_reward_model_cpu(cfg)
    penalty = float(cfg["missing_eos_penalty"])
    rewards = []
    for i, (row, r) in enumerate(zip(rows, results)):
        s = score_reward_pairs(rm, rm_tok, [row["messages"]], [r["text"]], max_length=int(cfg["reward_max_length"])).item()
        rewards.append(s)
        if (i + 1) % 20 == 0:
            print(f"scored {i + 1}/{len(rows)}", flush=True)
    effective = [s - (0.0 if r["eos"] else penalty) for s, r in zip(rewards, results)]

    L = np.array([r["length"] for r in results], dtype=float)
    cap = int(cfg["eval_max_response_length"])
    summary = {
        "name": name, "adapter": adapter, "n_prompts": len(rows),
        "decoding": "sampling (config)" if sample else "greedy", "max_new_tokens": cap,
        "reward_mean": float(np.mean(rewards)), "reward_std": float(np.std(rewards)),
        "reward_sem": float(np.std(rewards, ddof=1) / np.sqrt(len(rewards))),
        "reward_effective_mean": float(np.mean(effective)),
        "length_mean": float(L.mean()), "length_std": float(L.std()), "length_median": float(np.median(L)),
        "eos_rate": float(np.mean([r["eos"] for r in results])),
        "truncation_rate": float(np.mean([not r["eos"] for r in results])),
        "kl_token_mean": kl_token, "kl_sequence_sum_mean": float(np.mean(seq_kl)),
        "entropy_mean": entropy,
        "wall_time_s": time.perf_counter() - t0, "generation_time_s": t_gen,
    }
    out_dir = cfg["results_dir"]
    save_json(f"{out_dir}/eval_{name}.json", summary)
    gen_path = repo_path(f"{out_dir}/eval_{name}_generations.jsonl")
    gen_path.unlink(missing_ok=True)
    for row, r, s, e, k in zip(rows, results, rewards, effective, seq_kl):
        append_jsonl(gen_path, {
            "prompt_id": row["prompt_id"], "prompt": row["messages"][-1]["content"], "response": r["text"],
            "length": r["length"], "eos": r["eos"], "reward": s, "reward_effective": e,
            "seq_kl": k, "entropy": r["entropy"],
        })
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base model")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, help="use only the first N held-out prompts (keep identical across compared runs)")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--sample", action="store_true", help="sample with the config decoding instead of greedy")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.max_prompts, args.gen_batch_size, args.sample)


if __name__ == "__main__":
    main()
