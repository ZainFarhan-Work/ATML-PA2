from __future__ import annotations

import argparse
import gc

import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sampled_kl
from common.models import (
    load_policy,
    load_reward_model_cpu,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from common.rl_utils import fixed_prompt_schedule, full_entropy, zero_dropout
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

STD_TOL = 1e-6  # a group is "informative" when its population reward std exceeds the helper's epsilon


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model_cpu(cfg)
    zero_dropout(policy)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, tokenizer, optimizer = bundle["policy"], bundle["tokenizer"], bundle["optimizer"]
    reward_model, reward_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    device = next(policy.parameters()).device
    K = int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    n_updates, per_update = int(cfg["updates"]), int(cfg["prompts_per_update"])
    schedule = fixed_prompt_schedule(cfg, tokenizer, bundle["prompt_rows"], n_updates * per_update)
    base_cfg = policy.get_base_model().config

    results_dir = cfg["results_dir"]
    log_path = repo_path(f"{results_dir}/train_log_{run_name}.jsonl")
    comp_path = repo_path(f"{results_dir}/completions_{run_name}.jsonl")
    log_path.unlink(missing_ok=True)
    comp_path.unlink(missing_ok=True)

    set_seed(int(cfg["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    history = []
    mean = lambda xs: sum(xs) / len(xs)

    for update in range(1, n_updates + 1):
        t_update = wall_timer()
        rows = schedule[(update - 1) * per_update: update * per_update]

        # ---- 1. sample K completions for every prompt of this update (adapter on, KV cache on) ----
        comps = []  # one dict per completion, flat over (prompt, k)
        policy.eval()
        for g, row in enumerate(rows):
            base_cfg.use_cache = True
            gen = batch_generate(
                policy, tokenizer, [row["messages"]] * K,
                max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=max_len, **cfg["generation"],
            )
            base_cfg.use_cache = False
            pw = gen["prompt_width"]  # batch_generate runs under inference_mode -> clone for autograd
            for k in range(K):
                n = int(gen["response_lengths"][k])
                comps.append({
                    "group": g, "row": row, "text": gen["responses"][k], "n": n,
                    "eos": bool(gen["terminated_with_eos"][k]), "truncated": bool(gen["truncated"][k]),
                    "seq": gen["sequences"][k:k + 1, : pw + n].clone(), "attn": gen["attention_mask"][k:k + 1, : pw + n].clone(),
                    "resp": gen["response_ids"][k:k + 1, :n].clone(), "pw": pw,
                })

        # ---- 2. rewards (CPU reward model) and group-relative advantages ----
        rewards = torch.tensor([
            score_reward_pairs(reward_model, reward_tokenizer, [c["row"]["messages"]], [c["text"]],
                               max_length=int(cfg.get("reward_max_length", 1280))).item()
            for c in comps
        ], device=device)
        group_ids = torch.tensor([c["group"] for c in comps], device=device)
        adv = group_relative_advantages(rewards, group_ids)

        # ---- 3. token mask (max-length completions masked out of the LOSS only) ----
        R = max(c["n"] for c in comps)
        token_mask = torch.zeros(len(comps), R, device=device)
        for i, c in enumerate(comps):
            token_mask[i, : c["n"]] = 1.0
        if mask_trunc:
            token_mask = mask_truncated_sequences(token_mask, [c["truncated"] for c in comps])

        # ---- 4. old / reference log-probs and entropy (no grad). Fully masked completions are skipped. ----
        old_lp = torch.zeros(len(comps), R, device=device)
        ref_lp = torch.zeros(len(comps), R, device=device)
        ent_sum, ent_n = 0.0, 0
        with torch.no_grad():
            for i, c in enumerate(comps):
                if token_mask[i].sum() == 0:
                    continue
                args = (c["seq"], c["attn"], c["pw"], c["resp"])
                lp, logits = response_token_logprobs(policy, *args)
                old_lp[i, : c["n"]] = lp[0]
                ent_sum += full_entropy(logits, torch.ones_like(lp)) * c["n"]
                ent_n += c["n"]
                del logits
                with reference_mode(policy):
                    ref_lp[i, : c["n"]] = response_token_logprobs(policy, *args)[0][0]
        torch.cuda.empty_cache()

        # ---- 5. clipped GRPO update (policy_epochs passes over this one batch) ----
        policy.train()
        stats = []
        # If every completion hit the length cap they are all masked out of the loss: there is nothing to
        # backpropagate, so the update is a no-op (recorded as skipped) instead of a crash.
        skipped = bool(token_mask.sum() == 0)
        for _ in range(0 if skipped else int(cfg["policy_epochs"])):
            rows_lp = []
            for i, c in enumerate(comps):
                if token_mask[i].sum() == 0:
                    rows_lp.append(torch.zeros(R, device=device))
                    continue
                lp, _ = response_token_logprobs(policy, c["seq"], c["attn"], c["pw"], c["resp"])
                rows_lp.append(torch.nn.functional.pad(lp[0], (0, R - c["n"])))
            new_lp = torch.stack(rows_lp)
            loss, diag = grpo_policy_loss(new_lp, old_lp, adv, token_mask, ref_lp, eps, beta, loss_type, max_len)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            optimizer.step()
            stats.append((loss.item(), diag["policy_term"].item(), diag["sampled_kl"].item(),
                          diag["clip_fraction"].item(), float(gn)))
            del new_lp, loss
        if skipped:
            stats = [(0.0, 0.0, 0.0, 0.0, 0.0)]
        torch.cuda.empty_cache()

        # ---- 6. logging ----
        r_g = rewards.view(len(rows), K)
        group_std = r_g.std(dim=1, unbiased=False)
        lengths = torch.tensor([c["n"] for c in comps], dtype=torch.float32)
        rec = {
            "update": update, "prompt_ids": [r["prompt_id"] for r in rows], "loss_type": loss_type, "kl_beta": beta,
            "reward": rewards.mean().item(), "reward_max": rewards.max().item(), "reward_min": rewards.min().item(),
            "group_reward_std": group_std.mean().item(),
            "uninformative_group_fraction": (group_std <= STD_TOL).float().mean().item(),
            "kl_ref": sampled_kl(old_lp, ref_lp, token_mask).item(),
            "kl_k3": mean([s[2] for s in stats]),
            "policy_loss": mean([s[0] for s in stats]), "policy_term": mean([s[1] for s in stats]),
            "clip_fraction": mean([s[3] for s in stats]), "grad_norm": mean([s[4] for s in stats]),
            "entropy": ent_sum / max(ent_n, 1),
            "response_length": lengths.mean().item(), "response_length_std": lengths.std(unbiased=False).item(),
            "update_skipped": skipped, "n_truncated": sum(c["truncated"] for c in comps), "n_loss_tokens": int(token_mask.sum().item()),
            "update_time_s": t_update(), "time_s": elapsed(), "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        history.append(rec)
        append_jsonl(log_path, rec)
        for i, c in enumerate(comps):
            append_jsonl(comp_path, {
                "update": update, "k": i, "prompt_id": c["row"]["prompt_id"], "prompt": c["row"]["messages"][-1]["content"],
                "response": c["text"], "length": c["n"], "reward": rewards[i].item(), "advantage": adv[i].item(),
                "truncated": c["truncated"], "masked": bool(token_mask[i].sum() == 0),
            })
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "prompt_ids"}, flush=True)

    policy.save_pretrained(str(out))
    summary = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "num_generations": K,
        "wall_time_s": elapsed(), "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "reward_first5": mean([h["reward"] for h in history[:5]]), "reward_last5": mean([h["reward"] for h in history[-5:]]),
        "kl_ref_last5": mean([h["kl_ref"] for h in history[-5:]]),
        "uninformative_group_fraction": mean([h["uninformative_group_fraction"] for h in history]),
        "skipped_updates": sum(h["update_skipped"] for h in history),
    }
    save_json(f"{results_dir}/train_summary_{run_name}.json", summary)
    print(summary)
    del bundle, policy, reward_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
