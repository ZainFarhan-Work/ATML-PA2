from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model_cpu,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl_utils import fixed_prompt_schedule, full_entropy, zero_dropout
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def prepare_critic(value_model):
    """Keep trainable critic parameters in fp32 (fp16 AdamW underflows) and feed the head fp32 input."""
    for p in value_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    for m in value_model.modules():
        if m.__class__.__name__ == "ModulesToSaveWrapper":
            m.register_forward_pre_hook(lambda mod, args: (args[0].float(), *args[1:]))


def critic_diagnostics(values, returns, mask):
    v, r = values[mask.bool()], returns[mask.bool()]
    var_r = r.var(unbiased=False)
    ev = (1.0 - (r - v).var(unbiased=False) / var_r).item() if var_r > 1e-8 else float("nan")
    corr = torch.corrcoef(torch.stack([v, r]))[0, 1].item() if v.numel() > 1 else float("nan")
    return ev, corr


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    prepare_critic(value_model)
    zero_dropout(policy, value_model)
    value_model.to("cpu")  # only resident on the GPU during the critic phases (6 GB budget)
    reward_model, reward_tokenizer = load_reward_model_cpu(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, value_model = bundle["policy"], bundle["value_model"]
    tokenizer = bundle["tokenizer"]
    reward_model, reward_tokenizer = bundle["reward_model"], bundle["reward_tokenizer"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    device = next(policy.parameters()).device
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_gn = float(cfg["max_grad_norm"])
    n_updates, ppo_epochs = int(cfg["updates"]), int(cfg["ppo_epochs"])
    per_update = int(cfg["prompts_per_update"])
    schedule = fixed_prompt_schedule(cfg, tokenizer, bundle["prompt_rows"], n_updates * per_update)
    base_cfg = policy.get_base_model().config

    results_dir = Path(cfg["results_dir"])
    log_path = repo_path(results_dir / f"train_log_{run_name}.jsonl")
    roll_path = repo_path(results_dir / f"rollouts_{run_name}.jsonl")
    log_path.unlink(missing_ok=True)
    roll_path.unlink(missing_ok=True)

    set_seed(int(cfg["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    history = []
    mean = lambda xs: sum(xs) / len(xs)

    for update in range(1, n_updates + 1):
        t_update = wall_timer()
        row = schedule[update - 1]
        messages = row["messages"]

        # ---- 1. on-policy rollout (adapter on, KV cache on) ----
        base_cfg.use_cache = True
        gen = batch_generate(
            policy, tokenizer, [messages],
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            **cfg["generation"],
        )
        base_cfg.use_cache = False
        # batch_generate runs under inference_mode; clone so these tensors can enter autograd later.
        seq, attn = gen["sequences"].clone(), gen["attention_mask"].clone()
        pw, resp_ids = gen["prompt_width"], gen["response_ids"].clone()
        mask = gen["response_mask"].to(device).clone()
        n_tok = int(gen["response_lengths"][0])
        lp_args = (seq, attn, pw, resp_ids)

        # ---- 2. old / reference log-probs, entropy (no grad, temperature-1 distribution) ----
        policy.eval()
        with torch.no_grad():
            old_lp, logits = response_token_logprobs(policy, *lp_args)
            entropy = full_entropy(logits, mask)
            del logits
            with reference_mode(policy):
                ref_lp, _ = response_token_logprobs(policy, *lp_args)
        kl_ref = sampled_kl(old_lp, ref_lp, mask).item()
        sampled_ent = sample_entropy(old_lp, mask).item()
        torch.cuda.empty_cache()

        # ---- 3. learned reward (CPU reward model) with the missing-EOS penalty ----
        raw_reward = score_reward_pairs(
            reward_model, reward_tokenizer, [messages], [gen["responses"][0]],
            max_length=int(cfg["reward_max_length"]),
        ).item()
        eos = bool(gen["terminated_with_eos"][0])
        reward = raw_reward - (0.0 if eos else float(cfg["missing_eos_penalty"]))

        # ---- 4. critic values (value of state s_t sits at the hidden state of position P-1+t) ----
        value_model.to(device).eval()
        with torch.no_grad():
            values = token_values(value_model, seq, attn).float()[:, pw - 1: pw - 1 + resp_ids.shape[1]]
        value_model.to("cpu")
        torch.cuda.empty_cache()

        # ---- 5. KL-shaped rewards, GAE, returns ----
        rewards = shaped_rewards(torch.tensor([reward], device=device), old_lp, ref_lp, mask, beta)
        adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        adv_n = normalize_advantages(adv, mask)
        ev, corr = critic_diagnostics(values, returns, mask)

        # ---- 6. PPO epochs: policy (critic stays on CPU) ----
        policy.train()
        pol_stats = []
        for _ in range(ppo_epochs):
            new_lp, _ = response_token_logprobs(policy, *lp_args)
            p_loss, ratio, clip_frac = ppo_policy_loss(new_lp, old_lp, adv_n, mask, eps)
            pol_opt.zero_grad(set_to_none=True)
            p_loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_gn)
            pol_opt.step()
            r_valid = ratio[mask.bool()]
            pol_stats.append((p_loss.item(), clip_frac.item(), float(gn), r_valid.max().item(), r_valid.min().item()))
            del new_lp, p_loss
        torch.cuda.empty_cache()

        # ---- 7. PPO epochs: critic (policy forward not needed here) ----
        value_model.to(device).train()
        val_stats = []
        for _ in range(ppo_epochs):
            pred = token_values(value_model, seq, attn).float()[:, pw - 1: pw - 1 + resp_ids.shape[1]]
            mse = value_mse_loss(pred, returns, mask)
            val_opt.zero_grad(set_to_none=True)
            (float(cfg["value_coef"]) * mse).backward()
            vgn = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_gn)
            val_opt.step()
            val_stats.append((mse.item(), float(vgn)))
            del pred, mse
        value_model.to("cpu")
        torch.cuda.empty_cache()

        rec = {
            "update": update, "prompt_id": row["prompt_id"], "clip_epsilon": eps, "kl_beta": beta,
            "reward_raw": raw_reward, "reward": reward, "terminated_with_eos": eos, "response_length": n_tok,
            "kl_ref": kl_ref, "entropy": entropy, "entropy_sampled": sampled_ent,
            "policy_loss": mean([s[0] for s in pol_stats]), "clip_fraction": mean([s[1] for s in pol_stats]),
            "grad_norm": mean([s[2] for s in pol_stats]),
            "ratio_max": max(s[3] for s in pol_stats), "ratio_min": min(s[4] for s in pol_stats),
            "value_loss": mean([s[0] for s in val_stats]), "value_grad_norm": mean([s[1] for s in val_stats]),
            "critic_explained_var": ev, "critic_corr": corr,
            "value_mean": values[mask.bool()].mean().item(), "return_mean": returns[mask.bool()].mean().item(),
            "update_time_s": t_update(), "time_s": elapsed(),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
        history.append(rec)
        append_jsonl(log_path, rec)
        append_jsonl(roll_path, {"update": update, "prompt": messages[-1]["content"],
                                 "response": gen["responses"][0], "reward_raw": raw_reward, "reward": reward})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "value"))
    summary = {
        "run_name": run_name, "updates": n_updates, "clip_epsilon": eps, "kl_beta": beta,
        "wall_time_s": elapsed(), "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_vram_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "reward_first5": mean([h["reward"] for h in history[:5]]), "reward_last5": mean([h["reward"] for h in history[-5:]]),
        "kl_ref_last5": mean([h["kl_ref"] for h in history[-5:]]),
    }
    save_json(results_dir / f"train_summary_{run_name}.json", summary)
    print(summary)
    del bundle, policy, value_model, reward_model, pol_opt, val_opt
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
