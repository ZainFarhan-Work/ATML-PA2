from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json
from common.models import load_policy, load_tokenizer, trainable_parameters
from common.rl_utils import zero_dropout
from task3_grpo.grpo import grpo_policy_loss

# (tag, loss_type). The third fork repeats the canonical configuration: it measures run-to-run noise so that a
# canonical-vs-Dr.GRPO difference can be compared against it.
FORKS = [("norm_grpo", "grpo"), ("norm_dr_grpo", "dr_grpo"), ("norm_grpo_rep", "grpo")]


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------------
# Length-conditioned gradient statistic
# --------------------------------------------------------------------------------------------------------------
def length_gradient_stats(config_path):
    """Per-completion gradient norm of the policy term under each normalization, binned by completion length.

    Uses the cached midpoint completions (lengths 2..~770), each with the SAME advantage A=1 and beta=0, so the
    only thing that differs between completions is its length. With ratio = 1 the per-token gradient is A*grad
    log p, so canonical GRPO scales a completion's gradient by 1/n and Dr.GRPO by 1/L_max (a constant).
    """
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    zero_dropout(policy)
    policy.eval()  # no dropout; gradient checkpointing still needs train mode, so toggle only where required
    policy.train()
    params = trainable_parameters(policy)
    L = int(cfg["max_completion_length"])

    pool = {r["prompt_id"]: r for f in ("rl_prompt_train", "rl_prompt_eval") for r in read_jsonl(cfg["paths"][f])}
    cache = read_jsonl(cfg["group_cache"])
    rows = []
    for c in cache:
        row = pool.get(c["prompt_id"])
        if row is None:
            continue
        prompt_ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
        resp_ids = tokenizer(c["completion"], add_special_tokens=False)["input_ids"]
        if c["terminated_with_eos"]:
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        n = len(resp_ids)
        # keep completions that would NOT be masked at the training cap, and prompts that fit the training prompt cap
        if n < 2 or n >= L or len(prompt_ids) > int(cfg["max_prompt_length"]):
            continue
        device = next(policy.parameters()).device
        seq = torch.tensor([prompt_ids + resp_ids], device=device)
        attn = torch.ones_like(seq)
        resp = torch.tensor([resp_ids], device=device)
        lp, _ = response_token_logprobs(policy, seq, attn, len(prompt_ids), resp)
        out = {"prompt_id": c["prompt_id"], "n": n}
        for loss_type in ("grpo", "dr_grpo"):
            loss, _ = grpo_policy_loss(
                lp, lp.detach(), torch.ones(1, device=device), torch.ones_like(lp), lp.detach(),
                float(cfg["clip_epsilon"]), 0.0, loss_type, L,
            )
            grads = torch.autograd.grad(loss, params, retain_graph=True)
            out[f"grad_norm_{loss_type}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads)))
        rows.append(out)
        del lp, seq
        free_gpu()
    del policy
    free_gpu()

    df = pd.DataFrame(rows)
    df["ratio"] = df["grad_norm_dr_grpo"] / df["grad_norm_grpo"]
    df["expected_ratio"] = df["n"] / L  # analytic: (1/L) / (1/n)
    df.to_csv(repo_path(f"{cfg['results_dir']}/norm_gradient_per_completion.csv"), index=False)

    edges = np.quantile(df["n"], [0, 0.25, 0.5, 0.75, 1.0])
    df["length_bin"] = pd.cut(df["n"], bins=np.unique(edges), include_lowest=True)
    by_bin = df.groupby("length_bin", observed=True).agg(
        n_completions=("n", "size"), mean_tokens=("n", "mean"),
        grad_norm_grpo=("grad_norm_grpo", "mean"), grad_norm_dr_grpo=("grad_norm_dr_grpo", "mean"),
    )
    by_bin["dr_over_grpo"] = by_bin["grad_norm_dr_grpo"] / by_bin["grad_norm_grpo"]

    half = df["n"] <= df["n"].median()
    summary = {"n_completions": len(df), "length_min": int(df["n"].min()), "length_median": float(df["n"].median()),
               "length_max": int(df["n"].max()), "L_max": L}
    for lt in ("grpo", "dr_grpo"):
        g = df[f"grad_norm_{lt}"]
        slope = np.polyfit(np.log(df["n"]), np.log(g), 1)[0]
        summary[lt] = {"loglog_slope_gradnorm_vs_length": float(slope),
                       "share_of_total_gradient_from_shorter_half": float(g[half].sum() / g.sum())}
    summary["max_abs_error_of_ratio_vs_n_over_L"] = float((df["ratio"] - df["expected_ratio"]).abs().max())
    save_json(f"{cfg['results_dir']}/norm_gradient_summary.json", summary)
    by_bin.to_csv(repo_path(f"{cfg['results_dir']}/norm_gradient_by_length.csv"))
    pd.set_option("display.width", 200)
    print(by_bin.round(4).to_string())
    print(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------------------------------------------
# Forks, held-out evaluation, tables
# --------------------------------------------------------------------------------------------------------------
def train_stats(results_dir, tag):
    log = pd.read_json(repo_path(f"{results_dir}/train_log_{tag}.jsonl"), lines=True)
    ok = log[~log["update_skipped"]]
    corr = ok["grad_norm"].corr(np.log(ok["response_length"])) if len(ok) > 2 else float("nan")
    short = ok["response_length"] <= ok["response_length"].median()
    return {
        "updates": len(log), "skipped_updates": int(log["update_skipped"].sum()),
        "train_reward_mean": log["reward"].mean(), "train_reward_last4": log["reward"].iloc[-4:].mean(),
        "train_kl_ref_mean": log["kl_ref"].mean(), "train_kl_k3_mean": log["kl_k3"].mean(),
        "train_length_mean": log["response_length"].mean(), "train_length_last4": log["response_length"].iloc[-4:].mean(),
        "grad_norm_mean": ok["grad_norm"].mean(),
        "grad_norm_short_updates": ok.loc[short, "grad_norm"].mean(), "grad_norm_long_updates": ok.loc[~short, "grad_norm"].mean(),
        "corr_gradnorm_loglength": corr, "entropy_mean": ok["entropy"].mean(),
    }


def heldout(results_dir, tag):
    p = repo_path(f"{results_dir}/eval_{tag}.json")
    if not p.exists():
        return {}
    ev = json.loads(p.read_text())
    return {f"heldout_{k}": ev[k] for k in ("reward_mean", "reward_sem", "length_mean", "length_median", "truncation_rate",
                                           "kl_token_mean", "kl_sequence_sum_mean", "entropy_mean")}


def paired(results_dir, tag, other):
    g = lambda t: pd.read_json(repo_path(f"{results_dir}/eval_{t}_generations.jsonl"), lines=True).set_index("prompt_id")
    a, b = g(tag), g(other)
    d = a["reward"] - b["reward"].reindex(a.index)
    dl = a["length"] - b["length"].reindex(a.index)
    return {"reward_diff": d.mean(), "reward_sem": d.std(ddof=1) / np.sqrt(len(d)), "length_diff": dl.mean(),
            "length_sem": dl.std(ddof=1) / np.sqrt(len(dl)),
            "identical": int((a["response"] == b["response"].reindex(a.index)).sum()), "n": len(d)}


def write_cases(results_dir, tags, baseline="midpoint", k=3, max_chars=500):
    """Held-out prompts where the two normalizations' LENGTHS differ the most, for the qualitative section."""
    base = pd.read_json(repo_path(f"{results_dir}/eval_{baseline}_generations.jsonl"), lines=True).set_index("prompt_id")
    gens = {t: pd.read_json(repo_path(f"{results_dir}/eval_{t}_generations.jsonl"), lines=True).set_index("prompt_id") for t in tags}
    lines = ["# Normalization cases (held-out, greedy)\n"]
    a, b = gens["norm_grpo"], gens["norm_dr_grpo"]
    both = a.index.intersection(b.index)
    diff = (b.loc[both, "length"] - a.loc[both, "length"])
    for title, ids in (("Dr.GRPO much LONGER than canonical", diff.sort_values(ascending=False).head(k).index),
                       ("Dr.GRPO much SHORTER than canonical", diff.sort_values().head(k).index)):
        lines.append(f"\n## {title}\n")
        for pid in ids:
            lines += [f"\n### prompt {pid[:10]}  canonical len {a.loc[pid, 'length']} (reward {a.loc[pid, 'reward']:.2f}) vs "
                      f"Dr.GRPO len {b.loc[pid, 'length']} (reward {b.loc[pid, 'reward']:.2f}); midpoint len {base.loc[pid, 'length']}",
                      f"**Prompt:** {a.loc[pid, 'prompt'][:300]}",
                      f"**canonical:** {a.loc[pid, 'response'][:max_chars]}",
                      f"**Dr.GRPO:** {b.loc[pid, 'response'][:max_chars]}"]
    out = repo_path(f"{results_dir}/norm_cases.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--eval-max-prompts", type=int, default=100)
    ap.add_argument("--skip-gradstats", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = cfg["results_dir"]
    n_updates = int(cfg["fork_updates"])

    if not args.skip_gradstats and (args.force or not repo_path(f"{results_dir}/norm_gradient_summary.json").exists()):
        print("\n=== length-conditioned gradient statistic ===", flush=True)
        length_gradient_stats(args.config)

    from task3_grpo.continue_train import run_grpo
    from task3_grpo.evaluate import evaluate

    if not args.skip_train:
        for tag, loss_type in FORKS:
            adapter = f"outputs/task3_grpo/{tag}"
            if args.force or not (repo_path(adapter) / "adapter_config.json").exists():
                print(f"\n=== GRPO fork {tag}: loss_type={loss_type}, {n_updates} updates ===", flush=True)
                run_grpo(args.config, output=adapter, updates=n_updates, loss_type=loss_type, run_name=tag)
                free_gpu()

    if not args.skip_eval:
        todo = [("midpoint", cfg["paths"]["grpo_midpoint_policy"])] + [(t, f"outputs/task3_grpo/{t}") for t, _ in FORKS]
        for tag, adapter in todo:
            if args.force or not repo_path(f"{results_dir}/eval_{tag}.json").exists():
                print(f"\n=== held-out evaluation {tag} ===", flush=True)
                evaluate(args.config, adapter, tag, max_prompts=args.eval_max_prompts)
                free_gpu()

    rows = []
    for tag, loss_type in FORKS:
        if not repo_path(f"{results_dir}/train_log_{tag}.jsonl").exists():
            continue
        row = {"fork": tag, "loss_type": loss_type, **train_stats(results_dir, tag), **heldout(results_dir, tag)}
        if repo_path(f"{results_dir}/eval_{tag}_generations.jsonl").exists() and repo_path(f"{results_dir}/eval_midpoint_generations.jsonl").exists():
            p = paired(results_dir, tag, "midpoint")
            row.update({"paired_reward_vs_midpoint": p["reward_diff"], "paired_reward_sem": p["reward_sem"],
                        "paired_length_vs_midpoint": p["length_diff"], "paired_length_sem": p["length_sem"],
                        "identical_to_midpoint": p["identical"]})
        rows.append(row)
    if not rows:
        print("no fork logs yet; nothing to tabulate")
        return
    df = pd.DataFrame(rows)
    df.to_csv(repo_path(f"{results_dir}/norm_forks.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.set_index("fork").T.round(4).to_string())

    have = lambda t: repo_path(f"{results_dir}/eval_{t}_generations.jsonl").exists()
    if have("norm_grpo") and have("norm_dr_grpo"):
        p = paired(results_dir, "norm_dr_grpo", "norm_grpo")
        print(f"\nDr.GRPO vs canonical (paired): reward {p['reward_diff']:+.4f} (sem {p['reward_sem']:.4f}), "
              f"length {p['length_diff']:+.1f} (sem {p['length_sem']:.1f}), identical {p['identical']}/{p['n']}")
    if have("norm_grpo") and have("norm_grpo_rep"):
        p = paired(results_dir, "norm_grpo_rep", "norm_grpo")
        print(f"NOISE FLOOR canonical replicate vs canonical (paired): reward {p['reward_diff']:+.4f} (sem {p['reward_sem']:.4f}), "
              f"length {p['length_diff']:+.1f} (sem {p['length_sem']:.1f}), identical {p['identical']}/{p['n']}")
    if all(have(t) for t in ("midpoint", "norm_grpo", "norm_dr_grpo")):
        print("wrote", write_cases(results_dir, ["norm_grpo", "norm_dr_grpo"]))


if __name__ == "__main__":
    main()
