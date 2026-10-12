from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, repo_path


def kl_tag(beta: float) -> str:
    return f"kl_{beta:g}".replace(".", "p")


def train_summary(results_dir, tag):
    """Training-time behaviour of one fork, from its per-update log."""
    log = pd.read_json(repo_path(f"{results_dir}/train_log_{tag}.jsonl"), lines=True)
    return {
        "train_reward_mean": log["reward"].mean(),
        "train_reward_first4": log["reward"].iloc[:4].mean(), "train_reward_last4": log["reward"].iloc[-4:].mean(),
        "train_kl_mean": log["kl_ref"].mean(), "train_kl_last": log["kl_ref"].iloc[-1],
        "train_entropy_mean": log["entropy"].mean(), "train_entropy_last": log["entropy"].iloc[-1],
        "train_length_mean": log["response_length"].mean(),
        "train_eos_rate": log["terminated_with_eos"].mean(),
        "grad_norm_mean": log["grad_norm"].mean(),
    }


def heldout_summary(results_dir, tag):
    p = repo_path(f"{results_dir}/eval_{tag}.json")
    if not p.exists():
        return {}
    ev = json.loads(p.read_text())
    keep = ("reward_mean", "reward_sem", "reward_effective_mean", "length_mean", "truncation_rate",
            "kl_token_mean", "kl_sequence_sum_mean", "entropy_mean")
    return {f"heldout_{k}": ev[k] for k in keep}


def load_generations(results_dir, tag):
    return pd.read_json(repo_path(f"{results_dir}/eval_{tag}_generations.jsonl"), lines=True).set_index("prompt_id")


def paired_vs(results_dir, tag, other):
    """Per-prompt reward change of `tag` relative to `other` (same held-out prompts)."""
    a, b = load_generations(results_dir, tag), load_generations(results_dir, other)
    d = a["reward"] - b["reward"].reindex(a.index)
    return {
        "mean": d.mean(), "sem": d.std(ddof=1) / np.sqrt(len(d)),
        "identical_responses": int((a["response"] == b["response"].reindex(a.index)).sum()), "n": len(d),
    }


def write_cases(results_dir, tags, baseline="midpoint", k=3, max_chars=700):
    """Held-out prompts where a fork's reward went UP while its response changed, for manual inspection.

    Reward gain is ranked, then the response is shown next to the baseline's so a human can judge whether
    the gain reflects a better answer or just a different (longer / more repetitive / formatted) one.
    """
    base = load_generations(results_dir, baseline)
    lines = ["# Reward-increase cases (fork vs untouched midpoint, held-out, greedy)\n"]
    for tag in tags:
        gen = load_generations(results_dir, tag)
        diff = (gen["reward"] - base["reward"].reindex(gen.index))
        changed = gen["response"] != base["response"].reindex(gen.index)
        top = diff[changed].sort_values(ascending=False).head(k)
        lines.append(f"\n## {tag}: {int(changed.sum())}/{len(gen)} responses differ from the midpoint\n")
        for pid, d in top.items():
            g, b = gen.loc[pid], base.loc[pid]
            lines += [
                f"\n### prompt {pid[:10]}  reward {b['reward']:.2f} -> {g['reward']:.2f} (+{d:.2f}); "
                f"length {b['length']} -> {g['length']}",
                f"**Prompt:** {g['prompt'][:300]}",
                f"**Midpoint:** {b['response'][:max_chars]}",
                f"**{tag}:** {g['response'][:max_chars]}",
            ]
    out = repo_path(f"{results_dir}/kl_cases.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--eval-max-prompts", type=int, default=100, help="held-out prompts per fork (identical for all)")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--analysis-only", action="store_true", help="do not train/evaluate; only rebuild tables")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = cfg["results_dir"]
    betas = [float(b) for b in cfg["kl_values"]]
    n_updates = int(cfg["fork_updates"])

    if not args.analysis_only:
        from task2_ppo.continue_train import run_ppo
        from task2_ppo.evaluate import evaluate

        for beta in betas:
            tag = kl_tag(beta)
            adapter = f"outputs/task2_ppo/{tag}"
            if args.force or not (repo_path(adapter) / "adapter_config.json").exists():
                print(f"\n=== PPO fork {tag}: beta_KL={beta}, {n_updates} updates ===", flush=True)
                run_ppo(args.config, output=adapter, updates=n_updates, kl_beta=beta, run_name=tag)
                gc.collect()
                torch.cuda.empty_cache()
            if not args.skip_eval and (args.force or not repo_path(f"{results_dir}/eval_{tag}.json").exists()):
                print(f"\n=== held-out evaluation {tag} ===", flush=True)
                evaluate(args.config, adapter, tag, max_prompts=args.eval_max_prompts)
                gc.collect()
                torch.cuda.empty_cache()

    rows = []
    for beta in betas:
        tag = kl_tag(beta)
        row = {"kl_beta": beta, **train_summary(results_dir, tag), **heldout_summary(results_dir, tag)}
        if repo_path(f"{results_dir}/eval_{tag}.json").exists() and repo_path(f"{results_dir}/eval_midpoint.json").exists():
            p = paired_vs(results_dir, tag, "midpoint")
            row.update({"paired_reward_vs_midpoint": p["mean"], "paired_sem": p["sem"],
                        "identical_to_midpoint": p["identical_responses"]})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(repo_path(f"{results_dir}/kl_forks.csv"), index=False)
    print(df.set_index("kl_beta").T.round(4).to_string())

    # Noise-floor control: the beta=0.10 fork and the epsilon=0.20 fork from Task 2.2 have IDENTICAL settings.
    rep_a, rep_b = "clip_0p2", kl_tag(0.1)
    if all(repo_path(f"{results_dir}/eval_{t}_generations.jsonl").exists() for t in (rep_a, rep_b)):
        p = paired_vs(results_dir, rep_b, rep_a)
        print(f"\nReplicate control ({rep_b} vs {rep_a}, identical config): paired reward diff "
              f"{p['mean']:+.4f} (sem {p['sem']:.4f}); identical responses {p['identical_responses']}/{p['n']}")

    if repo_path(f"{results_dir}/eval_midpoint_generations.jsonl").exists():
        tags = [kl_tag(b) for b in betas if repo_path(f"{results_dir}/eval_{kl_tag(b)}_generations.jsonl").exists()]
        print("\nwrote", write_cases(results_dir, tags))


if __name__ == "__main__":
    main()
