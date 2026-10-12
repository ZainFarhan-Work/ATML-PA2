from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


STD_TOL = 1e-6                    # released helper's tolerance: a group is informative iff reward std > this
PRACTICAL_THRESHOLDS = (0.1, 0.25, 0.5)  # exploratory: rewards come from a continuous model, so ties are rare
TRAIN_CAP = 512                   # completions at/over this length would be masked out of the loss in training
BINS = ("hard", "medium", "easy")


def regroup_equal_generation_budget(by_prompt, k: int, order: dict | None = None):
    """Partition the cached completions into groups of exactly K at a FIXED total number of generations.

    Every prompt contributes its 8 cached completions, so 192 completions always become 192/K groups:
    K=8 -> 24 groups (1 per prompt), K=4 -> 48 groups (2 per prompt), K=2 -> 96 groups (4 per prompt).
    A group never mixes prompts. Default partition = consecutive blocks of `generation_index`
    (0..K-1, K..2K-1, ...); `order` (prompt_id -> permutation of 0..7) gives a random partition instead.
    The 8 completions of a prompt are i.i.d. samples, so any partition is statistically equivalent; the
    random partitions are used only to check that the block choice does not matter.
    """
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        idx = list(order[pid]) if order is not None else list(range(8))
        for start in range(0, 8, k):
            groups.append({"prompt": pid, "rows": [rows[i] for i in idx[start:start + k]]})
    return groups


def add_true_lengths(by_prompt, tokenizer):
    """The cache's `completion_tokens` is an UPPER BOUND (equal within generation pairs, 45% of rows overstate
    the length by up to ~670 tokens, apparently batch-padded). Recompute each completion's own length by
    re-tokenising its text (+1 for the EOS token when it terminated)."""
    for rows in by_prompt.values():
        for r in rows:
            n = len(tokenizer(r["completion"], add_special_tokens=False)["input_ids"])
            r["true_tokens"] = n + (1 if r["terminated_with_eos"] else 0)
    return by_prompt


def prompt_difficulty_bins(by_prompt):
    """Define difficulty ONCE: rank prompts by their mean cached reward (all 8 completions) and cut into
    terciles -> hard (lowest third), medium, easy (highest third). Every K uses the same assignment."""
    means = {pid: float(np.mean([r["reward"] for r in rows[:8]])) for pid, rows in by_prompt.items()}
    ranked = sorted(means, key=means.get)
    n = len(ranked)
    return {pid: BINS[min(3 * i // n, 2)] for i, pid in enumerate(ranked)}, means


def group_stats(groups, k):
    """Per-group diagnostics for one regrouping."""
    out = []
    for g in groups:
        r = np.array([x["reward"] for x in g["rows"]], dtype=float)
        std = r.std()  # population std, as in group_relative_advantages
        adv = (r - r.mean()) / (std + STD_TOL) if std > STD_TOL else np.zeros_like(r)
        usable = sum(1 for x in g["rows"] if x["true_tokens"] < TRAIN_CAP and not x["clipped_at_max"])
        out.append({
            "prompt": g["prompt"], "std": std, "centred_var": float(((r - r.mean()) ** 2).mean()),
            "informative": bool(std > STD_TOL), "adv_var": float(adv.var()), "max_abs_adv": float(np.abs(adv).max()),
            "usable": usable, "no_usable": usable == 0,
        })
    return pd.DataFrame(out)


def summarise(df, k, bin_of, label_col="bin"):
    df = df.assign(**{label_col: df["prompt"].map(bin_of)})
    rows = []
    for name, sub in [("all", df)] + [(b, df[df[label_col] == b]) for b in BINS]:
        row = {
            "K": k, "bin": name, "n_groups": len(sub), "n_completions": len(sub) * k,
            "informative_rate": sub["informative"].mean(),
            "reward_std_mean": sub["std"].mean(),
            "within_group_variance": sub["centred_var"].mean(),        # variance of the un-normalised signal r - mu
            "advantage_variance": sub["adv_var"].mean(),               # variance of the normalised signal A (1 if informative)
            "max_abs_advantage": sub["max_abs_adv"].mean(),
            "fraction_no_usable_completion": sub["no_usable"].mean(),  # every member would be masked by the 512 cap
            "mean_usable_completions": sub["usable"].mean(),
        }
        for t in PRACTICAL_THRESHOLDS:
            row[f"informative_std>{t}"] = (sub["std"] > t).mean()
        rows.append(row)
    return rows


def baseline_noise(by_prompt, bin_of, k):
    """Standard error of a K-sample group mean (the GRPO baseline): s / sqrt(K), s = pooled within-prompt std."""
    out = {}
    for name in ("all",) + BINS:
        pids = [p for p in by_prompt if name == "all" or bin_of[p] == name]
        var = np.mean([np.var([r["reward"] for r in by_prompt[p][:8]], ddof=1) for p in pids])
        out[name] = float(np.sqrt(var / k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--random-partitions", type=int, default=200)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = add_true_lengths(load_k8_cache(cfg["group_cache"]), load_tokenizer(cfg["base_model"]))
    bin_of, prompt_mean = prompt_difficulty_bins(by_prompt)
    print(f"{len(by_prompt)} prompts x 8 completions = {sum(len(v[:8]) for v in by_prompt.values())} generations (fixed budget)")
    print("difficulty bins (terciles of prompt mean reward):",
          {b: round(float(np.mean([prompt_mean[p] for p in by_prompt if bin_of[p] == b])), 2) for b in BINS})

    rows = []
    for k in cfg["group_sizes"]:
        k = int(k)
        res = summarise(group_stats(regroup_equal_generation_budget(by_prompt, k), k), k, bin_of)
        noise = baseline_noise(by_prompt, bin_of, k)
        for r in res:
            r["baseline_se"] = noise[r["bin"]]
        rows += res
    table = pd.DataFrame(rows)
    table.to_csv(repo_path(f"{cfg['results_dir']}/group_size.csv"), index=False)

    # robustness: average the headline numbers over many random partitions
    rng = np.random.default_rng(int(cfg["seed"]))
    rob = []
    for k in cfg["group_sizes"]:
        vals = []
        for _ in range(args.random_partitions):
            order = {p: rng.permutation(8) for p in by_prompt}
            r = summarise(group_stats(regroup_equal_generation_budget(by_prompt, int(k), order), int(k)), int(k), bin_of)[0]
            vals.append((r["informative_rate"], r["reward_std_mean"], r["fraction_no_usable_completion"]))
        v = np.array(vals)
        rob.append({"K": int(k), "informative_rate": v[:, 0].mean(), "reward_std_mean": v[:, 1].mean(),
                    "reward_std_sd_across_partitions": v[:, 1].std(), "fraction_no_usable_completion": v[:, 2].mean()})
    rob = pd.DataFrame(rob)
    rob.to_csv(repo_path(f"{cfg['results_dir']}/group_size_random_partitions.csv"), index=False)

    show = ["K", "bin", "n_groups", "informative_rate", "reward_std_mean", "within_group_variance", "advantage_variance",
            "max_abs_advantage", "baseline_se", "fraction_no_usable_completion", "informative_std>0.25"]
    pd.set_option("display.width", 250)
    print(table[show].round(3).to_string(index=False))
    print("\nrandom-partition check (mean over", args.random_partitions, "partitions):")
    print(rob.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
