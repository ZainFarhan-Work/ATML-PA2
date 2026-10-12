from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.metrics import masked_mean
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def build_cached_batch(cfg, rows):
    """Pad the 32 cached rollouts into [B, R] tensors and compute shaped rewards, GAE and advantages.

    The cache stores old (rollout-policy) log-probs, reference log-probs, critic values and the
    effective terminal reward, which is everything needed to rebuild the PPO advantages.
    """
    B, R = len(rows), max(len(r["old_logprobs"]) for r in rows)

    def pad(key):
        t = torch.zeros(B, R)
        for i, r in enumerate(rows):
            t[i, : len(r[key])] = r[key]
        return t

    old, ref, values = pad("old_logprobs"), pad("ref_logprobs"), pad("values")
    mask = torch.zeros(B, R)
    for i, r in enumerate(rows):
        mask[i, : len(r["old_logprobs"])] = 1.0
    task_reward = torch.tensor([r["effective_terminal_reward"] for r in rows])
    rewards = shaped_rewards(task_reward, old, ref, mask, float(cfg["kl_beta"]))
    adv, returns = compute_gae(rewards, values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    return {"old": old, "ref": ref, "mask": mask, "adv": normalize_advantages(adv, mask), "returns": returns}


def clip_geometry_table(batch, eps_values, scales):
    """Clipped surrogate and affected-token fraction on the fixed cached batch.

    The "new" policy is the supplied reference policy: rho = pi_ref / pi_old, a real displacement of
    known size away from the rollout policy. Larger `scale` multiplies the log-ratio (rho_s = rho^s),
    i.e. a synthetic "what if the update were s times larger" sweep; scale 1 is the real displacement.
    """
    old, ref, mask, adv = batch["old"], batch["ref"], batch["mask"], batch["adv"]
    rows = []
    for scale in scales:
        new = old + scale * (ref - old)
        for eps in eps_values:
            loss, ratio, outside = ppo_policy_loss(new, old, adv, mask, eps)
            unclipped = masked_mean(ratio * adv, mask).item()
            clipped = -loss.item()
            # clipping *binds* (zero gradient) only when the clipped branch is the minimum
            binds = (((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))).float()
            valid = mask.bool()
            rows.append({
                "scale": scale, "epsilon": eps,
                "unclipped_surrogate": unclipped, "clipped_surrogate": clipped,
                "surrogate_gap": unclipped - clipped,
                "affected_fraction": outside.item(),
                "gradient_blocked_fraction": masked_mean(binds, mask).item(),
                "ratio_min": ratio[valid].min().item(), "ratio_max": ratio[valid].max().item(),
                "n_tokens": int(mask.sum().item()),
            })
    return pd.DataFrame(rows)


def fork_stability(cfg, tag):
    log = pd.read_json(repo_path(f"{cfg['results_dir']}/train_log_{tag}.jsonl"), lines=True)
    out = {
        "reward_mean": log["reward"].mean(), "reward_first4": log["reward"].iloc[:4].mean(),
        "reward_last4": log["reward"].iloc[-4:].mean(),
        "kl_ref_mean": log["kl_ref"].mean(), "kl_ref_last": log["kl_ref"].iloc[-1],
        "entropy_mean": log["entropy"].mean(), "entropy_last": log["entropy"].iloc[-1],
        "grad_norm_mean": log["grad_norm"].mean(), "grad_norm_max": log["grad_norm"].max(),
        "clip_fraction_mean": log["clip_fraction"].mean(), "clip_fraction_max": log["clip_fraction"].max(),
        "policy_loss_abs_mean": log["policy_loss"].abs().mean(),
        "value_loss_mean": log["value_loss"].mean(),
        "response_length_mean": log["response_length"].mean(),
    }
    if "ratio_max" in log:
        out["ratio_max"], out["ratio_min"] = log["ratio_max"].max(), log["ratio_min"].min()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true", help="only run the cached-batch study")
    ap.add_argument("--skip-eval", action="store_true", help="train the forks but skip held-out evaluation")
    ap.add_argument("--eval-max-prompts", type=int, help="held-out prompts per fork (identical for every fork)")
    ap.add_argument("--force", action="store_true", help="redo forks/evaluations that already exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = cfg["results_dir"]
    eps_values = [float(e) for e in cfg["clip_values"]]

    # ---- Part A: cached-batch geometry ----
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    batch = build_cached_batch(cfg, rows)
    table = clip_geometry_table(batch, eps_values, scales=[1, 2, 4, 8])
    table.to_csv(repo_path(f"{results_dir}/clipping_cached_batch.csv"), index=False)
    print("Cached PPO rollouts:", len(rows), "| valid tokens:", int(batch["mask"].sum().item()))
    print(table.round(4).to_string(index=False))
    if args.skip_forks:
        return

    # ---- Part B: matched short forks from the identical midpoint checkpoint ----
    from task2_ppo.continue_train import run_ppo
    from task2_ppo.evaluate import evaluate

    n_updates = int(cfg["fork_updates"])
    summary_rows = []
    for eps in eps_values:
        tag = f"clip_{eps:g}".replace(".", "p")
        adapter = f"outputs/task2_ppo/{tag}"
        if args.force or not (repo_path(adapter) / "adapter_config.json").exists():
            print(f"\n=== PPO fork {tag}: epsilon={eps}, {n_updates} updates ===", flush=True)
            run_ppo(args.config, output=adapter, updates=n_updates, clip_epsilon=eps, run_name=tag)
            gc.collect()
            torch.cuda.empty_cache()
        if not args.skip_eval and (args.force or not repo_path(f"{results_dir}/eval_{tag}.json").exists()):
            print(f"\n=== held-out evaluation {tag} ===", flush=True)
            evaluate(args.config, adapter, tag, max_prompts=args.eval_max_prompts)
            gc.collect()
            torch.cuda.empty_cache()
        row = {"epsilon": eps, **fork_stability(cfg, tag)}
        ev = repo_path(f"{results_dir}/eval_{tag}.json")
        if ev.exists():
            row.update({f"heldout_{k}": v for k, v in __import__("json").loads(ev.read_text()).items()
                        if k in ("reward_mean", "reward_sem", "reward_effective_mean", "length_mean", "truncation_rate",
                                 "kl_token_mean", "entropy_mean", "n_prompts")})
        summary_rows.append(row)
    fork_df = pd.DataFrame(summary_rows)
    fork_df.to_csv(repo_path(f"{results_dir}/clipping_forks.csv"), index=False)
    print("\n" + fork_df.T.to_string(header=[f"eps={e}" for e in fork_df["epsilon"]]))


if __name__ == "__main__":
    main()
