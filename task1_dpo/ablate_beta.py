from __future__ import annotations

import argparse
import gc

import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task1_dpo.evaluate import evaluate
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p")


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_table(cfg, betas):
    """Collect eval summaries into one table (standard run included for reference, different budget)."""
    results_dir = cfg["results_dir"]
    rows = []
    std_path = repo_path(f"{results_dir}/eval_standard.json")
    if std_path.exists():
        rows.append({"condition": "standard (1 epoch)", **load_json(std_path)})
    for beta in betas:
        p = repo_path(f"{results_dir}/eval_{beta_tag(beta)}.json")
        if p.exists():
            rows.append({"condition": f"short fork, beta={beta:g}", **load_json(p)})
    cols = ["condition", "beta", "dpo_loss", "preference_accuracy", "margin_mean", "kl_token_mean",
            "kl_sequence_sum_mean", "reward_mean", "reward_std", "length_mean", "length_std",
            "length_iqr", "frac_hit_cap_no_eos"]
    df = pd.DataFrame(rows)[cols]
    out = repo_path(f"{results_dir}/beta_summary.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="+", help="override the beta list from the config")
    ap.add_argument("--gen-examples", type=int, help="generate for only N held-out prompts (keep identical across runs)")
    ap.add_argument("--gen-batch-size", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if outputs already exist")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n_train = int(cfg["short_ablation_examples"])
    print("betas:", betas, "| short-run examples:", n_train)

    for beta in betas:
        tag = beta_tag(beta)
        adapter_dir = f"outputs/task1_dpo/{tag}"
        eval_json = repo_path(f"{cfg['results_dir']}/eval_{tag}.json")

        # Every fork restarts from a fresh LoRA on the base model (same seed, same 600 examples).
        if args.force or not (repo_path(adapter_dir) / "adapter_config.json").exists():
            print(f"\n=== training {tag} ===", flush=True)
            run_training(args.config, tag, output_path=adapter_dir, beta=beta, max_examples=n_train)
            free_gpu()
        else:
            print(f"skip training {tag} (adapter exists; use --force to redo)")

        if args.force or not eval_json.exists():
            print(f"\n=== evaluating {tag} ===", flush=True)
            evaluate(args.config, adapter_dir, tag, beta=beta,
                     gen_examples=args.gen_examples, gen_batch_size=args.gen_batch_size)
            free_gpu()
        else:
            print(f"skip evaluation {tag} (results exist; use --force to redo)")

    build_table(cfg, betas)


if __name__ == "__main__":
    main()
