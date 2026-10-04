from __future__ import annotations

import argparse
import textwrap

import pandas as pd

from common.data import repo_path


def show(row, width=100, max_chars=1500):
    prompt = row["prompt"][-1]["content"]
    print("=" * width)
    print(f"idx={row['idx']}  reward={row['reward']:.3f}  length={row['length']}  seq_kl={row['seq_kl']:.3f}")
    print("-" * width)
    print("PROMPT:", textwrap.shorten(prompt, 600, placeholder=" ..."))
    print("-" * width)
    resp = row["response"]
    print("RESPONSE:", resp[:max_chars] + (" ...[cut for display]" if len(resp) > max_chars else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="standard", help="eval run name (reads results/task1_dpo/eval_<name>_generations.jsonl)")
    ap.add_argument("--results-dir", default="results/task1_dpo")
    ap.add_argument("-k", type=int, default=3, help="how many examples per group")
    ap.add_argument("--cap", type=int, default=256, help="generation cap, used to flag cut-off responses")
    args = ap.parse_args()

    df = pd.read_json(repo_path(f"{args.results_dir}/eval_{args.name}_generations.jsonl"), lines=True)

    print(df[["length", "reward", "seq_kl"]].describe().round(3))
    print(f"\nresponses at the length cap ({args.cap}): {(df['length'] >= args.cap).sum()} / {len(df)}")
    print(f"correlation(length, reward) = {df['length'].corr(df['reward']):.3f}")

    groups = {
        f"HIGHEST {args.k} REWARDS": df.nlargest(args.k, "reward"),
        f"LOWEST {args.k} REWARDS": df.nsmallest(args.k, "reward"),
        f"LONGEST {args.k} RESPONSES": df.nlargest(args.k, "length"),
        f"SHORTEST {args.k} RESPONSES": df.nsmallest(args.k, "length"),
    }
    for title, sub in groups.items():
        print("\n\n" + "#" * 100 + f"\n# {title}\n" + "#" * 100)
        for _, row in sub.iterrows():
            show(row)


if __name__ == "__main__":
    main()
