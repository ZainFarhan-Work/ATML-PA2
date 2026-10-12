from __future__ import annotations

import argparse
import gc
import json
import time

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def generated_path(cfg, policy_name):
    """Where each policy's responses live (make_audit_sheet.py expects generated_sft.jsonl in this folder)."""
    return repo_path(cfg["results_dir"]) / "task4_safety" / f"generated_{policy_name}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", help="subset of sft dpo ppo grpo (default: all four)")
    ap.add_argument("--batch-size", type=int, default=8, help="same for every policy so decoding is identical")
    ap.add_argument("--force", action="store_true", help="regenerate even if a complete file exists")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    names = args.policies or list(policy_specs(cfg))
    n_rows = len(load_xstest(cfg))
    print("Policies:", names, "| XSTest rows:", n_rows, "| max new tokens:", cfg["safety_max_new_tokens"], "| greedy")

    timing = {}
    for name in names:
        path = generated_path(cfg, name)
        if path.exists() and not args.force and len(read_jsonl(path)) == n_rows:
            print(f"[{name}] complete file exists -> skip ({path.name})")
            continue
        print(f"\n=== generating {name} ===", flush=True)
        t0 = time.perf_counter()
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)
        timing[name] = time.perf_counter() - t0
        # the fixed XSTest order must be preserved and complete
        assert [r["xstest_id"] for r in records] == sorted(r["xstest_id"] for r in records) and len(records) == n_rows
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, records)
        lens = pd.Series([r["response_tokens"] for r in records])
        print(f"[{name}] saved {len(records)} responses -> {path} | mean tokens {lens.mean():.1f} | "
              f"at 256-token cap {(lens >= int(cfg['safety_max_new_tokens'])).mean():.1%} | {timing[name]:.0f}s", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    if timing:
        save_json(f"{cfg['results_dir']}/task4_safety/generation_time.json", timing)


if __name__ == "__main__":
    main()
