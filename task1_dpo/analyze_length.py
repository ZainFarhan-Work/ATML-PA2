from __future__ import annotations

import argparse
import gc

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json
from common.metrics import parse_word_limit, word_count, word_limit_compliance
from common.models import load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate, preference_eval
from task1_dpo.train import filter_long_prompts, run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def stratum_eval(policy, tokenizer, cfg, rows, beta):
    """Preference accuracy / margin on the length-stratified held-out set, split by stratum."""
    pref, margins = preference_eval(policy, rows, tokenizer, cfg, beta)
    strata = np.array([r["length_stratum"] for r in rows])
    out = {"overall": {"n": len(rows), "accuracy": pref["preference_accuracy"],
                       "margin_mean": pref["margin_mean"], "dpo_loss": pref["dpo_loss"]}}
    for s in STRATA:
        m = margins[strata == s]
        out[s] = {"n": int(len(m)), "accuracy": float((m > 0).mean()), "margin_mean": float(m.mean())}
    return out


def word_limit_eval(policy, tokenizer, cfg, name, gen_batch_size=2):
    """Greedy responses to the fixed word-limit prompts; length and compliance with the stated limit."""
    prompts = read_jsonl(cfg["paths"]["word_limit_prompts"])
    rows = []
    for i in range(0, len(prompts), gen_batch_size):
        chunk = prompts[i:i + gen_batch_size]
        out = batch_generate(
            policy, tokenizer, [p["messages"] for p in chunk],
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            do_sample=False,  # deterministic: only 10 prompts, so avoid sampling noise
        )
        for p, text, n_tok in zip(chunk, out["responses"], out["response_lengths"]):
            question = p["messages"][-1]["content"]
            rows.append({
                "model": name, "prompt_id": p["prompt_id"], "prompt": question, "response": text,
                "limit": parse_word_limit(question), "words": word_count(text), "tokens": n_tok,
                "compliant": word_limit_compliance(question, text),
            })
    gen_path = repo_path(f"{cfg['results_dir']}/word_limit_{name}.jsonl")
    gen_path.unlink(missing_ok=True)
    for r in rows:
        append_jsonl(gen_path, r)
    return {
        "n_prompts": len(rows),
        "compliance_rate": float(np.mean([r["compliant"] for r in rows])),
        "words_mean": float(np.mean([r["words"] for r in rows])),
        "words_std": float(np.std([r["words"] for r in rows])),
        "tokens_mean": float(np.mean([r["tokens"] for r in rows])),
    }


def analyze_model(cfg, name, adapter, strat_rows, beta):
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    result = {
        "model": name, "adapter": adapter,
        "stratified": stratum_eval(policy, tokenizer, cfg, strat_rows, beta),
        "word_limit": word_limit_eval(policy, tokenizer, cfg, name),
    }
    del policy
    free_gpu()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--force-train", action="store_true", help="retrain the length-balanced model even if it exists")
    ap.add_argument("--full-eval", action="store_true",
                    help="also run the full evaluate.py (loss/KL/reward/length) on the length-balanced model")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    beta = float(cfg["beta"])
    standard_adapter = cfg["standard_output"]
    length_adapter = cfg["length_output"]

    # 1. Train the length-balanced condition (same config as the standard run, different data).
    if args.force_train or not (repo_path(length_adapter) / "adapter_config.json").exists():
        print("=== training length-balanced DPO ===", flush=True)
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=length_adapter)
        free_gpu()
    else:
        print("skip training (length-balanced adapter exists; use --force-train to redo)")

    # 2. Held-out stratified pairs, filtered with the same long-prompt rule as training.
    tokenizer = load_tokenizer(cfg["base_model"])
    strat_rows = filter_long_prompts(read_jsonl(cfg["paths"]["dpo_length_eval"]), tokenizer,
                                     int(cfg["max_sequence_length"]))
    print({s: sum(r["length_stratum"] == s for r in strat_rows) for s in STRATA})

    # 3. Same analysis for both models.
    results = []
    for name, adapter in [("standard", standard_adapter), ("length_balanced", length_adapter)]:
        print(f"\n=== analyzing {name} ===", flush=True)
        results.append(analyze_model(cfg, name, adapter, strat_rows, beta))
    save_json(f"{cfg['results_dir']}/length_analysis.json", results)

    table = []
    for r in results:
        row = {"model": r["model"]}
        for s in ["overall"] + STRATA:
            row[f"acc_{s}"] = r["stratified"][s]["accuracy"]
            row[f"n_{s}"] = r["stratified"][s]["n"]
        row.update({f"wl_{k}": v for k, v in r["word_limit"].items()})
        table.append(row)
    df = pd.DataFrame(table)
    df.to_csv(repo_path(f"{cfg['results_dir']}/length_analysis.csv"), index=False)
    print("\n" + df.T.to_string(header=False))

    # 4. Optional: full metrics for the length-balanced model on the standard held-out set.
    if args.full_eval:
        evaluate(args.config, length_adapter, "length_balanced")


if __name__ == "__main__":
    main()
