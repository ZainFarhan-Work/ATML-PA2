from __future__ import annotations

import argparse
import gc
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json
from common.models import load_policy, load_tokenizer
from task5_feedback.judge_utils import RESULTS, TimedJudge, pref_score
from task5_feedback.rlvr import exact_reward, extract_designated_final

PAIRS = [("rlaif", "sft"), ("rlvr", "sft"), ("rlaif", "rlvr")]  # (first, second): first's win rate against second


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def responses_path(dataset, name):
    return repo_path(f"{RESULTS}/{dataset}_responses_{name}.jsonl")


def generate_for_policy(cfg, rows, tokenizer, dataset, name, batch_size):
    """Greedy, identical decoding for every policy (deterministic, so SFT/RLVR/RLAIF are matched)."""
    path = responses_path(dataset, name)
    if path.exists() and len(read_jsonl(path)) == len(rows):
        print(f"reuse {path.name}")
        return read_jsonl(path)
    policy = load_frozen_policy(cfg, name)
    out, t0 = [], time.perf_counter()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        gen = batch_generate(policy, tokenizer, [r["messages"] for r in chunk], max_prompt_length=512,
                             max_new_tokens=int(cfg["math_max_new_tokens"]), do_sample=False)
        for j, r in enumerate(chunk):
            out.append({"idx": i + j, "policy": name, "response": gen["responses"][j],
                        "tokens": int(gen["response_lengths"][j]), "hit_cap": bool(gen["truncated"][j])})
        print(f"[{dataset}/{name}] generated {len(out)}/{len(rows)}", flush=True)
    gen_seconds = time.perf_counter() - t0
    path.unlink(missing_ok=True)
    for r in out:
        append_jsonl(path, r)
    save_json(f"{RESULTS}/{dataset}_gen_time_{name}.json", {"seconds": gen_seconds, "n": len(rows)})
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return out


def failure_type(pred, correct, hit_cap):
    if correct:
        return "correct"
    if pred is None:
        return "truncated_no_final" if hit_cap else "missing_final_format"
    return "wrong_final_answer"


def policy_metrics(rows, resp):
    pred = [extract_designated_final(r["response"]) for r in resp]
    correct = [exact_reward(r["response"], str(row["gold_final"])) for r, row in zip(resp, rows)]
    types = [failure_type(p, c, r["hit_cap"]) for p, c, r in zip(pred, correct, resp)]
    L = np.array([r["tokens"] for r in resp], dtype=float)
    n = len(rows)
    return {
        "n": n, "accuracy": float(np.mean(correct)),
        "accuracy_sem": float(np.std(correct, ddof=1) / np.sqrt(n)),
        "format_compliance": float(np.mean([p is not None for p in pred])),
        "length_mean": float(L.mean()), "length_std": float(L.std()), "length_median": float(np.median(L)),
        "hit_cap_rate": float(np.mean([r["hit_cap"] for r in resp])),
        "failure_types": {t: types.count(t) for t in ("correct", "wrong_final_answer", "missing_final_format", "truncated_no_final")},
    }, correct


def pairwise_vs_verifier(judge, rows, resp, correct, first, second):
    """Judge `first` against `second` on every problem and compare the judge's preference to the verifier's."""
    scores, rec = [], []
    for i, row in enumerate(rows):
        res = judge.compare(row["question"], resp[first][i]["response"], resp[second][i]["response"])
        s = pref_score(res)
        scores.append(s)
        d = correct[first][i] - correct[second][i]  # verifier: +1 first correct only, -1 second correct only, 0 tie
        rec.append({"idx": i, "pair": f"{first}_vs_{second}", "judge": res, "judge_score_first": s, "verifier_diff": d})
    df = pd.DataFrame(rec)
    sign = df["judge_score_first"].map({1.0: 1, 0.5: 0, 0.0: -1})
    disc = df["verifier_diff"] != 0
    out = {
        "win_rate_first": float(np.mean(scores)),
        "wins": int((df["judge"] == "A").sum()), "ties": int((df["judge"] == "TIE").sum()), "losses": int((df["judge"] == "B").sum()),
        "n_verifier_discriminates": int(disc.sum()), "n_verifier_tie": int((~disc).sum()),
        # where the verifier separates the two responses: does the judge pick the verifier-correct one?
        "judge_agrees_when_verifier_discriminates": float((sign[disc] == np.sign(df["verifier_diff"][disc])).mean()) if disc.any() else None,
        "judge_tie_when_verifier_discriminates": float((sign[disc] == 0).mean()) if disc.any() else None,
        "judge_opposes_when_verifier_discriminates": float((sign[disc] == -np.sign(df["verifier_diff"][disc])).mean()) if disc.any() else None,
        # where the verifier cannot separate them (both right / both wrong): how often does the judge still choose?
        "judge_prefers_when_verifier_ties": float((sign[~disc] != 0).mean()) if (~disc).any() else None,
        "full_agreement_rate": float((sign == np.sign(df["verifier_diff"])).mean()),
    }
    return out, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    if args.max_examples:
        rows = rows[: args.max_examples]
    print("Rows:", len(rows), "| policies:", list(policy_specs(cfg)))

    resp, correct, metrics = {}, {}, {}
    for name in policy_specs(cfg):
        resp[name] = generate_for_policy(cfg, rows, tokenizer, args.dataset, name, args.gen_batch_size)
        metrics[name], correct[name] = policy_metrics(rows, resp[name])
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics[name].items() if k != "failure_types"}, metrics[name]["failure_types"])

    pairwise, frames = {}, []
    if not args.skip_judge:
        judge = TimedJudge(cfg)
        for first, second in PAIRS:
            pairwise[f"{first}_vs_{second}"], df = pairwise_vs_verifier(judge, rows, resp, correct, first, second)
            frames.append(df)
            print(f"{first} vs {second}:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pairwise[f"{first}_vs_{second}"].items()}, flush=True)
        judge.save_timing(args.dataset)
        pd.concat(frames).to_csv(repo_path(f"{RESULTS}/{args.dataset}_pairwise.csv"), index=False)

    per_example = pd.DataFrame({f"correct_{n}": correct[n] for n in correct})
    per_example.insert(0, "idx", range(len(rows)))
    per_example.to_csv(repo_path(f"{RESULTS}/{args.dataset}_per_example.csv"), index=False)
    save_json(f"{RESULTS}/{args.dataset}_metrics.json", {"dataset": args.dataset, "n": len(rows), "policies": metrics, "pairwise": pairwise})


if __name__ == "__main__":
    main()
