from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from task5_feedback.judge_utils import JUDGE_TIMING, RESULTS

POLICIES = ("sft", "rlvr", "rlaif")


def load(name):
    p = repo_path(f"{RESULTS}/{name}")
    return json.loads(p.read_text()) if p.exists() else None


def accuracy_table(gsm, transfer):
    rows = []
    for pol in POLICIES:
        g, t = gsm["policies"][pol], transfer["policies"][pol]
        row = {"policy": pol,
               "gsm_accuracy": g["accuracy"], "gsm_format": g["format_compliance"], "gsm_length": g["length_mean"],
               "svamp_accuracy": t["accuracy"], "svamp_format": t["format_compliance"], "svamp_length": t["length_mean"],
               "accuracy_drop_gsm_to_svamp": g["accuracy"] - t["accuracy"],
               "relative_drop": (g["accuracy"] - t["accuracy"]) / g["accuracy"] if g["accuracy"] else None}
        for k, v in g["failure_types"].items():
            row[f"gsm_fail_{k}"] = v
        for k, v in t["failure_types"].items():
            row[f"svamp_fail_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(gsm, transfer):
    rows = []
    for pair in gsm["pairwise"]:
        g, t = gsm["pairwise"][pair], transfer["pairwise"][pair]
        rows.append({
            "pair": pair, "gsm_win_rate_first": g["win_rate_first"], "svamp_win_rate_first": t["win_rate_first"],
            "gsm_W/T/L": f"{g['wins']}/{g['ties']}/{g['losses']}", "svamp_W/T/L": f"{t['wins']}/{t['ties']}/{t['losses']}",
            "gsm_judge_agrees_when_verifier_discriminates": g["judge_agrees_when_verifier_discriminates"],
            "gsm_judge_opposes_when_verifier_discriminates": g["judge_opposes_when_verifier_discriminates"],
            "gsm_judge_prefers_when_verifier_ties": g["judge_prefers_when_verifier_ties"],
            "gsm_full_agreement": g["full_agreement_rate"],
            "svamp_judge_agrees_when_verifier_discriminates": t["judge_agrees_when_verifier_discriminates"],
            "svamp_full_agreement": t["full_agreement_rate"],
        })
    return pd.DataFrame(rows)


def identical_breakdown(dataset):
    """Split every judged pair into IDENTICAL responses (trivial) vs DIFFERENT ones (the informative comparisons).

    The adapters change the model very little, so under greedy decoding most responses coincide with the SFT
    baseline; a judge tie on identical text says nothing about the judge. Report everything on the different
    pairs separately.
    """
    from common.data import read_jsonl
    resp = {p: read_jsonl(repo_path(f"{RESULTS}/{dataset}_responses_{p}.jsonl")) for p in POLICIES}
    per = pd.read_csv(repo_path(f"{RESULTS}/{dataset}_per_example.csv"))
    pw = pd.read_csv(repo_path(f"{RESULTS}/{dataset}_pairwise.csv"))
    rows = []
    for pair, d in pw.groupby("pair"):
        a, b = pair.split("_vs_")
        d = d.sort_values("idx").reset_index(drop=True)
        d["identical"] = [resp[a][i]["response"] == resp[b][i]["response"] for i in d["idx"]]
        diff = d[~d.identical]
        score = diff["judge_score_first"]
        sign = score.map({1.0: 1, 0.5: 0, 0.0: -1})
        disc = diff["verifier_diff"] != 0
        rows.append({
            "dataset": dataset, "pair": pair, "n": len(d), "n_identical": int(d.identical.sum()), "n_different": len(diff),
            "judge_on_identical_W/T/L": f"{int((d[d.identical].judge == 'A').sum())}/{int((d[d.identical].judge == 'TIE').sum())}/{int((d[d.identical].judge == 'B').sum())}",
            "judge_on_different_W/T/L": f"{int((diff.judge == 'A').sum())}/{int((diff.judge == 'TIE').sum())}/{int((diff.judge == 'B').sum())}",
            "win_rate_first_on_different": float(score.mean()) if len(diff) else None,
            "judge_tie_rate_on_different": float((diff.judge == 'TIE').mean()) if len(diff) else None,
            "n_different_verifier_discriminates": int(disc.sum()),
            "judge_agrees_on_those": float((sign[disc] == diff["verifier_diff"][disc].map(lambda x: 1 if x > 0 else -1)).mean()) if disc.any() else None,
            "judge_tie_on_those": float((sign[disc] == 0).mean()) if disc.any() else None,
            "verifier_correct_first_only": int((diff["verifier_diff"] > 0).sum()), "verifier_correct_second_only": int((diff["verifier_diff"] < 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    load_yaml(args.config)
    gsm, transfer = load("gsm_metrics.json"), load("transfer_metrics.json")
    diag = load("diagnostic_summary.json")
    if gsm is None or transfer is None or diag is None:
        raise SystemExit("Run evaluate_math (--dataset gsm and --dataset transfer) and score_perturbations first.")

    acc = accuracy_table(gsm, transfer)
    pair = pairwise_table(gsm, transfer)
    acc.to_csv(repo_path(f"{RESULTS}/final_accuracy.csv"), index=False)
    pair.to_csv(repo_path(f"{RESULTS}/final_pairwise.csv"), index=False)
    pd.set_option("display.width", 250)
    print("== exact accuracy / format / length / failure types ==")
    print(acc.round(3).T.to_string(header=False))
    print("\n== pairwise AI judge and verifier-judge agreement ==")
    print(pair.round(3).T.to_string(header=False))

    brk = pd.concat([identical_breakdown("gsm"), identical_breakdown("transfer")])
    brk.to_csv(repo_path(f"{RESULTS}/final_pairwise_identical_split.csv"), index=False)
    print("\n== judge verdicts split by identical vs different responses ==")
    print(brk.round(3).T.to_string(header=False))

    sens = diag["sensitivities"]
    print("\n== controlled diagnostics (verifier vs judge) ==")
    print(pd.DataFrame(sens).round(3).to_string())
    print("judge order-consistency:", {k: round(v, 2) for k, v in diag["judge_order_consistency"].items()})

    # inference cost: one verifier call is a regex + float compare; the judge is a 3B LM per comparison.
    timing = load("judge_timing.json") or {}
    K = 4
    print("\n== inference cost ==")
    for label, d in timing.items():
        print(f"judge[{label}]: {d['mean_seconds']:.2f} s per uncached comparison over {d['n_uncached_calls']} calls")
    print(f"a group of K={K} needs K(K-1)/2 = {K * (K - 1) // 2} judge comparisons per prompt vs {K} regex checks for the verifier")

    save_json = {"accuracy": acc.to_dict(orient="records"), "pairwise": pair.to_dict(orient="records"),
                 "identical_split": brk.to_dict(orient="records"), "diagnostics": diag, "judge_timing": timing}
    repo_path(f"{RESULTS}/final_comparison.json").write_text(json.dumps(save_json, indent=2, default=float))
    print("\nwrote", repo_path(f"{RESULTS}/final_comparison.json"))


if __name__ == "__main__":
    main()
