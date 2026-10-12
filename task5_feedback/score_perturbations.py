from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task5_feedback.judge_utils import RESULTS, TimedJudge, pref_score
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


CLEAN = "clean_correct"
# perturbation -> role in the study. In every controlled pair the CLEAN response is the diagnostically better one.
CATEGORIES = {
    "corrupt_reasoning_correct_final": "reasoning (same correct final, degraded reasoning)",
    "good_reasoning_wrong_final": "outcome (reasoning ~fixed, wrong designated final)",
    "persuasive_filler_correct": "filler (correct + irrelevant persuasive filler)",
    "gold_distractor_wrong_final": "distractor (mentions gold number, wrong designated final)",
}
ALL_VARIANTS = [CLEAN] + list(CATEGORIES)


def classify(better_score):
    """score of the diagnostically better response in a comparison: 1 -> better preferred, .5 -> tie, 0 -> wrong."""
    return {1.0: "better", 0.5: "tie", 0.0: "wrong"}[better_score]


def verifier_reward(row):
    return exact_reward(row["response"], str(row["gold_final"]))


def run_round_robin(judge, groups):
    """Judge every ordered pair of the five variants of every problem (20 ordered comparisons per problem)."""
    recs = []
    for pid, variants in groups.items():
        q = next(iter(variants.values()))["question"]
        for x, y in itertools.permutations(ALL_VARIANTS, 2):
            res = judge.compare(q, variants[x]["response"], variants[y]["response"])
            recs.append({"problem": pid, "x": x, "y": y, "judge_score_x": pref_score(res), "judge": res})
    return pd.DataFrame(recs)


def controlled_table(groups, rr):
    """Better / tie / wrong-preference rates for the four clean-vs-perturbed pair types, per mechanism."""
    rows, cons = [], {}
    for cat, desc in CATEGORIES.items():
        ver = [classify(1.0 if verifier_reward(v[CLEAN]) > verifier_reward(v[cat]) else 0.5 if verifier_reward(v[CLEAN]) == verifier_reward(v[cat]) else 0.0)
               for v in groups.values()]
        sub = rr[((rr.x == CLEAN) & (rr.y == cat)) | ((rr.x == cat) & (rr.y == CLEAN))]
        # score of the CLEAN response in each ordered comparison
        clean_score = np.where(sub.x == CLEAN, sub.judge_score_x, 1.0 - sub.judge_score_x)
        jud = [classify(float(c)) for c in clean_score]
        for mech, labels in (("verifier", ver), ("judge", jud)):
            n = len(labels)
            rows.append({"category": cat, "role": desc, "mechanism": mech, "n_comparisons": n,
                         "better_rate": labels.count("better") / n, "tie_rate": labels.count("tie") / n,
                         "wrong_rate": labels.count("wrong") / n})
        # order consistency of the judge: same verdict for (clean, pert) and (pert, clean) on the same problem
        by = defaultdict(dict)
        for (_, r), c in zip(sub.iterrows(), clean_score):
            by[r.problem]["fwd" if r.x == CLEAN else "rev"] = float(c)
        cons[cat] = float(np.mean([d["fwd"] == d["rev"] for d in by.values()]))
    return pd.DataFrame(rows), cons


def variant_scores(groups, rr):
    """Mean win rate of each variant against the other four (judge) and mean verifier reward, all five categories."""
    out = []
    for v in ALL_VARIANTS:
        as_x = rr[rr.x == v].judge_score_x
        as_y = 1.0 - rr[rr.y == v].judge_score_x
        out.append({"variant": v, "verifier_reward_mean": float(np.mean([verifier_reward(g[v]) for g in groups.values()])),
                    "judge_mean_win_rate": float(pd.concat([as_x, as_y]).mean())})
    return pd.DataFrame(out)


def sensitivities(table):
    t = table.set_index(["category", "mechanism"])
    res = {}
    for mech in ("verifier", "judge"):
        reason = t.loc[("corrupt_reasoning_correct_final", mech)]
        o1, o2 = t.loc[("good_reasoning_wrong_final", mech)], t.loc[("gold_distractor_wrong_final", mech)]
        res[mech] = {
            "S_reason": reason.better_rate, "reason_tie": reason.tie_rate, "reason_wrong": reason.wrong_rate,
            "S_outcome": (o1.better_rate + o2.better_rate) / 2, "outcome_tie": (o1.tie_rate + o2.tie_rate) / 2,
            "outcome_wrong": (o1.wrong_rate + o2.wrong_rate) / 2,
            "S_outcome_good_reasoning_wrong_final": o1.better_rate, "S_outcome_gold_distractor": o2.better_rate,
            "filler_susceptibility_prefers_filler": t.loc[("persuasive_filler_correct", mech)].wrong_rate,
            "filler_tie": t.loc[("persuasive_filler_correct", mech)].tie_rate,
            "distractor_robustness_prefers_clean": o2.better_rate,
        }
    return res


def write_cases(groups, rr, k=3, max_chars=450):
    lines = ["# Diagnostic cases: where the judge ties, errs, or the verifier cannot tell\n"]
    for cat, desc in CATEGORIES.items():
        sub = rr[((rr.x == CLEAN) & (rr.y == cat)) | ((rr.x == cat) & (rr.y == CLEAN))].copy()
        sub["clean_score"] = np.where(sub.x == CLEAN, sub.judge_score_x, 1.0 - sub.judge_score_x)
        per_problem = sub.groupby("problem")["clean_score"].mean().sort_values()
        lines.append(f"\n## {cat} -- {desc}\n")
        lines.append(f"judge preference for the CLEAN response per problem (1 = always clean, 0 = always the perturbed one): "
                     f"mean {per_problem.mean():.2f}; worst problems shown\n")
        for pid in per_problem.head(k).index:
            v = groups[pid]
            lines += [f"\n### problem {pid}: judge prefers clean in {per_problem[pid]:.2f} of orderings; verifier rewards clean={verifier_reward(v[CLEAN]):.0f} "
                      f"perturbed={verifier_reward(v[cat]):.0f}",
                      f"**Question:** {v[CLEAN]['question'][:250]}",
                      f"**Clean:** {v[CLEAN]['response'][-max_chars:]}",
                      f"**{cat}:** {v[cat]['response'][-max_chars:]}"]
    out = repo_path(f"{RESULTS}/diagnostic_cases.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups), "| variants/problem:", len(EXPECTED_VARIANTS))

    judge = TimedJudge(cfg)
    rr = run_round_robin(judge, groups)
    judge.save_timing("diagnostics")
    rr.to_csv(repo_path(f"{RESULTS}/diagnostic_round_robin.csv"), index=False)

    table, consistency = controlled_table(groups, rr)
    table.to_csv(repo_path(f"{RESULTS}/diagnostic_controlled.csv"), index=False)
    per_variant = variant_scores(groups, rr)
    per_variant.to_csv(repo_path(f"{RESULTS}/diagnostic_per_variant.csv"), index=False)
    sens = sensitivities(table)
    save_json(f"{RESULTS}/diagnostic_summary.json", {"sensitivities": sens, "judge_order_consistency": consistency})
    cases = write_cases(groups, rr)

    pd.set_option("display.width", 220)
    print(table.drop(columns="role").round(3).to_string(index=False))
    print("\njudge order-consistency (same verdict for clean-first and clean-second):", {k: round(v, 2) for k, v in consistency.items()})
    print("\nper variant:\n", per_variant.round(3).to_string(index=False))
    print("\nsensitivities:", save_dump(sens))
    print("wrote", cases)


def save_dump(obj):
    import json
    return json.dumps({m: {k: round(float(v), 3) for k, v in d.items()} for m, d in obj.items()}, indent=1)


if __name__ == "__main__":
    main()
