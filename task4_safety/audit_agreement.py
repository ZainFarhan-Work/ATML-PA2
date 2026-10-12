"""Step 4.3: manual-vs-AI agreement on the 60 audited SFT responses. Run: python -m task4_safety.audit_agreement"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task4_safety.evaluate_safety import LABELS, REFUSAL_LABELS, ANSWER_LABELS, wilson


def cohen_kappa(a, b, labels):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = sum(float((a == l).mean()) * float((b == l).mean()) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def family(label):
    return "refusal" if label in REFUSAL_LABELS else "answer" if label in ANSWER_LABELS else "ambiguous"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"]) / "task4_safety"
    man = pd.read_csv(args.labels or d / "manual_audit_labels.csv")
    man = man[(man["set"] == "main") & man["manual_label"].notna()].copy()
    key = {k["item_id"]: k for k in json.loads((d / "manual_audit_key.json").read_text())}
    judged = pd.DataFrame(read_jsonl(d / "judged_sft.jsonl")).set_index("xstest_id")
    man["judge_label"] = [judged.loc[int(i), "label"] for i in man["xstest_id"]]
    man["benchmark_class"] = [judged.loc[int(i), "benchmark_class"] for i in man["xstest_id"]]
    man["type"] = [judged.loc[int(i), "type"] for i in man["xstest_id"]]
    man["prompt"] = [judged.loc[int(i), "prompt"] for i in man["xstest_id"]]
    man["response"] = [judged.loc[int(i), "response"] for i in man["xstest_id"]]
    man["judge_confidence"] = [judged.loc[int(i), "confidence"] for i in man["xstest_id"]]
    assert set(man.manual_label) <= set(LABELS), set(man.manual_label) - set(LABELS)
    n = len(man)
    agree = man.manual_label == man.judge_label
    lo, hi = wilson(int(agree.sum()), n)

    # coarse axis: refusal vs answer (ignores whether the judge thought the request was benign or unsafe)
    man["manual_family"], man["judge_family"] = man.manual_label.map(family), man.judge_label.map(family)
    fam_agree = man.manual_family == man.judge_family
    # does the manual label match the benchmark's own class? (was the human's benign/unsafe call consistent with XSTest)
    expected = {"SAFE": {"SAFE_ANSWER", "OVER_REFUSAL"}, "UNSAFE": {"JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE"}}
    man["manual_matches_class"] = [l in expected[c] or l == "AMBIGUOUS" for l, c in zip(man.manual_label, man.benchmark_class)]
    man["judge_matches_class"] = [l in expected[c] or l == "AMBIGUOUS" for l, c in zip(man.judge_label, man.benchmark_class)]

    conf = pd.crosstab(man.manual_label.astype(pd.CategoricalDtype(LABELS)), man.judge_label.astype(pd.CategoricalDtype(LABELS)),
                       rownames=["manual"], colnames=["judge"], dropna=False)
    conf.to_csv(d / "audit_confusion.csv")

    summary = {
        "n": n, "exact_agreement": float(agree.mean()), "exact_agreement_ci95": [lo, hi], "n_agree": int(agree.sum()),
        "cohens_kappa": cohen_kappa(man.manual_label, man.judge_label, LABELS),
        "refusal_vs_answer_agreement": float(fam_agree.mean()),
        "refusal_vs_answer_kappa": cohen_kappa(man.manual_family, man.judge_family, ["refusal", "answer", "ambiguous"]),
        "ambiguous_rate_manual": float((man.manual_label == "AMBIGUOUS").mean()),
        "ambiguous_rate_judge": float((man.judge_label == "AMBIGUOUS").mean()),
        "by_class": {c: {"n": int((man.benchmark_class == c).sum()), "agreement": float(agree[man.benchmark_class == c].mean()),
                         "refusal_vs_answer_agreement": float(fam_agree[man.benchmark_class == c].mean())} for c in ("SAFE", "UNSAFE")},
        "manual_label_counts": man.manual_label.value_counts().to_dict(), "judge_label_counts": man.judge_label.value_counts().to_dict(),
        "manual_label_consistent_with_xstest_class": float(man.manual_matches_class.mean()),
        "judge_label_consistent_with_xstest_class": float(man.judge_matches_class.mean()),
        "judge_confidence_on_wrong_labels_mean": float(man.loc[~agree, "judge_confidence"].mean()) if (~agree).any() else None,
    }
    # what the judge's errors do to the headline numbers on the audited prompts
    safe = man[man.benchmark_class == "SAFE"]
    unsafe = man[man.benchmark_class == "UNSAFE"]
    summary["audited_safe_prompts"] = {
        "n": len(safe), "over_refusal_manual": float((safe.manual_label == "OVER_REFUSAL").mean()),
        "over_refusal_judge": float((safe.judge_label == "OVER_REFUSAL").mean()),
        "refusal_family_manual": float((safe.manual_family == "refusal").mean()), "refusal_family_judge": float((safe.judge_family == "refusal").mean())}
    summary["audited_unsafe_prompts"] = {
        "n": len(unsafe), "unsafe_compliance_manual": float((unsafe.manual_label == "UNSAFE_COMPLIANCE").mean()),
        "unsafe_compliance_judge": float((unsafe.judge_label == "UNSAFE_COMPLIANCE").mean()),
        "justified_refusal_manual": float((unsafe.manual_label == "JUSTIFIED_REFUSAL").mean()),
        "justified_refusal_judge": float((unsafe.judge_label == "JUSTIFIED_REFUSAL").mean())}
    save_json(d / "audit_agreement.json", summary)

    dis = man[~agree][["xstest_id", "benchmark_class", "type", "manual_label", "judge_label", "judge_confidence", "note", "prompt", "response"]]
    dis.to_csv(d / "audit_disagreements.csv", index=False)
    pd.set_option("display.width", 220)
    print(json.dumps(summary, indent=1, default=float))
    print("\nconfusion (rows = manual, cols = judge):\n", conf.to_string())
    print(f"\n{len(dis)} disagreements -> {d / 'audit_disagreements.csv'}")


if __name__ == "__main__":
    main()
