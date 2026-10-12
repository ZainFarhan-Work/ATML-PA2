"""Shared helpers for Task 5: a timed wrapper around the fixed course judge and pairwise bookkeeping."""
from __future__ import annotations

import json
import time

from common.data import repo_path
from task5_feedback.rlaif import PairwiseAIJudge

RESULTS = "results/task5"
JUDGE_CACHE = f"{RESULTS}/judge_cache.json"
JUDGE_TIMING = f"{RESULTS}/judge_timing.json"


class TimedJudge(PairwiseAIJudge):
    """The unchanged course judge, plus wall-clock timing of every UNCACHED comparison (inference-cost evidence)."""

    def __init__(self, cfg):
        super().__init__(cfg, JUDGE_CACHE)
        self.call_seconds: list[float] = []

    def compare(self, problem, a, b):
        cached = self._key(problem, a, b) in self.cache
        t0 = time.perf_counter()
        out = super().compare(problem, a, b)
        if not cached:
            self.call_seconds.append(time.perf_counter() - t0)
        return out

    def save_timing(self, label):
        p = repo_path(JUDGE_TIMING)
        data = json.loads(p.read_text()) if p.exists() else {}
        if self.call_seconds:
            data[label] = {"n_uncached_calls": len(self.call_seconds),
                           "mean_seconds": sum(self.call_seconds) / len(self.call_seconds),
                           "total_seconds": sum(self.call_seconds)}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))


def pref_score(result: str) -> float:
    """Judge output for (a, b) -> score of `a`: win 1, tie 0.5, loss 0 (manual's convention)."""
    return {"A": 1.0, "TIE": 0.5, "B": 0.0}[result]
