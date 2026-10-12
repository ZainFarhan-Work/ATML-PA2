from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.evaluate import evaluate as _evaluate

# The GRPO config has no held-out settings. Use the PPO config's values so the GRPO, PPO and DPO policies
# are scored under one identical protocol (greedy decoding, 768-token cap, same reward-model input length).
PPO_EVAL_KEYS = ("eval_max_response_length", "missing_eos_penalty", "reward_max_length")


def evaluate(config_path, adapter, name, max_prompts=None, gen_batch_size=4, sample=False):
    ppo = load_yaml("configs/ppo.yaml")
    overrides = {k: ppo[k] for k in PPO_EVAL_KEYS}
    return _evaluate(config_path, adapter, name, max_prompts, gen_batch_size, sample, overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter dir, or 'none' for the base model")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, help="use only the first N held-out prompts (keep identical across compared runs)")
    ap.add_argument("--gen-batch-size", type=int, default=4)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()
    evaluate(args.config, args.adapter, args.name, args.max_prompts, args.gen_batch_size, args.sample)


if __name__ == "__main__":
    main()
