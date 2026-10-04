from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_long_prompts(rows, tokenizer, max_length):
    """Drop pairs whose prompt alone does not fit in max_length.

    encode_prompt_response refuses to truncate prompts, so these pairs cannot be encoded.
    Shared by training and evaluation so both use the same rule.
    """
    kept = [
        r for r in rows
        if len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(r), tokenize=True, add_generation_prompt=True
        )) < max_length
    ]
    print(f"filtered {len(rows) - len(kept)} / {len(rows)} pairs with over-long prompts")
    return kept


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    rows = filter_long_prompts(rows, tokenizer, int(cfg["max_sequence_length"]))
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    model, loader, optimizer = bundle["model"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    accum = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg["max_grad_norm"])
    device = next(model.parameters()).device
    log_path = Path(cfg["results_dir"]) / f"train_log_{run_name}.jsonl"
    elapsed = wall_timer()

    def to_device(batch):
        return {k: v.to(device) for k, v in batch.items()}

    def log_step(step, window):
        # window: list of per-microbatch stats since the last optimizer step
        rec = {"step": step, "beta": beta, "time_s": elapsed()}
        for key in window[0]:
            rec[key] = sum(w[key] for w in window) / len(window)
        append_jsonl(log_path, rec)
        print(rec, flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step, window = 0, []

    def optimizer_step():
        nonlocal step, window
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        for w in window:
            w["grad_norm"] = float(grad_norm)
        log_step(step, window)
        window = []

    for epoch in range(int(cfg["epochs"])):
        for i, (chosen, rejected) in enumerate(loader):
            chosen, rejected = to_device(chosen), to_device(rejected)

            # Reference log-probs: adapter disabled, no gradients.
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, chosen)
                ref_r, _, _ = response_sequence_logprobs(model, rejected)

            # Policy log-probs: adapter on, gradients on.
            pol_c, _, _ = response_sequence_logprobs(model, chosen)
            pol_r, _, _ = response_sequence_logprobs(model, rejected)

            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            (loss / accum).backward()

            window.append({"loss": loss.item(), **{k: v.item() for k, v in diag.items()}})
            if (i + 1) % accum == 0:
                optimizer_step()

        if window:  # leftover micro-batches at the end of the epoch
            optimizer_step()

    model.save_pretrained(str(output))
    save_json(Path(cfg["results_dir"]) / f"train_summary_{run_name}.json",
              {"run_name": run_name, "beta": beta, "optimizer_steps": step,
               "train_examples": len(bundle["rows"]), "wall_time_s": elapsed()})
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
