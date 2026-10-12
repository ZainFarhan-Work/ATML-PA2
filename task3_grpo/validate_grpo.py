"""Numeric checks for task3_grpo.grpo against the manual. Run: python -m task3_grpo.validate_grpo"""
from __future__ import annotations

import math

import torch

from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def check_group_advantages():
    # two prompts with very different reward scales; stats must be computed PER group
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 20.0, 20.0])
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    adv = group_relative_advantages(rewards, groups)
    std0 = math.sqrt(1.25)  # population std of 1..4
    assert torch.allclose(adv[:4], torch.tensor([-1.5, -0.5, 0.5, 1.5]) / std0, atol=1e-4), adv
    assert torch.allclose(adv[4:], torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=1e-4), adv  # mean 15, std 5
    for g in (0, 1):  # every group is zero-mean, unit-std
        a = adv[groups == g]
        assert abs(a.mean().item()) < 1e-5 and abs(a.std(unbiased=False).item() - 1) < 1e-4
    # group ids need not be sorted or contiguous
    perm = torch.tensor([4, 0, 5, 1, 6, 2, 7, 3])
    assert torch.allclose(group_relative_advantages(rewards[perm], groups[perm]), adv[perm], atol=1e-5)
    # an all-equal group is uninformative: advantage exactly 0, no NaN
    flat = group_relative_advantages(torch.tensor([2.0, 2.0, 2.0, 2.0, 1.0, 3.0]), torch.tensor([0, 0, 0, 0, 1, 1]))
    assert torch.all(flat[:4] == 0) and torch.isfinite(flat).all()
    print("group-relative advantages OK")


def check_loss():
    eps, beta = 0.2, 0.1
    old = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])  # lengths 3 and 1
    adv = torch.tensor([1.0, -1.0])
    ref = old.clone()
    # ratio 1: objective = A on every valid token
    loss, d = grpo_policy_loss(old, old, adv, mask, ref, eps, beta, "grpo")
    assert abs(loss.item() - 0.0) < 1e-6  # -mean(1, -1)
    # canonical: per-sequence mean first -> short and long completions weigh the same
    # dr_grpo: divide by constant L -> a token's weight does not depend on its sequence length
    adv2 = torch.tensor([1.0, 1.0])
    l_grpo, _ = grpo_policy_loss(old, old, adv2, mask, ref, eps, 0.0, "grpo")
    l_dr, _ = grpo_policy_loss(old, old, adv2, mask, ref, eps, 0.0, "dr_grpo", max_completion_length=4)
    assert abs(l_grpo.item() + 1.0) < 1e-6                      # -mean(3/3, 1/1)
    assert abs(l_dr.item() + (3 / 4 + 1 / 4) / 2) < 1e-6        # -mean(3/4, 1/4)
    # clip geometry (min): A>0 and ratio 1.5 -> capped at (1+eps)A ; A<0 and ratio 1.5 -> unclipped
    one = torch.ones(1, 1)
    new = torch.log(torch.tensor([[1.5]]))
    lp, _ = grpo_policy_loss(new, torch.zeros(1, 1), torch.tensor([1.0]), one, torch.zeros(1, 1) + new, eps, 0.0)
    ln, _ = grpo_policy_loss(new, torch.zeros(1, 1), torch.tensor([-1.0]), one, torch.zeros(1, 1) + new, eps, 0.0)
    assert abs(lp.item() + 1.2) < 1e-5 and abs(ln.item() - 1.5) < 1e-5, (lp.item(), ln.item())
    # k3 KL: exp(d) - d - 1 with d = ref - new ; zero when equal, positive otherwise
    ref2 = torch.full((1, 1), -0.5)
    new2 = torch.zeros(1, 1)
    _, d2 = grpo_policy_loss(new2, new2, torch.tensor([0.0]), one, ref2, eps, 1.0)
    assert abs(d2["sampled_kl"].item() - (math.exp(-0.5) + 0.5 - 1)) < 1e-6
    print("loss, normalisations, clipping, KL OK")


def check_truncation_mask():
    mask = torch.ones(3, 2)
    out = mask_truncated_sequences(mask, [False, True, False])
    assert out.sum(-1).tolist() == [2.0, 0.0, 2.0]
    # a fully masked completion contributes nothing and does not produce NaN
    loss, _ = grpo_policy_loss(torch.zeros(3, 2), torch.zeros(3, 2), torch.tensor([1.0, 5.0, -1.0]), out, torch.zeros(3, 2), 0.2, 0.1)
    assert torch.isfinite(loss) and abs(loss.item()) < 1e-6
    print("truncation masking OK")


if __name__ == "__main__":
    check_group_advantages()
    check_loss()
    check_truncation_mask()
