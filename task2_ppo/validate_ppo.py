"""Numeric checks for task2_ppo.ppo against the manual's equations. Run: python -m task2_ppo.validate_ppo"""
from __future__ import annotations

import torch

from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def check_clip_geometry():
    eps = 0.2
    mask = torch.ones(1, 1)
    # (ratio, advantage, expected per-token objective min(rho*A, clip(rho)*A))
    cases = [
        (1.5, 1.0, 1.2),    # A>0, ratio above 1+eps  -> clipped at (1+eps)*A, no incentive to go further
        (0.5, 1.0, 0.5),    # A>0, ratio below 1-eps  -> unclipped (min picks rho*A)
        (1.5, -1.0, -1.5),  # A<0, ratio above 1+eps  -> unclipped (min picks rho*A, pessimistic)
        (0.5, -1.0, -0.8),  # A<0, ratio below 1-eps  -> clipped at (1-eps)*A
        (1.0, 2.0, 2.0),
    ]
    for ratio, adv, expected in cases:
        new_logp = torch.log(torch.tensor([[ratio]]))
        loss, r, frac = ppo_policy_loss(new_logp, torch.zeros(1, 1), torch.tensor([[adv]]), mask, eps)
        assert abs(loss.item() + expected) < 1e-5, (ratio, adv, loss.item(), -expected)
    print("clip geometry OK")


def check_gae():
    # single sequence of 3 valid tokens + 1 padding; gamma=1, lambda=0.5, terminal reward 1 on last valid token
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 0.4, 0.6, 9.9]])  # padding value must not leak
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.5)
    d2 = 1.0 - 0.6                  # last token bootstraps with 0
    d1 = 0.0 + 0.6 - 0.4
    d0 = 0.0 + 0.4 - 0.2
    a2 = d2
    a1 = d1 + 0.5 * a2
    a0 = d0 + 0.5 * a1
    expected = torch.tensor([[a0, a1, a2, 0.0]])
    assert torch.allclose(adv, expected, atol=1e-6), (adv, expected)
    assert torch.allclose(ret[:, :3], expected[:, :3] + values[:, :3], atol=1e-6)
    print("GAE OK")


def check_shaping():
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    pol = torch.tensor([[-1.0, -2.0, -3.0, -9.0]])
    ref = torch.tensor([[-1.5, -2.0, -2.0, -9.0]])
    r = shaped_rewards(torch.tensor([2.0]), pol, ref, mask, beta_kl=0.1)
    expected = torch.tensor([[-0.05, 0.0, 0.1 + 2.0, 0.0]])  # r_t = r_task*1[t=T] - beta*(logp - ref)
    assert torch.allclose(r, expected, atol=1e-6), (r, expected)
    print("KL shaping OK")


def check_normalize():
    adv = torch.tensor([[1.0, 2.0, 3.0, 50.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    n = normalize_advantages(adv, mask)
    assert abs(n[0, :3].mean().item()) < 1e-6 and n[0, 3].item() == 0.0
    print("advantage normalisation OK")


if __name__ == "__main__":
    check_clip_geometry()
    check_gae()
    check_shaping()
    check_normalize()
