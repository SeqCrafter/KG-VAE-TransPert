"""WAVE loss types, with delta supervision and an explicit baseline anchor."""
import torch
from torch.nn import functional as F


def row_pearson(pred, target, eps=1e-8):
    pred = pred.float() - pred.float().mean(-1, keepdim=True)
    target = target.float() - target.float().mean(-1, keepdim=True)
    denom = pred.square().sum(-1).clamp_min(eps).sqrt() * target.square().sum(-1).clamp_min(eps).sqrt()
    # Constant rows get correlation zero, rather than NaN.
    return ((pred * target).sum(-1) / denom).clamp(-1, 1)


def objective(outputs, x0, xp, alpha=1.0, beta=0.01, gamma=1.0):
    target = xp.float() - x0.float()
    mse = F.mse_loss(outputs["delta"].float(), target)
    pearson = 1 - row_pearson(outputs["delta"], target).mean()
    mu, lv = outputs["mu"].float(), outputs["logvar"].float()
    kl = -0.5 * (1 + lv - mu.square() - lv.exp()).sum(-1).mean()
    rec = F.mse_loss(outputs["reconstruction"].float(), x0.float())
    loss = mse + alpha * pearson + beta * kl + gamma * rec
    return {"loss": loss, "delta_mse": mse, "pearson_loss": pearson,
            "kl": kl, "reconstruction_mse": rec,
            "delta_pearson": 1 - pearson,
            "expression_pearson": row_pearson(outputs["expression"], xp).mean()}
