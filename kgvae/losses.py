"""Two-branch loss and diagnostic metrics (模型架构.md, section 3).

delta_hat = b(x0) + g. b has no drug input, so MSE(b, delta) drives it to the
drug-shared response E[delta | x0]. g is supervised on r = delta - sg(b) with the
WAVE loss types (MSE + Pearson), so it only explains what b cannot. The
stop-gradient keeps the two branches from trading signal.
"""
import torch
from torch.nn import functional as F


def row_pearson(pred, target, eps=1e-8):
    pred = pred.float() - pred.float().mean(-1, keepdim=True)
    target = target.float() - target.float().mean(-1, keepdim=True)
    denom = pred.square().sum(-1).clamp_min(eps).sqrt() * target.square().sum(-1).clamp_min(eps).sqrt()
    # Constant rows get correlation zero, rather than NaN.
    return ((pred * target).sum(-1) / denom).clamp(-1, 1)


def top_k_genes(reference, k):
    return reference.abs().topk(min(k, reference.shape[-1]), dim=-1).indices


def objective(outputs, x0, xp, residual_std, corr=0.5, shared=1.0, wmse=0.0, wmse_c=2.0,
              direction=0.0, direction_margin=0.2, top_k=50):
    """L = shared * MSE(b, delta) + MSE(g, r) + corr * (1 - PCC(g, r)) [+ ablations].

    wMSE (ablation A): w_g = 1 + wmse * min(|r_g| / (c * sigma_g), 1), sigma_g the
    training residual std, so weights are dimensionless and outlier-bounded.
    Direction (ablation B): only the top-k |r| genes; m * tanh(g / m) saturates
    smoothly, unlike clamp, whose zero gradient freezes the worst wrong-sign errors.
    """
    target = xp.float() - x0.float()
    b = outputs["shared"].float()
    shared_mse = F.mse_loss(b, target)
    residual = target - b.detach()
    effect = outputs["drug_effect"].float()
    if wmse > 0:
        weights = 1 + wmse * (residual.abs() / (wmse_c * residual_std)).clamp(max=1.)
        mse = (weights * (effect - residual).square()).mean()
    else:
        mse = F.mse_loss(effect, residual)
    pearson = 1 - row_pearson(effect, residual).mean()
    loss = shared * shared_mse + mse + corr * pearson
    terms = {"shared_mse": shared_mse, "residual_mse": mse, "residual_pearson_loss": pearson}
    if direction > 0:
        indices = top_k_genes(residual, top_k)
        bounded = direction_margin * torch.tanh(effect.gather(-1, indices) / direction_margin)
        sign = torch.sign(residual.gather(-1, indices))
        direction_loss = F.softplus(direction_margin - bounded * sign).mean()
        loss = loss + direction * direction_loss
        terms["direction_loss"] = direction_loss
    return {"loss": loss, **terms, **metrics(outputs, target, top_k)}


@torch.no_grad()
def metrics(outputs, target, top_k=50):
    """Per-sample means, plus sums for the WAVE flattened delta Pearson (finalize)."""
    delta, b, ref = outputs["delta"].float(), outputs["shared"].float(), outputs["reference"].float()
    raw_top = top_k_genes(target, top_k)

    def direction(pred, truth):
        return (torch.sign(pred) == torch.sign(truth)).float().mean(-1)

    n = target.numel() / len(target)
    return {
        "delta_pcc": row_pearson(delta, target).mean(),
        # Model selection: zero for any drug-blind predictor of this cell, positive only
        # with drug-specific signal. Uses the nearest-reference mean, not the learned b.
        "residual_pcc": row_pearson(delta - ref, target - ref).mean(),
        "shared_pcc": row_pearson(b, target).mean(),
        "reference_pcc": row_pearson(ref, target).mean(),
        "top50_delta_pcc": row_pearson(delta.gather(-1, raw_top), target.gather(-1, raw_top)).mean(),
        "top50_direction_acc": direction(delta.gather(-1, raw_top), target.gather(-1, raw_top)).mean(),
        "delta_mse": (delta - target).square().mean(),
        "reference_mse": (ref - target).square().mean(),
        "drug_attention": outputs["drug_attention"].float().mean(),
        "target_attention": outputs["target_attention"].float().mean(),
        # Sufficient statistics of WAVE's np.corrcoef(pred.flatten(), truth.flatten()),
        # stored per sample (scaled by 1/n_genes) so that run_epoch's sample averaging is exact.
        "_sp": delta.sum(-1).mean() / n, "_st": target.sum(-1).mean() / n,
        "_spp": delta.square().sum(-1).mean() / n, "_stt": target.square().sum(-1).mean() / n,
        "_spt": (delta * target).sum(-1).mean() / n,
    }


def finalize(totals):
    """Epoch-level derived metrics from per-sample-averaged totals."""
    if totals.get("reference_mse", 0) > 0:
        totals["r2_vs_reference"] = 1 - totals["delta_mse"] / totals["reference_mse"]
    if "_sp" in totals:
        sp, st = totals.pop("_sp"), totals.pop("_st")
        spp, stt, spt = totals.pop("_spp"), totals.pop("_stt"), totals.pop("_spt")
        cov, vp, vt = spt - sp * st, spp - sp ** 2, stt - st ** 2
        # Comparable to the "ΔGene Pearson" WAVE reports and selects on.
        totals["wave_delta_pcc"] = cov / (vp * vt) ** 0.5 if vp > 0 and vt > 0 else 0.
    return totals
