"""Training-set statistics and drug-blind reference baselines.

Every statistic is computed from training rows only. Cell lines are never an
input to the model; they are used here only to build evaluation yardsticks.
"""
import numpy as np


def _blocks(dataset, chunk=4096):
    rows = np.asarray(dataset.indices)
    x0 = np.load(dataset.root / "x0.npy", mmap_mode="r", allow_pickle=False)
    xp = np.load(dataset.root / "xp.npy", mmap_mode="r", allow_pickle=False)
    for start in range(0, len(rows), chunk):
        idx = np.sort(rows[start:start + chunk])
        a = np.asarray(x0[idx], dtype=np.float64)
        yield idx, a, np.asarray(xp[idx], dtype=np.float64) - a


def _group_means(dataset, groups, n_groups, values="delta"):
    sums, counts = np.zeros((n_groups, len(dataset.genes))), np.zeros(n_groups)
    for idx, x0, delta in _blocks(dataset):
        g = groups[idx]
        np.add.at(sums, g, delta if values == "delta" else x0)
        np.add.at(counts, g, 1)
    return sums / np.maximum(counts, 1)[:, None], counts


def log_doses(dataset):
    """Single-drug log10(dose in uM) for every metadata row (training uses single drugs only)."""
    doses = np.array([d[0] for d in dataset.doses], dtype=np.float64)
    if not np.isfinite(doses).all() or (doses <= 0).any():
        raise ValueError("Doses must be finite and > 0 for log10 encoding")
    return np.log10(doses)


def training_cells(train):
    """Sorted training cell lines and a per-row index (-1 for rows of other cell lines)."""
    cells = train.metadata.cell_id.astype(str).to_numpy()
    names = sorted(set(cells[train.indices]))
    lookup = {c: i for i, c in enumerate(names)}
    return names, np.array([lookup.get(c, -1) for c in cells], dtype=np.int64)


def training_statistics(train, features):
    """Keyword arguments for KResidual.set_statistics, plus the training cell-line names."""
    n_genes = len(train.genes)
    total, square, n = np.zeros(n_genes), np.zeros(n_genes), 0
    for _, x0, _ in _blocks(train):
        total += x0.sum(0)
        square += np.square(x0).sum(0)
        n += len(x0)
    expr_mean = total / n
    expr_std = np.sqrt(np.maximum(square / n - expr_mean ** 2, 0.))
    names, cell_index = training_cells(train)
    reference_delta, counts = _group_means(train, cell_index, len(names))
    reference_x0, _ = _group_means(train, cell_index, len(names), values="x0")
    residual_square = np.zeros(n_genes)
    for idx, _, delta in _blocks(train):
        residual_square += np.square(delta - reference_delta[cell_index[idx]]).sum(0)
    drugs = np.unique([train.drugs[i][0] for i in train.indices])
    embeddings = features["drug_embeddings"].double().numpy()[drugs]
    doses = log_doses(train)[train.indices]
    stats = {"expr_mean": expr_mean, "expr_std": expr_std,
             "drug_mean": embeddings.mean(0), "drug_std": embeddings.std(0),
             "dose_mean": float(doses.mean()), "dose_std": float(doses.std()),
             "residual_std": np.sqrt(residual_square / n),
             "reference_x0": reference_x0, "reference_delta": reference_delta}
    global_mean = (reference_delta * counts[:, None]).sum(0) / counts.sum()
    return stats, names, global_mean


def _row_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    denom = np.sqrt(np.square(a).sum(1) * np.square(b).sum(1))
    return np.divide((a * b).sum(1), denom, out=np.zeros(len(a)), where=denom > 0)


def _nearest(x0, reference_x0, mean, std):
    a, b = (x0 - mean) / std, (reference_x0 - mean) / std
    distance = np.square(a).sum(1)[:, None] - 2 * a @ b.T + np.square(b).sum(1)[None]
    return distance.argmin(1)


def reference_baselines(train, evaluation, stats, global_mean, n_bins=4):
    """Drug-blind predictors scored with the model's metrics.

    global_mean: one training mean delta for every sample.
    nearest_cell: mean delta of the nearest training cell line in standardized x0
      space; for a seen cell line that is its own mean (residual PCC zero by construction).
    nearest_cell_dose: that cell line's mean per training dose-quantile bin. Dose alone
      drives a nonspecific stress response, so beating this shows drug identity is used.
    A model that does not beat nearest_cell_dose has learned no drug-specific signal.
    """
    expr_std = np.maximum(stats["expr_std"], 1e-2)
    names, cell_index = training_cells(train)
    train_doses = log_doses(train)
    edges = np.unique(np.quantile(train_doses[train.indices], np.linspace(0, 1, n_bins + 1)[1:-1]))
    width = len(edges) + 1
    groups = np.where(cell_index >= 0, cell_index * width + np.digitize(train_doses, edges), 0)
    dose_means, counts = _group_means(train, groups, len(names) * width)
    eval_bins = np.digitize(log_doses(evaluation), edges)
    keys = ("delta_pcc", "residual_pcc", "delta_mse", "reference_mse", "sp", "st", "spp", "stt", "spt")
    result = {name: {k: [] for k in keys} for name in ("global_mean", "nearest_cell", "nearest_cell_dose")}
    for idx, x0, delta in _blocks(evaluation):
        nearest = _nearest(x0, stats["reference_x0"], stats["expr_mean"], expr_std)
        ref = stats["reference_delta"][nearest]
        group = nearest * width + eval_bins[idx]
        cell_dose = np.where((counts[group] > 0)[:, None], dose_means[group], ref)
        for name, pred in (("global_mean", np.broadcast_to(global_mean, ref.shape)),
                           ("nearest_cell", ref), ("nearest_cell_dose", cell_dose)):
            r = result[name]
            r["delta_pcc"].append(_row_pearson(pred, delta))
            r["residual_pcc"].append(_row_pearson(pred - ref, delta - ref))
            r["delta_mse"].append(np.square(pred - delta).mean(1))
            r["reference_mse"].append(np.square(delta - ref).mean(1))
            for key, value in (("sp", pred), ("st", delta), ("spp", pred ** 2), ("stt", delta ** 2),
                               ("spt", pred * delta)):
                r[key].append(value.mean(1))
    summary = {}
    for name, r in result.items():
        v = {k: float(np.concatenate(x).mean()) for k, x in r.items()}
        cov, vp, vt = v["spt"] - v["sp"] * v["st"], v["spp"] - v["sp"] ** 2, v["stt"] - v["st"] ** 2
        summary[name] = {"delta_pcc": v["delta_pcc"], "residual_pcc": v["residual_pcc"],
                         "r2_vs_reference": 1 - v["delta_mse"] / v["reference_mse"],
                         "wave_delta_pcc": cov / np.sqrt(vp * vt) if vp > 0 and vt > 0 else 0.}
    summary["dose_bin_edges_log10_um"] = edges.tolist()
    summary["evaluation_cells_unseen_in_training"] = int(
        len(set(evaluation.metadata.cell_id.astype(str).to_numpy()[evaluation.indices]) - set(names)))
    return summary
