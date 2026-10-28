"""Build disk-backed paired expression arrays from LINCS Level 3 or paired h5ad."""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .consensus import modz

from .common import canonical_smiles, parse_drugs, save_json


def strings(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


class GCTX:
    """Read only sample blocks; accept either conventional matrix orientation."""
    def __init__(self, path, gene_ids):
        self.file = h5py.File(path, "r")
        self.matrix = self.file["0/DATA/0/matrix"]
        genes = strings(self.file["0/META/ROW/id"][:])
        self.samples = strings(self.file["0/META/COL/id"][:])
        mapping = {g: i for i, g in enumerate(genes)}
        self.gene_indices = np.array([mapping[str(g)] for g in gene_ids])
        self.sample_indices = {s: i for i, s in enumerate(self.samples)}
        self.samples_first = self.matrix.shape == (len(self.samples), len(genes))
        if not self.samples_first and self.matrix.shape != (len(genes), len(self.samples)):
            raise ValueError("GCTX shape disagrees with its row/column IDs")

    def read(self, sample_ids):
        inds = np.array([self.sample_indices[str(x)] for x in sample_ids])
        unique, inverse = np.unique(inds, return_inverse=True)
        if self.samples_first:
            values = self.matrix[unique, :][:, self.gene_indices]
        else:
            values = self.matrix[:, unique][self.gene_indices, :].T
        values = np.asarray(values[inverse], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite expression in GCTX")
        return values

    def close(self):
        self.file.close()


def allocate(output, n, genes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if n < 1 or len(genes) != 978:
        raise ValueError(f"Expected nonempty data and 978 genes, got {n}, {len(genes)}")
    if genes.gene_id.duplicated().any() or genes.gene_symbol.duplicated().any():
        raise ValueError("Gene IDs and symbols must be unique and ordered")
    if any((output / name).exists() for name in ("x0.npy", "xp.npy", "metadata.csv")):
        raise FileExistsError("Output data already exists; choose a new output directory")
    genes.to_csv(output / "genes.csv", index=False)
    arrays = [np.lib.format.open_memmap(output / name, mode="w+", dtype="float32",
                                       shape=(n, len(genes))) for name in ("x0.npy", "xp.npy")]
    return output, arrays


def cmap(args):
    genes = pd.read_csv(args.gene_info, sep="\t", dtype=str)
    if "feature_space" in genes:
        genes = genes.loc[genes.feature_space.str.lower() == "landmark", ["gene_id", "gene_symbol"]]
    elif "pr_is_lm" in genes:
        genes = genes.loc[pd.to_numeric(genes.pr_is_lm) == 1, ["pr_gene_id", "pr_gene_symbol"]]
        genes.columns = ["gene_id", "gene_symbol"]
    else:
        raise ValueError("Gene metadata needs feature_space (LINCS2020) or pr_is_lm (legacy GEO)")
    genes = genes.reset_index(drop=True)
    info = pd.read_csv(args.inst_info, sep="\t", low_memory=False)
    pert = pd.read_csv(args.pert_info, sep="\t", low_memory=False)
    # LINCS2020 beta uses sample_id/cell_iname/cmap_name, unlike legacy GEO.
    for source, target in (("sample_id", "inst_id"), ("cell_iname", "cell_id"), ("cmap_name", "pert_iname")):
        if target not in info and source in info:
            info = info.rename(columns={source: target})
    required = {"inst_id", "pert_id", "pert_type", "cell_id", "pert_time", "pert_dose"}
    if not required <= set(info):
        raise ValueError(f"Missing metadata columns: {required - set(info)}")
    # Other perturbation types in instinfo_beta are irrelevant to these two matrices.
    info = info.loc[info.pert_type.isin(["ctl_vehicle", "trt_cp"])].copy()
    info["pert_time"] = pd.to_numeric(info.pert_time, errors="coerce")
    info = info.loc[info.pert_time.notna() & (info.pert_time > 0)].copy()
    if "pert_time_unit" in info:
        time_factor = info.pert_time_unit.astype(str).str.lower().str.strip().map(
            {"h": 1., "hr": 1., "hrs": 1., "hour": 1., "hours": 1., "min": 1 / 60, "d": 24.})
        if time_factor.isna().any():
            raise ValueError("Unrecognized pert_time_unit; convert metadata times to hours first")
        info["pert_time"] *= time_factor
    info = info.loc[np.isclose(info.pert_time, args.time_hours)].copy()
    if info["cell_id"].isna().any():
        raise ValueError("Missing cell identity")
    if args.qc_pass_only:
        if "qc_pass" not in info:
            raise ValueError("--qc-pass-only requires qc_pass in metadata")
        passed = info.qc_pass.astype(str).str.strip().str.lower().isin(["1", "1.0", "true"])
        info = info.loc[passed].copy()
    if "pert_dose_unit" in info:
        factors = info.pert_dose_unit.astype(str).str.lower().str.strip().map(
            {"um": 1., "µm": 1., "μm": 1., "nm": 1e-3, "mm": 1e3, "m": 1e6})
    else:
        factors = 1.0  # Explicit input convention when unit column is absent: micromolar.
    info["dose_um"] = pd.to_numeric(info.pert_dose, errors="coerce") * factors
    if info.inst_id.duplicated().any():
        raise ValueError("Duplicate sample IDs in instance metadata")
    if args.gctx:
        if args.control_gctx or args.case_gctx:
            raise ValueError("Use either --gctx or both --control-gctx/--case-gctx")
        control_path = case_path = args.gctx
    else:
        if not args.control_gctx or not args.case_gctx:
            raise ValueError("LINCS2020 requires --control-gctx and --case-gctx")
        control_path, case_path = args.control_gctx, args.case_gctx
    with ExitStack() as stack:
        control_reader = GCTX(control_path, genes.gene_id.tolist())
        stack.callback(control_reader.close)
        case_reader = control_reader if control_path == case_path else GCTX(case_path, genes.gene_id.tolist())
        if case_reader is not control_reader:
            stack.callback(case_reader.close)
        controls = info.loc[(info.pert_type == "ctl_vehicle") &
                            info.inst_id.astype(str).isin(control_reader.sample_indices)]
        if "pert_iname" not in controls:
            raise ValueError("DMSO identification requires cmap_name/pert_iname in instance metadata")
        controls = controls.loc[controls.pert_iname.astype(str).str.strip().str.lower() == "dmso"]
        control_keys = ["cell_id", "pert_time"]
        control_groups = {key: group.inst_id.tolist() for key, group in controls.groupby(control_keys)}
        treated = info.loc[(info.pert_type == "trt_cp") & np.isfinite(info.dose_um) & (info.dose_um > 0) &
                           info.inst_id.astype(str).isin(case_reader.sample_indices)].copy()
        if args.dose_um is not None:
            treated = treated.loc[np.isclose(treated.dose_um, args.dose_um)].copy()
        if args.cells:
            treated = treated.loc[treated.cell_id.isin(args.cells)].copy()
        before = len(treated)
        mapping = pert.drop_duplicates("pert_id").set_index("pert_id")[args.smiles_column].to_dict()
        valid_smiles = {}
        for drug in treated.pert_id.dropna().unique():
            try:
                valid_smiles[drug] = canonical_smiles(mapping.get(drug, ""))
            except ValueError:
                pass
        treated["smiles"] = treated.pert_id.map(valid_smiles)
        treated = treated.loc[treated.smiles.notna()].copy()
        after_smiles = len(treated)
        matched = [key in control_groups for key in treated[control_keys].itertuples(index=False, name=None)]
        treated = treated.loc[matched].copy()
        after_controls = len(treated)
        # Dose is part of the condition: NEVER combine different doses of one compound.
        condition_keys = ["cell_id", "pert_id", "dose_um", "pert_time"]
        conditions = []
        for key, group in treated.groupby(condition_keys, sort=True):
            ids = group.inst_id.astype(str).tolist()
            if len(ids) < args.min_replicates:
                continue
            cell, drug, dose, time = key
            conditions.append({"cell_id": cell, "pert_id": drug, "dose_um": float(dose),
                               "time_hours": float(time), "smiles": group.smiles.iloc[0],
                               "replicate_count": len(ids), "replicate_ids": json.dumps(ids)})
        metadata = pd.DataFrame(conditions)
        if metadata.empty:
            raise ValueError("No conditions meet the minimum replicate requirement")
        # Count distinct chemical structures, not dose conditions or raw replicate rows.
        counts = metadata.groupby("cell_id").smiles.nunique()
        cell_summary = pd.DataFrame({"cell_id": counts.index, "compound_count": counts.values})
        cell_summary["retained"] = cell_summary.compound_count >= args.min_compounds
        retained_cells = set(cell_summary.loc[cell_summary.retained, "cell_id"])
        metadata = metadata.loc[metadata.cell_id.isin(retained_cells)].reset_index(drop=True)
        eligible_conditions = len(metadata)
        if args.max_samples:
            metadata = metadata.iloc[:args.max_samples].copy()
        output, (x0, xp) = allocate(args.output, len(metadata), genes)
        cell_summary.to_csv(output / "cell_filter_summary.csv", index=False)
        baseline = {}
        used_controls = []
        for key in tqdm(sorted(set(zip(metadata.cell_id, metadata.time_hours))), desc="Cell-line DMSO means"):
            ids = control_groups[key]
            total = np.zeros(978, dtype=np.float64)
            for start in range(0, len(ids), args.chunk_size):
                total += control_reader.read(ids[start:start + args.chunk_size]).sum(0, dtype=np.float64)
            baseline[key] = (total / len(ids)).astype(np.float32)
            used_controls.extend({"cell_id": key[0], "time_hours": key[1], "control_sample_id": sid} for sid in ids)
        weights_json = []
        for i, row in enumerate(tqdm(metadata.itertuples(index=False), total=len(metadata), desc="MODZ consensus")):
            ids = json.loads(row.replicate_ids)
            # Memory bounded by one condition (typically a handful of replicates).
            profiles = np.concatenate([case_reader.read(ids[j:j + args.chunk_size])
                                       for j in range(0, len(ids), args.chunk_size)], axis=0)
            consensus, weights = modz(profiles)
            xp[i] = consensus
            x0[i] = baseline[(row.cell_id, row.time_hours)]
            weights_json.append(json.dumps(weights.tolist()))
        x0.flush()
        xp.flush()
        metadata.insert(0, "sample_id", [f"MODZ_{i:08d}" for i in range(len(metadata))])
        metadata["replicate_weights"] = weights_json
        metadata["control_count"] = [len(control_groups[key]) for key in zip(metadata.cell_id, metadata.time_hours)]
        metadata.to_csv(output / "metadata.csv", index=False)
        pd.DataFrame(used_controls).to_csv(output / "control_samples.csv", index=False)
        save_json(output / "provenance.json", {
            "control_source": str(Path(control_path).resolve()), "case_source": str(Path(case_path).resolve()),
            "expression_scale": "LINCS Level3, MODZ-weighted expression (not Level5 z-scores)",
            "time_hours": args.time_hours, "dose_um_filter": args.dose_um,
            "condition_keys": condition_keys, "min_replicates": args.min_replicates,
            "min_compounds_per_cell": args.min_compounds, "compound_identity": "canonical SMILES",
            "qc_pass_only": args.qc_pass_only,
            "modz": "Spearman on 978 genes; off-diagonal correlations clipped at 0; weights=max(0.5*row_sum,0.01), normalized",
            "control_matching": control_keys, "control_type": "ctl_vehicle/DMSO; arithmetic mean across plates",
            "treated_instances_before_filtering": before, "instances_after_smiles": after_smiles,
            "instances_after_control_matching": after_controls, "conditions_with_enough_replicates": len(conditions),
            "conditions_after_cell_filter": eligible_conditions, "retained": len(metadata),
            "retained_cell_lines": int(metadata.cell_id.nunique()),
            "debug_max_samples": args.max_samples,
            "note": "One output row per drug/cell/dose/time condition. Cell filter precedes optional debug truncation."
        })


def dense(matrix):
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float32)


def h5ad(args):
    import anndata as ad
    # anndata backed mode only backs X; layers may be loaded into RAM by anndata.
    data = ad.read_h5ad(args.input, backed="r")
    try:
        genes = pd.read_csv(args.genes, dtype=str)
        names = data.var[args.gene_column].astype(str) if args.gene_column else data.var_names.astype(str)
        if names.duplicated().any():
            raise ValueError("h5ad gene identifiers must be unique")
        lookup = {g: i for i, g in enumerate(names)}
        inds = [lookup[g] for g in genes[args.gene_key]]
        metadata = data.obs.copy()
        times = pd.to_numeric(metadata[args.time_column], errors="raise")
        selected = np.flatnonzero(np.isclose(times, args.time_hours))
        metadata = metadata.iloc[selected].copy()
        metadata["sample_id"] = metadata.index.astype(str)
        metadata["cell_id"] = metadata[args.cell_column].astype(str)
        metadata["time_hours"] = times.iloc[selected].to_numpy()
        smiles, doses = [], []
        for s, d in zip(metadata[args.smiles_column], metadata[args.dose_column]):
            ss, ds = parse_drugs(s, d)
            smiles.append(json.dumps([canonical_smiles(x) for x in ss]))
            doses.append(json.dumps([x * args.dose_factor for x in ds]))
        metadata["smiles"], metadata["dose_um"] = smiles, doses
        output, (x0, xp) = allocate(args.output, len(selected), genes)
        ctrl = data.X if args.control_layer == "X" else data.layers[args.control_layer]
        perturbed = data.X if args.perturbed_layer == "X" else data.layers[args.perturbed_layer]
        for start in tqdm(range(0, len(selected), args.chunk_size), desc="Import h5ad"):
            rows = selected[start:start + args.chunk_size]
            a, b = dense(ctrl[rows, :])[:, inds], dense(perturbed[rows, :])[:, inds]
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite expression values")
            x0[start:start + len(rows)], xp[start:start + len(rows)] = a, b
        x0.flush()
        xp.flush()
        metadata.to_csv(output / "metadata.csv", index=False)
        save_json(output / "provenance.json", {"source": str(Path(args.input).resolve()),
                  "time_hours": args.time_hours, "control_layer": args.control_layer,
                  "perturbed_layer": args.perturbed_layer, "expression_scale": "unchanged"})
    finally:
        data.file.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("cmap")
    c.add_argument("--gctx", help="Legacy combined Control/Case GCTX only")
    c.add_argument("--control-gctx", help="LINCS2020 level3_beta_ctl GCTX")
    c.add_argument("--case-gctx", help="LINCS2020 level3_beta_trt_cp GCTX")
    for name in ("gene-info", "inst-info", "pert-info"):
        c.add_argument("--" + name, required=True)
    c.add_argument("--smiles-column", default="canonical_smiles")
    c.add_argument("--min-replicates", type=int, default=2)
    c.add_argument("--min-compounds", type=int, default=50)
    c.add_argument("--qc-pass-only", action="store_true", help="Optional additional qc_pass filter for both case and control")
    c.add_argument("--dose-um", type=float, help="Omit to retain all positive doses; use 10 for WAVE setting")
    c.add_argument("--cells", nargs="+")
    c.add_argument("--max-samples", type=int)
    h = sub.add_parser("h5ad")
    h.add_argument("--input", required=True)
    h.add_argument("--genes", required=True, help="CSV: gene_id,gene_symbol; exactly 978 rows in output order")
    h.add_argument("--gene-column", help="var column; default: var_names")
    h.add_argument("--gene-key", choices=["gene_id", "gene_symbol"], default="gene_symbol")
    h.add_argument("--control-layer", default="unpert_expr")
    h.add_argument("--perturbed-layer", default="X")
    h.add_argument("--smiles-column", default="smiles")
    h.add_argument("--dose-column", default="dose_um")
    h.add_argument("--dose-factor", type=float, default=1., help="Multiply input doses to obtain uM")
    h.add_argument("--time-column", default="time_hours")
    h.add_argument("--cell-column", default="cell_id")
    for parser in (c, h):
        parser.add_argument("--output", default="data/processed")
        parser.add_argument("--time-hours", type=float, default=24.)
        parser.add_argument("--chunk-size", type=int, default=256)
    args = p.parse_args()
    if args.chunk_size < 1:
        p.error("chunk-size must be positive")
    if args.command == "cmap":
        if args.min_replicates < 2 or args.min_compounds < 1:
            p.error("min-replicates must be >=2 and min-compounds must be >=1")
        if args.max_samples is not None and args.max_samples < 1:
            p.error("max-samples must be positive")
    (cmap if args.command == "cmap" else h5ad)(args)


if __name__ == "__main__":
    main()
