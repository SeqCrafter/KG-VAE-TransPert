"""Disjoint drug/cell/condition splits; replicate conditions never cross partitions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import canonical_smiles, parse_drugs, save_json, sha256


def make_split(root, method="drug", seed=42, val_fraction=0.1, test_fraction=0.1):
    root = Path(root)
    frame = pd.read_csv(root / "metadata.csv")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Require positive val/test fractions summing to less than one")
    keys = []
    for row in frame.itertuples(index=False):
        ss, ds = parse_drugs(row.smiles, row.dose_um)
        if len(ss) != 1:
            raise ValueError("Training splits accept single drugs only; use combinations for inference")
        smi = canonical_smiles(ss[0])
        if method == "drug":
            keys.append(smi)
        elif method == "cell":
            keys.append(str(row.cell_id))
        else:
            # All technical/biological replicates of a condition stay together.
            keys.append(repr((smi, str(row.cell_id), float(ds[0]), float(row.time_hours))))
    groups = np.array(sorted(set(keys)), dtype=object)
    if len(groups) < 3:
        raise ValueError("At least three distinct groups are needed")
    np.random.default_rng(seed).shuffle(groups)
    nval, ntest = max(1, int(len(groups) * val_fraction)), max(1, int(len(groups) * test_fraction))
    if nval + ntest >= len(groups):
        raise ValueError("Fractions leave no training groups")
    val, test = set(groups[:nval]), set(groups[nval:nval + ntest])
    result = {"train": [], "val": [], "test": []}
    for i, key in enumerate(keys):
        result["val" if key in val else "test" if key in test else "train"].append(i)
    result.update({"method": method, "seed": seed, "metadata_sha256": sha256(root / "metadata.csv")})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/processed")
    p.add_argument("--method", choices=["drug", "cell", "condition"], default="drug")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--output")
    args = p.parse_args()
    result = make_split(args.root, args.method, args.seed, args.val_fraction, args.test_fraction)
    save_json(args.output or Path(args.root) / f"split_{args.method}.json", result)


if __name__ == "__main__":
    main()
