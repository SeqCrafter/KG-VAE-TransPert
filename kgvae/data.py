"""Lazy per-worker mmap loading and padding for variable-size drug combinations."""
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .common import canonical_smiles, parse_drugs


class PerturbationDataset(Dataset):
    def __init__(self, root, features, indices=None):
        self.root = Path(root)
        self.metadata = pd.read_csv(self.root / "metadata.csv")
        self.genes = pd.read_csv(self.root / "genes.csv", dtype=str)
        if self.genes.gene_id.tolist() != features["gene_ids"] or self.genes.gene_symbol.tolist() != features["gene_symbols"]:
            raise ValueError("Expression and MAP-KG gene identities/order disagree")
        indices = np.arange(len(self.metadata)) if indices is None else np.asarray(indices, dtype=np.int64)
        if not len(indices) or (indices < 0).any() or (indices >= len(self.metadata)).any():
            raise ValueError("Empty or out-of-range dataset indices")
        self.drug_embeddings = features["drug_embeddings"].detach().float().cpu()
        lookup = {s: i for i, s in enumerate(features["smiles"])}
        canonical = {}  # RDKit parsing once per distinct SMILES, not once per row
        self.drugs, self.doses = [], []
        for row in self.metadata.itertuples(index=False):
            smiles, doses = parse_drugs(row.smiles, row.dose_um)
            keys = [canonical[s] if s in canonical else canonical.setdefault(s, canonical_smiles(s)) for s in smiles]
            # None: a compound the MAP-KG export skipped (longer than the encoder context).
            self.drugs.append(None if any(k not in lookup for k in keys) else [lookup[k] for k in keys])
            self.doses.append(doses)
        missing = np.array([self.drugs[i] is None for i in indices])
        if missing.any():
            warnings.warn(f"Dropping {int(missing.sum())} of {len(indices)} rows whose compounds are absent "
                          "from the MAP-KG features (see features['skipped_smiles'])", stacklevel=2)
        self.indices = indices[~missing]
        if not len(self.indices):
            raise ValueError("No rows left with MAP-KG drug features")
        self._arrays = None
        self._pid = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"], state["_pid"] = None, None
        return state

    def __getitem__(self, item):
        if self._arrays is None or self._pid != os.getpid():
            self._arrays = [np.load(self.root / name, mmap_mode="r", allow_pickle=False)
                            for name in ("x0.npy", "xp.npy")]
            if any(x.shape != (len(self.metadata), len(self.genes)) for x in self._arrays):
                raise ValueError("Expression array shape disagrees with metadata")
            self._pid = os.getpid()
        idx = int(self.indices[item])
        x0, xp = [torch.from_numpy(np.array(x[idx], dtype=np.float32, copy=True)) for x in self._arrays]
        if not torch.isfinite(x0).all() or not torch.isfinite(xp).all():
            raise ValueError(f"Nonfinite expression in sample {idx}")
        drugs = self.drugs[idx]
        return {"x0": x0, "xp": xp, "drug_embeddings": self.drug_embeddings[drugs],
                "doses_um": torch.tensor(self.doses[idx], dtype=torch.float32), "index": idx}


def collate(batch):
    count = max(len(x["doses_um"]) for x in batch)
    drugs = torch.zeros(len(batch), count, batch[0]["drug_embeddings"].shape[-1])
    doses = torch.zeros(len(batch), count)
    mask = torch.zeros(len(batch), count, dtype=torch.bool)
    for i, row in enumerate(batch):
        n = len(row["doses_um"])
        drugs[i, :n], doses[i, :n], mask[i, :n] = row["drug_embeddings"], row["doses_um"], True
    return {"x0": torch.stack([x["x0"] for x in batch]), "xp": torch.stack([x["xp"] for x in batch]),
            "drug_embeddings": drugs, "doses_um": doses, "drug_mask": mask,
            "index": torch.tensor([x["index"] for x in batch])}


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size=16, num_workers=0, shuffle=False, seed=42, pin_memory=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, pin_memory=pin_memory, worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=num_workers > 0, drop_last=False)
