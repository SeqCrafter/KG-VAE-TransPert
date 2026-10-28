"""Small shared utilities; no model or dataset code is executed on import."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_torch(path):
    # Original MAP checkpoints also store NumPy scalar training metadata.
    # Allow these data types locally, without enabling arbitrary pickle loading.
    numpy_core = np._core if hasattr(np, "_core") else np.core
    scalar = numpy_core.multiarray.scalar
    dtype_types = {type(np.dtype(t)) for t in np.sctypeDict.values()}
    with torch.serialization.safe_globals([
        (scalar, "numpy.core.multiarray.scalar"),
        (scalar, "numpy._core.multiarray.scalar"),
        np.dtype, *dtype_types,
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def canonical_smiles(value):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(str(value))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {value!r}")
    return Chem.MolToSmiles(mol, canonical=True)


def parse_drugs(smiles, doses):
    """One SMILES + scalar dose, or JSON lists. Never split SMILES on '.' or '+'."""
    smiles = str(smiles)
    ss = [smiles]
    if smiles.lstrip().startswith("["):
        # Bracket atoms such as [C-] and [NH4+] also start valid SMILES.
        # Only successfully decoded JSON arrays represent multiple drugs.
        try:
            ss = json.loads(smiles)
        except json.JSONDecodeError:
            pass
    ds = json.loads(str(doses)) if str(doses).lstrip().startswith("[") else [float(doses)]
    if not ss or len(ss) != len(ds):
        raise ValueError("smiles and dose_um must contain equally sized, nonempty lists")
    ds = [float(x) for x in ds]
    if any(not np.isfinite(x) or x < 0 for x in ds):
        raise ValueError("Doses must be finite and nonnegative, in micromolar")
    return ss, ds
