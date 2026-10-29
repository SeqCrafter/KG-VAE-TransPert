"""Export frozen MAP-KG drug/gene embeddings using the original MAP implementations.

MAP-KG aligns drug (smiles_projector) and gene (gene_projector) embeddings in one
space with InfoNCE over drug-gene mechanism edges. Their cosine similarity is thus
the encoder's drug-target interaction readout; it is computed from SMILES alone, so
it is available for drugs never seen in perturbation data. Besides the 978 landmark
gene embeddings, this exports each drug's top-K proteome-wide predicted targets.
"""
import argparse
import json
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .common import canonical_smiles, load_torch, parse_drugs, sha256
from .vendor.projector import ResidualProjector
from .vendor.molbart.STMencoder_ddp import MolSTM_Extractor

# Human NCBI Gene IDs anchor these historical LINCS symbols to the same gene.
# Sources: https://www.ncbi.nlm.nih.gov/gene/<gene_id>
GENE_RENAMES = {
    ("16", "AARS"): "AARS1", ("2058", "EPRS"): "EPRS1",
    ("7264", "TSTA3"): "GFUS", ("7485", "WRB"): "GET1",
    ("8349", "HIST2H2BE"): "H2BC21", ("9710", "KIAA0355"): "GARRE1",
    ("26128", "KIF1BP"): "KIFBP", ("79850", "FAM57A"): "TLCD3A",
    ("85236", "HIST1H2BK"): "H2BC12", ("94239", "H2AFV"): "H2AZ2",
}


def esm_keys(genes, esm, mapping):
    return [mapping.get(symbol, symbol if symbol in esm else
                        GENE_RENAMES.get((gene_id, symbol), symbol))
            for gene_id, symbol in zip(genes.gene_id, genes.gene_symbol)]


def checkpoint_state(path):
    raw = load_torch(path)
    for key in ("model_state_dict", "state_dict", "model"):
        if key in raw and isinstance(raw[key], dict):
            raw = raw[key]
            break
    result = {}
    for key, value in raw.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        result[key] = value
    return result


def substate(state, prefix):
    result = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not result:
        raise ValueError(f"Checkpoint has no {prefix} weights. Supply the full pretrained MAP-KG checkpoint.")
    return result


def projector_from_state(state):
    hidden, input_dim = state["mlp.0.weight"].shape
    output_dim = state["residual_proj.weight"].shape[0]
    module = ResidualProjector(input_dim, output_dim, hidden_dim=hidden)
    module.load_state_dict(state, strict=True)
    return module


class FrozenMAPKG(nn.Module):
    def __init__(self, checkpoint, vocab, molecule_checkpoint=None):
        super().__init__()
        state = checkpoint_state(checkpoint)
        mol_state = substate(state, "smiles_encoder.")
        # The upstream constructor needs a separate molecule_model.pth even when the
        # full MAP-KG checkpoint already contains all its tensors. Derive it losslessly.
        if molecule_checkpoint is None:
            base = substate(mol_state, "_model.")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "molecule_model.pth"
                torch.save(base, path)
                self.smiles_encoder = MolSTM_Extractor(str(path), str(vocab), pretrained=True)
        else:
            self.smiles_encoder = MolSTM_Extractor(str(molecule_checkpoint), str(vocab), pretrained=True)
        self.smiles_encoder.load_state_dict(mol_state, strict=True)
        self.smiles_projector = projector_from_state(substate(state, "smiles_projector."))
        self.gene_projector = projector_from_state(substate(state, "gene_projector."))
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.inference_mode()
    def encode_drugs(self, smiles, strict_vocab=False):
        smiles = [canonical_smiles(x) for x in smiles]
        # Upstream silently truncates long molecules; this adapter rejects them.
        tok = self.smiles_encoder.tokenizer.tokenize(smiles, pad=False)
        if any(len(x) > self.smiles_encoder.max_seq_len for x in tok["original_tokens"]):
            raise ValueError("SMILES exceeds MAP molecular encoder context length")
        unknown = self.smiles_encoder.tokenizer.unk_token
        vocab = self.smiles_encoder.tokenizer.vocab
        if any(t == unknown or t not in vocab for row in tok["original_tokens"] for t in row):
            if strict_vocab:
                raise ValueError("SMILES contains tokens absent from the supplied MAP vocabulary")
            warnings.warn("SMILES contains tokens absent from the MAP vocabulary; using the pretrained tokenizer's UNK mapping.", stacklevel=2)
        return self.smiles_projector(self.smiles_encoder(smiles))


@torch.inference_mode()
def proteome_targets(model, esm, drug_embeddings, k, chunk, device):
    """Top-k genes per drug by cosine similarity in the shared MAP-KG space.

    Only genes that are some drug's top-k are kept, as a compact vocabulary.
    """
    symbols = sorted(esm)
    drugs = torch.nn.functional.normalize(drug_embeddings.to(device).float(), dim=-1)
    best_score = torch.full((len(drugs), k), -2., device=device)
    best_index = torch.zeros((len(drugs), k), dtype=torch.long, device=device)
    for start in tqdm(range(0, len(symbols), chunk), desc="MAP-KG proteome targets"):
        block = torch.stack([torch.as_tensor(esm[s]).float().reshape(-1) for s in symbols[start:start + chunk]])
        genes = torch.nn.functional.normalize(model.gene_projector(block.to(device)), dim=-1)
        score = torch.cat((best_score, drugs @ genes.T), dim=1)
        index = torch.cat((best_index, torch.arange(start, start + len(block), device=device).expand(len(drugs), -1)), dim=1)
        best_score, order = score.topk(min(k, score.shape[1]), dim=1)
        best_index = index.gather(1, order)
    used, compact = torch.unique(best_index.cpu(), return_inverse=True)
    vectors = torch.stack([torch.as_tensor(esm[symbols[i]]).float().reshape(-1) for i in used.tolist()])
    embeddings = torch.cat([model.gene_projector(vectors[i:i + chunk].to(device)).cpu()
                            for i in range(0, len(vectors), chunk)])
    return compact, best_score.cpu(), [symbols[i] for i in used.tolist()], embeddings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True, help="The exact bart_vocab.txt used by MAP")
    p.add_argument("--molecule-checkpoint", help="Optional; normally extracted from full MAP-KG state")
    p.add_argument("--esm-dict", required=True, help="Raw ESM2 dict: gene symbol (or mapped key) -> vector")
    p.add_argument("--gene-map", help="Optional CSV: gene_symbol,esm_key for HGNC/Ensembl-keyed dictionaries")
    p.add_argument("--genes", default="data/processed/genes.csv")
    p.add_argument("--metadata", nargs="+", default=["data/processed/metadata.csv"])
    p.add_argument("--output", default="data/features/mapkg.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--strict-vocab", action="store_true", help="Reject unknown tokens instead of upstream UNK mapping")
    p.add_argument("--target-k", type=int, default=16,
                   help="Predicted targets per drug, ranked over every gene in the ESM dictionary")
    args = p.parse_args()
    genes = pd.read_csv(args.genes, dtype=str)
    if len(genes) != 978 or genes.gene_symbol.duplicated().any():
        raise ValueError("Expected 978 unique ordered landmark genes")
    smiles = set()
    for path in args.metadata:
        frame = pd.read_csv(path)
        for ss, ds in zip(frame.smiles, frame.dose_um):
            values, _ = parse_drugs(ss, ds)
            smiles.update(canonical_smiles(x) for x in values)
    smiles = sorted(smiles)
    if not smiles or args.batch_size < 1 or args.target_k < 1:
        raise ValueError("Empty drug list or nonpositive batch size")
    model = FrozenMAPKG(args.checkpoint, args.vocab, args.molecule_checkpoint).to(args.device)
    esm = load_torch(args.esm_dict)
    mapping = {}
    if args.gene_map:
        frame = pd.read_csv(args.gene_map, dtype=str)
        mapping = dict(zip(frame.gene_symbol, frame.esm_key))
    keys = esm_keys(genes, esm, mapping)
    absent = [k for k in keys if k not in esm]
    if absent:
        raise KeyError(f"Missing ESM vectors for {len(absent)} genes: {absent[:20]}. No zero filling.")
    gene_vectors = torch.stack([torch.as_tensor(esm[k]).float().reshape(-1) for k in keys])
    if gene_vectors.shape[1] != model.gene_projector.residual_proj.in_features:
        raise ValueError("ESM feature width disagrees with MAP-KG gene_projector")
    with torch.inference_mode():
        gene_embeddings = model.gene_projector(gene_vectors.to(args.device)).cpu()
        drug_embeddings = torch.cat([
            model.encode_drugs(smiles[i:i + args.batch_size], strict_vocab=args.strict_vocab).cpu()
            for i in tqdm(range(0, len(smiles), args.batch_size), desc="MAP-KG drugs")
        ])
    if not torch.isfinite(gene_embeddings).all() or not torch.isfinite(drug_embeddings).all():
        raise ValueError("Nonfinite MAP-KG embeddings")
    target_index, target_score, target_symbols, target_embeddings = proteome_targets(
        model, esm, drug_embeddings, args.target_k, args.batch_size * 32, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 2, "gene_ids": genes.gene_id.tolist(),
                "gene_symbols": genes.gene_symbol.tolist(), "gene_embeddings": gene_embeddings,
                "smiles": smiles, "drug_embeddings": drug_embeddings,
                # Per drug: indices into target_embeddings/target_symbols, and cosine scores.
                "target_index": target_index, "target_score": target_score,
                "target_symbols": target_symbols, "target_embeddings": target_embeddings,
                "provenance": {"esm_keys": keys, "strict_vocab": args.strict_vocab,
                               "target_k": args.target_k, "target_universe": len(esm),
                               "mapkg_sha256": sha256(args.checkpoint),
                               "vocab_sha256": sha256(args.vocab),
                               "esm_sha256": sha256(args.esm_dict)}}, output)


if __name__ == "__main__":
    main()
