"""Optional fallback when the original MAP ESM dictionary is unavailable.

This generates NEW sequence embeddings, not a claimed reproduction of the original
MAP sequence selection. Prefer the authors' ESM dictionary whenever available.
"""
import argparse
from pathlib import Path

import pandas as pd
import requests
import torch
from tqdm import tqdm

from .common import save_json, sha256


def fetch_fasta(genes_path, output):
    genes = pd.read_csv(genes_path, dtype=str).gene_symbol.tolist()
    sequences, provenance = {}, {}
    for gene in tqdm(genes, desc="UniProt reviewed human proteins"):
        response = requests.get("https://rest.uniprot.org/uniprotkb/search", params={
            "query": f"(gene_exact:{gene}) AND (organism_id:9606) AND (reviewed:true)",
            "format": "json", "size": 500}, timeout=(30, 120))
        response.raise_for_status()
        entries = response.json()["results"]
        exact = [entry for entry in entries if any(
            g.get("geneName", {}).get("value") == gene for g in entry.get("genes", []))]
        if len(exact) != 1:
            raise ValueError(f"{gene}: expected one reviewed canonical record, found {len(exact)}; supply curated FASTA instead")
        entry = exact[0]
        sequences[gene] = entry["sequence"]["value"]
        provenance[gene] = entry["primaryAccession"]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{gene}\n{seq}\n" for gene, seq in sequences.items()))
    save_json(path.with_suffix(".uniprot.json"), provenance)


def read_fasta(path):
    sequences, name = {}, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in sequences:
                raise ValueError(f"Duplicate FASTA gene header: {name}")
            sequences[name] = ""
        elif name is None:
            raise ValueError("FASTA must start with a header")
        else:
            sequences[name] += line
    if any(not seq for seq in sequences.values()):
        raise ValueError("Empty protein sequence")
    return sequences


def embed(args):
    # Optional dependency, separate from downstream training requirements.
    from transformers import AutoModel, AutoTokenizer
    model_name = "facebook/esm2_t48_15B_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if args.fp16 else torch.float32,
                                      device_map="auto" if args.device == "auto" else None)
    if args.device != "auto":
        model.to(args.device)
    model.requires_grad_(False).eval()
    device = model.get_input_embeddings().weight.device
    sequences = read_fasta(args.fasta)
    genes = pd.read_csv(args.genes, dtype=str).gene_symbol.tolist()
    vectors = {}
    for gene in tqdm(genes, desc="ESM2-15B mean pooling"):
        sequence = sequences[gene]
        total, count = None, 0
        # Explicit non-overlapping windows: no silent dropping of long proteins.
        for start in range(0, len(sequence), args.window):
            piece = sequence[start:start + args.window]
            encoded = tokenizer(piece, return_tensors="pt", return_special_tokens_mask=True)
            special = encoded.pop("special_tokens_mask").to(device).bool()
            encoded = {k: v.to(device) for k, v in encoded.items()}
            valid = encoded["attention_mask"].bool() & ~special
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state.float()
                subtotal = hidden[valid].sum(0).cpu()
            total = subtotal if total is None else total + subtotal
            count += int(valid.sum())
        vector = total / count
        if vector.numel() != 5120 or not torch.isfinite(vector).all():
            raise ValueError(f"Invalid ESM2 feature for {gene}")
        vectors[gene] = vector
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, path)
    save_json(path.with_suffix(".provenance.json"), {
        "model": model_name, "fasta_sha256": sha256(args.fasta), "window_residues": args.window,
        "pooling": "last hidden layer; residue mean; length-weighted window mean",
        "note": "Recomputed alternative; original MAP sequence selection/pooling may differ."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("download-fasta")
    f.add_argument("--genes", default="data/processed/genes.csv")
    f.add_argument("--output", default="data/proteins/landmark.fasta")
    e = sub.add_parser("embed")
    e.add_argument("--genes", default="data/processed/genes.csv")
    e.add_argument("--fasta", required=True, help="FASTA header first field must equal gene_symbol")
    e.add_argument("--output", default="weights/landmark_esm2.pt")
    e.add_argument("--device", default="cuda", help="cuda, cpu, or auto (requires accelerate)")
    e.add_argument("--fp16", action="store_true")
    e.add_argument("--window", type=int, default=1022)
    args = p.parse_args()
    if args.command == "download-fasta":
        fetch_fasta(args.genes, args.output)
    else:
        if not 1 <= args.window <= 1022:
            p.error("window must be within [1,1022]")
        embed(args)


if __name__ == "__main__":
    main()
