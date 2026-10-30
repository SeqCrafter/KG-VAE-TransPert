"""Predict paired held-out data or unlabelled single-/multi-drug requests."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import canonical_smiles, load_torch, parse_drugs, save_json, sha256
from .data import PerturbationDataset, make_loader
from .export_mapkg import encode_new_drugs
from .model import KResidual
from .statistics import reference_baselines, training_statistics
from .train import forward, run_epoch, to_device


def load_model(checkpoint, features, device):
    saved = load_torch(checkpoint)
    if saved["gene_ids"] != features["gene_ids"] or saved["gene_symbols"] != features["gene_symbols"]:
        raise ValueError("Feature gene identities/order differ from checkpoint")
    if saved["feature_provenance"] != features["provenance"]:
        raise ValueError("Feature source weights differ from training")
    if "reference_cell_ids" not in saved:
        raise ValueError("Checkpoint predates the current K-Residual model; retrain")
    model = KResidual(features["gene_embeddings"], saved["drug_dim"], len(saved["reference_cell_ids"]),
                      **saved["config"]["model"])
    model.load_state_dict(saved["model"], strict=True)
    if not torch.equal(model.gene_embeddings.cpu(), features["gene_embeddings"].float()):
        raise ValueError("Gene embeddings changed; re-export with the same source vectors")
    return model.to(device).eval(), saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", default="data/features/mapkg.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", default="runs/predictions")
    sub = p.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--root", default="data/processed")
    ev.add_argument("--split", default="data/processed/split_drug.json")
    ev.add_argument("--partition", choices=["val", "test"], default="test")
    ev.add_argument("--shuffle-seed", type=int, default=0, help="Seed of the shuffled-drug control")
    ev.add_argument("--dose-um", type=float,
                    help="Evaluate only this dose (e.g. 10 for the WAVE setting); 1%% relative tolerance")
    req = sub.add_parser("requests")
    req.add_argument("--baseline", required=True, help="N x 978 float .npy, same scale as training")
    req.add_argument("--genes", required=True, help="genes.csv matching baseline column order")
    req.add_argument("--requests", required=True, help="CSV: baseline_index,smiles,dose_um,time_hours")
    req.add_argument("--mapkg-checkpoint", help="MAP-KG checkpoint used for export; needed for new compounds")
    req.add_argument("--vocab", help="bart_vocab.txt used for export; needed for new compounds")
    req.add_argument("--molecule-checkpoint", help="Optional, as in kgvae.export_mapkg")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    device = torch.device(args.device)
    features = load_torch(args.features)
    model, saved = load_model(args.checkpoint, features, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "delta.npy").exists():
        raise FileExistsError("Prediction output exists; choose another directory")
    if args.command == "evaluate":
        split = json.loads(Path(args.split).read_text())
        if split["metadata_sha256"] != sha256(Path(args.root) / "metadata.csv"):
            raise ValueError("Split metadata mismatch")
        if split != saved["split"]:
            raise ValueError("Evaluation split differs from the training checkpoint")
        indices = split[args.partition]
        if args.dose_um is not None:
            # Training uses single drugs only, so each row has exactly one dose.
            doses = PerturbationDataset(args.root, features, indices).doses
            indices = [i for i in indices if np.isclose(doses[i][0], args.dose_um, rtol=1e-2, atol=0)]
            if not indices:
                raise ValueError(f"No {args.partition} samples at {args.dose_um} uM")
        ds = PerturbationDataset(args.root, features, indices)
        times = ds.metadata.iloc[ds.indices].time_hours.to_numpy(dtype=float)
        if not np.isclose(times, saved["time_hours"]).all():
            raise ValueError("Evaluation time differs from training")
        loader = make_loader(ds, args.batch_size)
        metrics = run_epoch(model, loader, device, saved["config"]["loss"], amp="off")
        # Shuffled-drug control: same cells and doses, drug embeddings permuted across samples.
        # If residual PCC barely drops, the model does not use MAP-KG drug knowledge.
        shuffled = PerturbationDataset(args.root, features, indices)
        order = np.random.default_rng(args.shuffle_seed).permutation(len(shuffled.indices))
        permuted = [shuffled.drugs[shuffled.indices[j]] for j in order]
        for i, j in zip(shuffled.indices, permuted):
            shuffled.drugs[i] = j
        control = run_epoch(model, make_loader(shuffled, args.batch_size), device, saved["config"]["loss"], amp="off")
        train_ds = PerturbationDataset(args.root, features, split["train"])
        stats, _, global_mean = training_statistics(train_ds, features)
        baselines = reference_baselines(train_ds, ds, stats, global_mean)
        save_json(output / "metrics.json", {"model": metrics, "shuffled_drug_control": control,
                                            "baselines": baselines, "dose_um_filter": args.dose_um,
                                            "samples": len(indices)})
        metadata = ds.metadata.iloc[ds.indices].copy()
        batches = (to_device(batch, device) for batch in loader)
    else:
        baseline = np.load(args.baseline, mmap_mode="r", allow_pickle=False)
        genes = pd.read_csv(args.genes, dtype=str)
        if genes.gene_id.tolist() != saved["gene_ids"] or genes.gene_symbol.tolist() != saved["gene_symbols"]:
            raise ValueError("Baseline genes/order disagree with training")
        if baseline.ndim != 2 or baseline.shape[1] != len(saved["gene_ids"]):
            raise ValueError("Baseline must have shape N x 978")
        metadata = pd.read_csv(args.requests)
        if not np.isclose(pd.to_numeric(metadata.time_hours), saved["time_hours"]).all():
            raise ValueError("Request time differs from the model's fixed training time")
        requested = [s for r in metadata.itertuples(index=False) for s in parse_drugs(r.smiles, r.dose_um)[0]]
        if not {canonical_smiles(s) for s in requested} <= set(features["smiles"]):
            if not args.mapkg_checkpoint or not args.vocab:
                raise ValueError("Requests contain compounds outside the feature cache; pass --mapkg-checkpoint "
                                 "and --vocab so the frozen MAP-KG encoder can embed them")
            # Same frozen encoder as the cache: SMILES -> MolSTM -> smiles_projector.
            features = encode_new_drugs(features, requested, args.mapkg_checkpoint, args.vocab, device,
                                        args.molecule_checkpoint)
        lookup = {s: i for i, s in enumerate(features["smiles"])}

        def request_batches():
            for start in range(0, len(metadata), args.batch_size):
                rows = metadata.iloc[start:start + args.batch_size]
                parsed = [parse_drugs(r.smiles, r.dose_um) for r in rows.itertuples(index=False)]
                count = max(len(x[0]) for x in parsed)
                drugs = torch.zeros(len(rows), count, saved["drug_dim"])
                doses = torch.zeros(len(rows), count)
                mask = torch.zeros(len(rows), count, dtype=torch.bool)
                for i, (ss, dd) in enumerate(parsed):
                    ids = [lookup[canonical_smiles(s)] for s in ss]
                    drugs[i, :len(ids)] = features["drug_embeddings"][ids]
                    doses[i, :len(ids)] = torch.tensor(dd)
                    mask[i, :len(ids)] = True
                numeric_ids = pd.to_numeric(rows.baseline_index, errors="raise").to_numpy(dtype=float)
                if not np.isfinite(numeric_ids).all() or not np.equal(numeric_ids, np.floor(numeric_ids)).all():
                    raise ValueError("baseline_index must be finite integers")
                ids = numeric_ids.astype(np.int64)
                if (ids < 0).any() or (ids >= len(baseline)).any():
                    raise IndexError("baseline_index out of bounds")
                x0 = torch.from_numpy(np.array(baseline[ids], dtype=np.float32, copy=True))
                if not torch.isfinite(x0).all():
                    raise ValueError("Nonfinite baseline expression")
                yield to_device({"x0": x0, "drug_embeddings": drugs, "doses_um": doses,
                                 "drug_mask": mask}, device)
        batches = request_batches()
    if len(metadata) == 0:
        raise ValueError("No prediction samples")
    delta = np.lib.format.open_memmap(output / "delta.npy", mode="w+", dtype="float32", shape=(len(metadata), 978))
    expression = np.lib.format.open_memmap(output / "expression.npy", mode="w+", dtype="float32", shape=delta.shape)
    offset = 0
    with torch.inference_mode():
        for batch in batches:
            result = forward(model, batch)
            n = len(batch["x0"])
            delta[offset:offset+n] = result["delta"].cpu().numpy()
            expression[offset:offset+n] = result["expression"].cpu().numpy()
            offset += n
    delta.flush()
    expression.flush()
    metadata.to_csv(output / "metadata.csv", index=False)
    pd.DataFrame({"gene_id": saved["gene_ids"], "gene_symbol": saved["gene_symbols"]}).to_csv(output / "genes.csv", index=False)
    save_json(output / "provenance.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
                                          "time_hours": saved["time_hours"],
                                          "expression": "x0 + b(x0) + sum_k g(x0, drug_k, dose_k)"})


if __name__ == "__main__":
    main()
