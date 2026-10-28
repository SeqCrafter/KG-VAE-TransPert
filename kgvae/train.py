"""Single-device PyTorch training with AMP, accumulation, early stopping and resume."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .common import load_torch, save_json, seed_all, sha256
from .data import PerturbationDataset, make_loader
from .losses import objective
from .model import KGVAEPerturbation


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def forward(model, batch):
    return model(batch["x0"], batch["drug_embeddings"], batch["doses_um"], batch["drug_mask"])


def amp_context(device, amp):
    return torch.autocast(device_type=device.type, enabled=device.type == "cuda" and amp != "off",
                          dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def run_epoch(model, loader, device, loss_config, amp="off", optimizer=None,
              scaler=None, accumulation_steps=1, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    totals, count = {}, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, raw in enumerate(loader):
            batch = to_device(raw, device)
            with amp_context(device, amp):
                outputs = forward(model, batch)
            # Correlation, KL and all reductions are evaluated in float32.
            metrics = objective(outputs, batch["x0"], batch["xp"], **loss_config)
            if not torch.isfinite(metrics["loss"]):
                raise FloatingPointError(f"Nonfinite loss at batch {step}")
            n = len(batch["x0"])
            if training:
                window_start = (step // accumulation_steps) * accumulation_steps
                window_end = min(window_start + accumulation_steps, len(loader))
                # Sample-weighted accumulation, including an incomplete last batch/window.
                window_n = min(window_end * loader.batch_size, len(loader.dataset)) - window_start * loader.batch_size
                scaler.scale(metrics["loss"] * n / window_n).backward()
                if step + 1 == window_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            count += n
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * n
    if count == 0:
        raise ValueError("Empty dataloader")
    return {key: value / count for key, value in totals.items()}


def rng_state(loader):
    state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "loader": loader.generator.get_state()}


def restore_rng(state, loader):
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    s = state["numpy"]
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), s[2], s[3], s[4]))
    loader.generator.set_state(state["loader"])


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--resume", help="Path to last.pt; resume at an epoch boundary")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_all(cfg["seed"])
    tc, dc = cfg["train"], cfg["data"]
    device = torch.device(tc["device"])
    if tc["amp"] not in ("off", "fp16", "bf16"):
        raise ValueError("amp must be off, fp16, or bf16")
    if device.type == "cuda" and tc["amp"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support bf16; set amp: fp16 or off")
    if tc["epochs"] < 1 or tc["accumulation_steps"] < 1 or dc["batch_size"] < 1:
        raise ValueError("epochs, accumulation_steps and batch_size must be positive")
    output = Path(tc["output"])
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError("Run already exists; use --resume or another output directory")
    features = load_torch(dc["features"])
    feature_hash = sha256(dc["features"])
    split = json.loads(Path(dc["split"]).read_text())
    if split["metadata_sha256"] != sha256(Path(dc["root"]) / "metadata.csv"):
        raise ValueError("Split belongs to different metadata")
    partitions = [set(split[name]) for name in ("train", "val", "test")]
    if any(not x for x in partitions) or any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError("Empty or overlapping partitions")
    datasets = {name: PerturbationDataset(dc["root"], features, split[name]) for name in ("train", "val")}
    if any(len(x) != 1 for ds in datasets.values() for x in ds.drugs):
        raise ValueError("This training protocol uses single-drug samples only")
    times = pd.to_numeric(datasets["train"].metadata.time_hours, errors="raise")
    if not np.isfinite(times).all() or times.nunique() != 1:
        raise ValueError("A model without time input must be trained on one fixed time point")
    loaders = {name: make_loader(ds, dc["batch_size"], dc["num_workers"], name == "train", cfg["seed"],
                                device.type == "cuda") for name, ds in datasets.items()}
    model = KGVAEPerturbation(features["gene_embeddings"], features["drug_embeddings"].shape[1], **cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and tc["amp"] == "fp16")
    start, best, bad_epochs = 0, math.inf, 0
    if args.resume:
        saved = load_torch(args.resume)
        if saved["feature_sha256"] != feature_hash or saved["split"] != split:
            raise ValueError("Resume features/split differ from checkpoint")
        for key in ("model", "loss", "data", "seed"):
            if saved["config"][key] != cfg[key]:
                raise ValueError(f"Resume config differs: {key}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start, best, bad_epochs = saved["epoch"] + 1, saved["best"], saved["bad_epochs"]
        restore_rng(saved["rng"], loaders["train"])
        # Copy the historical best into a new output directory, if needed.
        best_path = Path(args.resume).parent / "best.pt"
        if output.resolve() != Path(args.resume).parent.resolve() and best_path.exists():
            import shutil
            shutil.copy2(best_path, output / "best.pt")
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for epoch in range(start, tc["epochs"]):
        train_metrics = run_epoch(model, loaders["train"], device, cfg["loss"], tc["amp"], optimizer,
                                  scaler, tc["accumulation_steps"], tc["grad_clip"])
        val_metrics = run_epoch(model, loaders["val"], device, cfg["loss"], tc["amp"])
        scheduler.step(val_metrics["loss"])
        improved = val_metrics["loss"] < best
        best, bad_epochs = (val_metrics["loss"], 0) if improved else (best, bad_epochs + 1)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
                  "lr": optimizer.param_groups[0]["lr"]}
        print(json.dumps(record), flush=True)
        with open(output / "history.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "epoch": epoch, "best": best, "bad_epochs": bad_epochs,
                   "config": cfg, "split": split, "rng": rng_state(loaders["train"]),
                   "feature_sha256": feature_hash, "feature_provenance": features["provenance"],
                   "gene_ids": features["gene_ids"], "gene_symbols": features["gene_symbols"],
                   "drug_dim": features["drug_embeddings"].shape[1], "time_hours": float(times.iloc[0])}
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", payload)
        if tc["patience"] > 0 and bad_epochs >= tc["patience"]:
            break
    save_json(output / "training_summary.json", {"best_validation_loss": best,
                                               "selection": "validation loss; test set not used"})


if __name__ == "__main__":
    main()
