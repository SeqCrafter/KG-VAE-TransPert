#!/usr/bin/env bash
# Run from the repository root after preparing raw data and external weights.
set -euo pipefail
MAP_KG_CKPT="${MAP_KG_CKPT:-weights/mapkg/mapkg_encoder_v3.pt}"
MAP_VOCAB="${MAP_VOCAB:-weights/mapkg/bart_vocab.txt}"
ESM_DICT="${ESM_DICT:-weights/mapkg/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt}"

python -m kgvae.preprocess cmap \
  --control-gctx data/raw/level3_beta_ctl_n188708x12328.gctx \
  --case-gctx data/raw/level3_beta_trt_cp_n1805898x12328.gctx \
  --gene-info data/raw/geneinfo_beta.txt \
  --inst-info data/raw/instinfo_beta.txt \
  --pert-info data/raw/compoundinfo_beta.txt \
  --min-replicates 2 --min-compounds 50 \
  --time-hours 24 --output data/processed

python -m kgvae.split --method drug
python -m kgvae.export_mapkg \
  --checkpoint "$MAP_KG_CKPT" --vocab "$MAP_VOCAB" --esm-dict "$ESM_DICT"
python -m kgvae.train --config configs/default.yaml
