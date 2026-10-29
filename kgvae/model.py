"""K-Residual: knowledge-guided, dose-aware conditional residual model (模型架构.md).

delta_hat = b(x0) + sum_k g(x0, drug_k, dose_k)

The cell enters only through its baseline expression x0 (as in WAVE). The model is
built to generalize across drugs: a drug is described only by frozen MAP-KG
knowledge computed from its SMILES, never by an identity, and that knowledge
enters g through three dose-modulated routes:

  1. a drug token (MAP-KG molecular embedding, FiLM-modulated by log dose);
  2. drug-target tokens: the drug's top-K proteome-wide predicted targets in the
     shared MAP-KG drug-gene space, so knowledge of targets OUTSIDE the 978
     landmark genes (most drug targets) can reach the landmark responses;
  3. a per-landmark-gene affinity: cosine(drug, gene) in that same space, fed into
     each gene token so genes the drug is predicted to engage are marked directly.
"""
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, hidden_dim, output_dim, dropout=0.0):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden_dim, output_dim))


class CrossAttentionBlock(nn.Module):
    """Gene queries attend to [drug, targets..., cell] memory; no gene-gene self-attention.

    Only the queries are layer-normalized. The memory never passes through a
    LayerNorm, since LN(s * v) == LN(v) would erase the FiLM dose modulation.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = mlp(d_model, dim_feedforward, d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, memory, memory_mask=None):
        update, weights = self.attention(self.query_norm(tokens), memory, memory, key_padding_mask=memory_mask,
                                         need_weights=True, average_attn_weights=True)
        tokens = tokens + self.dropout(update)
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens, weights


class KResidual(nn.Module):
    def __init__(self, gene_embeddings, drug_dim, n_reference_cells, d_model=256, nhead=8,
                 num_layers=2, dim_feedforward=512, dropout=0.1, drug_bottleneck=64,
                 drug_dropout=0.3, drug_noise=0.1, target_dropout=0.2, expression_noise=0.1,
                 shared_hidden=256, shared_dropout=0.3):
        super().__init__()
        if d_model % nhead or num_layers < 1 or n_reference_cells < 1:
            raise ValueError("Require d_model divisible by nhead, num_layers >= 1 and n_reference_cells >= 1")
        n_genes, kg_dim = gene_embeddings.shape
        if kg_dim != drug_dim:
            raise ValueError("MAP-KG drug and gene embeddings must share one space (same width)")
        self.register_buffer("gene_embeddings", gene_embeddings.detach().float().clone())
        # Training-set statistics: set once by train.py, afterwards restored from state_dict.
        self.register_buffer("expr_mean", torch.zeros(n_genes))
        self.register_buffer("expr_std", torch.ones(n_genes))
        self.register_buffer("drug_mean", torch.zeros(drug_dim))
        self.register_buffer("drug_std", torch.ones(drug_dim))
        self.register_buffer("dose_stats", torch.tensor([0., 1.]))  # mean, std of log10(dose / uM)
        self.register_buffer("affinity_stats", torch.tensor([0., 1.]))  # mean, std of landmark cosine
        self.register_buffer("target_stats", torch.tensor([0., 1.]))  # mean, std of top-K target cosine
        self.register_buffer("residual_std", torch.ones(n_genes))
        # Evaluation-only reference: training cell lines' x0 and mean delta. Never used for prediction.
        self.register_buffer("reference_x0", torch.zeros(n_reference_cells, n_genes))
        self.register_buffer("reference_delta", torch.zeros(n_reference_cells, n_genes))
        # Few distinct x0 exist in LINCS (one DMSO mean per cell line). Input noise, like WAVE's
        # VAE sampling and MAP's control-set resampling, smooths the x0 -> response map.
        self.expression_noise = expression_noise
        # b(x0): drug-blind shared response, deliberately small and strongly regularized.
        self.shared = nn.Sequential(nn.Dropout(shared_dropout), mlp(n_genes, shared_hidden, n_genes, shared_dropout))
        # Dynamic gene tokens: MAP-KG gene knowledge + this sample's baseline expression (MAP phi_fuse).
        # Shared with target tokens, so landmark and off-landmark genes live in one token space.
        self.gene_projector = nn.Linear(kg_dim, d_model)
        self.expression_encoder = mlp(1, d_model, d_model)
        # Route 3: [drug-gene affinity, dose] -> per-gene update. Dose enters so engagement scales with it.
        self.affinity_encoder = mlp(2, d_model, d_model)
        self.cell_encoder = mlp(n_genes, d_model, d_model, dropout)
        # Route 1. Narrow bottleneck plus input noise/dropout limit memorization of training drug identities.
        self.drug_noise = drug_noise
        self.drug_dropout = nn.Dropout(drug_dropout)
        self.drug_encoder = nn.Sequential(nn.Linear(drug_dim, drug_bottleneck), nn.GELU(),
                                          nn.Linear(drug_bottleneck, d_model), nn.LayerNorm(d_model))
        self.dose_film = mlp(1, d_model, 2 * d_model)
        # Route 2: target token = gene knowledge + [score, dose] encoding, FiLM by dose after LayerNorm.
        self.target_score_encoder = mlp(2, d_model, d_model)
        self.target_norm = nn.LayerNorm(d_model)
        self.target_film = mlp(1, d_model, 2 * d_model)
        # Whole target tokens are dropped in training, so no single predicted target is relied on.
        self.target_dropout = target_dropout
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.readout = mlp(2 * d_model, d_model, 1, dropout)
        # Identity FiLM, zero affinity update and g ~ 0 at initialization. Readouts are small
        # rather than exactly zero, so per-sample Pearson is defined at step 0.
        for layer in (self.dose_film[-1], self.target_film[-1], self.affinity_encoder[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in (self.readout[-1], self.shared[-1][-1]):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def set_statistics(self, expr_mean, expr_std, drug_mean, drug_std, dose_mean, dose_std,
                       affinity_mean, affinity_std, target_mean, target_std,
                       residual_std, reference_x0, reference_delta):
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.expr_mean.device)

        def scale(value):
            # A constant feature (e.g. the WAVE single-dose setting) gives zero std.
            return value if value > 1e-6 else 1.
        drug_std = tensor(drug_std)
        self.expr_mean.copy_(tensor(expr_mean))
        self.expr_std.copy_(tensor(expr_std).clamp_min(1e-2))
        self.drug_mean.copy_(tensor(drug_mean))
        self.drug_std.copy_(torch.where(drug_std > 1e-6, drug_std, torch.ones_like(drug_std)))
        self.dose_stats.copy_(tensor([dose_mean, scale(dose_std)]))
        self.affinity_stats.copy_(tensor([affinity_mean, scale(affinity_std)]))
        self.target_stats.copy_(tensor([target_mean, scale(target_std)]))
        self.residual_std.copy_(tensor(residual_std).clamp_min(1e-3))
        self.reference_x0.copy_(tensor(reference_x0))
        self.reference_delta.copy_(tensor(reference_delta))

    def standardize(self, x0):
        return (x0 - self.expr_mean) / self.expr_std

    @torch.no_grad()
    def reference(self, x0):
        """Mean training delta of the nearest training cell line in standardized x0 space.

        For a training cell line this is its own mean (distance zero); for an unseen one,
        the most similar known cell line. A drug-blind yardstick, not part of the prediction.
        """
        distance = torch.cdist(self.standardize(x0.float()), self.standardize(self.reference_x0))
        return self.reference_delta[distance.argmin(-1)]

    def landmark_affinity(self, drug_embeddings):
        """Route 3 input: cosine(drug, landmark gene) in the shared MAP-KG space, standardized."""
        cosine = F.normalize(drug_embeddings.float(), dim=-1) @ F.normalize(self.gene_embeddings, dim=-1).T
        return (cosine - self.affinity_stats[0]) / self.affinity_stats[1]

    def drug_token(self, embeddings, dose):
        x = (embeddings - self.drug_mean) / self.drug_std
        if self.training and self.drug_noise > 0:
            x = x + torch.randn_like(x) * self.drug_noise
        h = self.drug_encoder(self.drug_dropout(x))
        gamma, beta = self.dose_film(dose[:, None]).chunk(2, dim=-1)
        return h * (1 + gamma) + beta

    def target_tokens(self, target_embeddings, target_scores, dose):
        score = (target_scores - self.target_stats[0]) / self.target_stats[1]
        inputs = torch.stack((score, dose[:, None].expand_as(score)), dim=-1)
        h = self.target_norm(self.gene_projector(target_embeddings) + self.target_score_encoder(inputs))
        gamma, beta = self.target_film(dose[:, None]).chunk(2, dim=-1)
        return h * (1 + gamma[:, None]) + beta[:, None]

    def drug_effect(self, gene_tokens, cell, drug, targets):
        """g(x0, d, dose): one single-drug pass, identical in training and combination inference."""
        memory = torch.cat((drug[:, None], targets, cell[:, None]), dim=1)
        mask = None
        if self.training and self.target_dropout > 0:
            # True = ignored. Drug and cell tokens are never dropped, so no row is fully masked.
            mask = torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device)
            mask[:, 1:-1] = torch.rand(targets.shape[:2], device=memory.device) < self.target_dropout
        tokens, drug_attention, target_attention = gene_tokens, 0., 0.
        for block in self.blocks:
            tokens, weights = block(tokens, memory, mask)
            drug_attention = drug_attention + weights[..., 0]
            target_attention = target_attention + weights[..., 1:-1].sum(-1)
        effect = self.readout(torch.cat((tokens, gene_tokens), dim=-1)).squeeze(-1)
        n = len(self.blocks)
        return effect, drug_attention / n, target_attention / n

    def forward(self, x0, drug_embeddings, doses_um, target_embeddings, target_scores, drug_mask=None):
        if drug_embeddings.ndim == 2:
            drug_embeddings = drug_embeddings[:, None]
            doses_um = doses_um.reshape(-1, 1)
            target_embeddings, target_scores = target_embeddings[:, None], target_scores[:, None]
        if drug_mask is None:
            drug_mask = torch.ones_like(doses_um, dtype=torch.bool)
        batch, count, _ = drug_embeddings.shape
        if doses_um.shape != (batch, count) or drug_mask.shape != (batch, count):
            raise ValueError("Expected dose and mask shapes [batch, number_of_drugs]")
        if target_scores.shape[:2] != (batch, count) or target_embeddings.shape[:3] != target_scores.shape:
            raise ValueError("Expected target shapes [batch, drugs, K, dim] and [batch, drugs, K]")
        if not torch.isfinite(doses_um).all() or (doses_um < 0).any():
            raise ValueError("Doses must be finite and >= 0")
        expression = self.standardize(x0)
        if self.training and self.expression_noise > 0:
            expression = expression + torch.randn_like(expression) * self.expression_noise
        base_tokens = self.gene_projector(self.gene_embeddings) + self.expression_encoder(expression[..., None])
        cell = self.cell_encoder(expression)
        effect = torch.zeros(x0.shape, device=x0.device)
        drug_attention = torch.zeros_like(effect)
        target_attention = torch.zeros_like(effect)
        drugs = torch.zeros(batch, device=x0.device)
        for i in range(count):
            # Zero dose is treated as an absent drug; log10 is applied only to active entries.
            indices = (drug_mask[:, i] & (doses_um[:, i] > 0)).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            # log10(dose) separates 0.04-10 uM far better than log10(dose + 1).
            dose = (torch.log10(doses_um[indices, i]) - self.dose_stats[0]) / self.dose_stats[1]
            embeddings = drug_embeddings[indices, i]
            affinity = self.landmark_affinity(embeddings)
            route3 = torch.stack((affinity, dose[:, None].expand_as(affinity)), dim=-1)
            gene_tokens = base_tokens[indices] + self.affinity_encoder(route3)
            drug = self.drug_token(embeddings, dose)
            targets = self.target_tokens(target_embeddings[indices, i], target_scores[indices, i], dose)
            g, a_drug, a_target = self.drug_effect(gene_tokens, cell[indices], drug, targets)
            # Autocast may produce lower-precision outputs; accumulate in float32.
            effect = effect.index_add(0, indices, g.float())
            drug_attention = drug_attention.index_add(0, indices, a_drug.float())
            target_attention = target_attention.index_add(0, indices, a_target.float())
            drugs = drugs.index_add(0, indices, torch.ones(len(indices), device=x0.device))
        treated = drugs > 0
        # The shared response is counted once, however many drugs are combined; untreated -> zero.
        shared = self.shared(expression).float() * treated[:, None]
        delta = shared + effect
        per_drug = drugs.clamp_min(1)[:, None]
        return {"delta": delta, "expression": x0 + delta, "shared": shared, "drug_effect": effect,
                "reference": self.reference(x0), "treated": treated,
                "drug_attention": drug_attention / per_drug, "target_attention": target_attention / per_drug}
