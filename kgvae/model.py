"""K-Residual: knowledge-guided, dose-aware conditional residual model.

delta_hat = b(x0) + sum_k g(x0, drug_k, dose_k)

The cell enters only through its baseline expression x0 (as in WAVE). A drug is
described only by its frozen MAP-KG molecular embedding, computed from SMILES, and
enters g as a single token, as in MAP: at the full MAP-KG width, with no bottleneck,
dropout or input noise, so none of the knowledge that generalization to new drugs
relies on is discarded. The model width therefore equals the MAP-KG width. Dose is
the one addition: a FiLM modulation of the drug token, the identity at initialization.
"""
import torch
from torch import nn


def mlp(input_dim, hidden_dim, output_dim, dropout=0.0):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden_dim, output_dim))


class CrossAttentionBlock(nn.Module):
    """Gene queries attend to the [drug, cell] memory; no gene-gene self-attention.

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

    def forward(self, tokens, memory):
        update, weights = self.attention(self.query_norm(tokens), memory, memory,
                                         need_weights=True, average_attn_weights=True)
        tokens = tokens + self.dropout(update)
        tokens = tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))
        return tokens, weights


class KResidual(nn.Module):
    def __init__(self, gene_embeddings, drug_dim, n_reference_cells, nhead=8, num_layers=2,
                 dim_feedforward=2048, dropout=0.1, expression_noise=0.1, shared_hidden=256,
                 shared_dropout=0.3):
        super().__init__()
        n_genes, kg_dim = gene_embeddings.shape
        if kg_dim != drug_dim:
            raise ValueError("MAP-KG drug and gene embeddings must share one space (same width)")
        d_model = drug_dim
        if d_model % nhead or num_layers < 1 or n_reference_cells < 1:
            raise ValueError("Require MAP-KG width divisible by nhead, num_layers >= 1 and n_reference_cells >= 1")
        self.register_buffer("gene_embeddings", gene_embeddings.detach().float().clone())
        # Training-set statistics: set once by train.py, afterwards restored from state_dict.
        self.register_buffer("expr_mean", torch.zeros(n_genes))
        self.register_buffer("expr_std", torch.ones(n_genes))
        self.register_buffer("drug_mean", torch.zeros(drug_dim))
        self.register_buffer("drug_std", torch.ones(drug_dim))
        self.register_buffer("dose_stats", torch.tensor([0., 1.]))  # mean, std of log10(dose / uM)
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
        self.gene_projector = nn.Linear(kg_dim, d_model)
        self.expression_encoder = mlp(1, d_model, d_model)
        self.cell_encoder = mlp(n_genes, d_model, d_model, dropout)
        self.dose_film = mlp(1, d_model, 2 * d_model)
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model, nhead, dim_feedforward, dropout)
                                     for _ in range(num_layers)])
        self.readout = mlp(2 * d_model, d_model, 1, dropout)
        # Identity FiLM and g ~ 0 at initialization. Readouts are small rather than
        # exactly zero, so per-sample Pearson is defined at step 0.
        nn.init.zeros_(self.dose_film[-1].weight)
        nn.init.zeros_(self.dose_film[-1].bias)
        for layer in (self.readout[-1], self.shared[-1][-1]):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)

    @torch.no_grad()
    def set_statistics(self, expr_mean, expr_std, drug_mean, drug_std, dose_mean, dose_std,
                       residual_std, reference_x0, reference_delta):
        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.expr_mean.device)
        drug_std = tensor(drug_std)
        self.expr_mean.copy_(tensor(expr_mean))
        self.expr_std.copy_(tensor(expr_std).clamp_min(1e-2))
        self.drug_mean.copy_(tensor(drug_mean))
        self.drug_std.copy_(torch.where(drug_std > 1e-6, drug_std, torch.ones_like(drug_std)))
        # A constant dose (e.g. the WAVE single-dose setting) gives zero std.
        self.dose_stats.copy_(tensor([dose_mean, dose_std if dose_std > 1e-6 else 1.]))
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

    def drug_token(self, embeddings, dose):
        # Per-dimension standardization is invertible: it rescales, never discards, MAP-KG features.
        x = (embeddings - self.drug_mean) / self.drug_std
        gamma, beta = self.dose_film(dose[:, None]).chunk(2, dim=-1)
        return x * (1 + gamma) + beta

    def drug_effect(self, gene_tokens, cell, drug):
        """g(x0, d, dose): one single-drug pass, identical in training and combination inference."""
        memory = torch.stack((drug, cell), dim=1)
        tokens, drug_attention = gene_tokens, 0.
        for block in self.blocks:
            tokens, weights = block(tokens, memory)
            drug_attention = drug_attention + weights[..., 0]
        effect = self.readout(torch.cat((tokens, gene_tokens), dim=-1)).squeeze(-1)
        return effect, drug_attention / len(self.blocks)

    def forward(self, x0, drug_embeddings, doses_um, drug_mask=None):
        if drug_embeddings.ndim == 2:
            drug_embeddings = drug_embeddings[:, None]
            doses_um = doses_um.reshape(-1, 1)
        if drug_mask is None:
            drug_mask = torch.ones_like(doses_um, dtype=torch.bool)
        batch, count, _ = drug_embeddings.shape
        if doses_um.shape != (batch, count) or drug_mask.shape != (batch, count):
            raise ValueError("Expected dose and mask shapes [batch, number_of_drugs]")
        if not torch.isfinite(doses_um).all() or (doses_um < 0).any():
            raise ValueError("Doses must be finite and >= 0")
        expression = self.standardize(x0)
        if self.training and self.expression_noise > 0:
            expression = expression + torch.randn_like(expression) * self.expression_noise
        gene_tokens = self.gene_projector(self.gene_embeddings) + self.expression_encoder(expression[..., None])
        cell = self.cell_encoder(expression)
        effect = torch.zeros(x0.shape, device=x0.device)
        drug_attention = torch.zeros_like(effect)
        drugs = torch.zeros(batch, device=x0.device)
        for i in range(count):
            # Zero dose is treated as an absent drug; log10 is applied only to active entries.
            indices = (drug_mask[:, i] & (doses_um[:, i] > 0)).nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            # log10(dose) separates 0.04-10 uM far better than log10(dose + 1).
            dose = (torch.log10(doses_um[indices, i]) - self.dose_stats[0]) / self.dose_stats[1]
            drug = self.drug_token(drug_embeddings[indices, i], dose)
            g, attention = self.drug_effect(gene_tokens[indices], cell[indices], drug)
            # Autocast may produce lower-precision outputs; accumulate in float32.
            effect = effect.index_add(0, indices, g.float())
            drug_attention = drug_attention.index_add(0, indices, attention.float())
            drugs = drugs.index_add(0, indices, torch.ones(len(indices), device=x0.device))
        treated = drugs > 0
        # The shared response is counted once, however many drugs are combined; untreated -> zero.
        shared = self.shared(expression).float() * treated[:, None]
        delta = shared + effect
        return {"delta": delta, "expression": x0 + delta, "shared": shared, "drug_effect": effect,
                "reference": self.reference(x0), "treated": treated,
                "drug_attention": drug_attention / drugs.clamp_min(1)[:, None]}
