"""Architecture from 模型架构.md. Frozen features enter as tensors, not trainable encoders."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .vendor.wave_model import GeneVAE
from .vendor.projector import ResidualProjector


class KGVAEPerturbation(nn.Module):
    def __init__(self, gene_embeddings, drug_dim, latent_dim=256, hidden_dim1=512,
                 hidden_dim2=256, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.1, dose_reference_um=1.0,
                 activation_checkpointing=False):
        super().__init__()
        if dose_reference_um <= 0 or d_model % nhead:
            raise ValueError("Positive dose reference and d_model divisible by nhead required")
        self.register_buffer("gene_embeddings", gene_embeddings.detach().float().clone())
        self.vae = GeneVAE(gene_embeddings.shape[0], latent_dim, hidden_dim1, hidden_dim2)
        self.cell_projector = ResidualProjector(latent_dim, d_model)
        self.drug_projector = ResidualProjector(drug_dim, d_model)
        self.gene_projector = ResidualProjector(gene_embeddings.shape[1], d_model)
        # MAP's bidirectional cell/drug/gene token fusion, expressed in native PyTorch.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.shift_head = nn.Linear(d_model, latent_dim)
        self.dose_reference_um = dose_reference_um
        self.activation_checkpointing = activation_checkpointing

    def encode_baseline(self, x0, sample=None):
        encoded = self.vae.encoder(x0)
        mu = self.vae.fc_mu(encoded)
        logvar = self.vae.fc_logvar(encoded).float().clamp(-20, 10)
        sample = self.training if sample is None else sample
        z0 = self.vae.reparameterize(mu, logvar) if sample else mu
        return z0, mu, logvar

    def latent_shift(self, z0, drug_embeddings, doses_um, drug_mask):
        """Each drug sees the SAME z0. Sum shifts only AFTER independent attention."""
        batch, count, _ = drug_embeddings.shape
        if doses_um.shape != (batch, count) or drug_mask.shape != (batch, count):
            raise ValueError("Expected dose and mask shapes [batch, number_of_drugs]")
        if not torch.isfinite(doses_um).all() or (doses_um < 0).any():
            raise ValueError("Doses must be finite and >= 0")
        gene_tokens = self.gene_projector(self.gene_embeddings)
        cell_tokens = self.cell_projector(z0)
        total = torch.zeros_like(z0)
        for i in range(count):
            # log1p(c / 1 uM): well-defined at zero, unlike log10(c).
            active = drug_mask[:, i] & (doses_um[:, i] > 0)
            indices = active.nonzero(as_tuple=True)[0]
            if indices.numel() == 0:
                continue
            scale = torch.log1p(doses_um[indices, i] / self.dose_reference_um)
            drug = self.drug_projector(drug_embeddings[indices, i]) * scale[:, None]
            tokens = torch.cat((cell_tokens[indices, None], drug[:, None],
                                gene_tokens[None].expand(len(indices), -1, -1)), dim=1)
            for layer in self.layers:
                if self.activation_checkpointing and self.training:
                    tokens = checkpoint(layer, tokens, use_reentrant=False)
                else:
                    tokens = layer(tokens)
            shift = self.shift_head(self.output_norm(tokens[:, 0]))
            total = total.index_add(0, indices, shift)
        return total

    def forward(self, x0, drug_embeddings, doses_um, drug_mask=None, sample=None):
        if drug_embeddings.ndim == 2:
            drug_embeddings = drug_embeddings[:, None]
            doses_um = doses_um.reshape(-1, 1)
        if drug_mask is None:
            drug_mask = torch.ones_like(doses_um, dtype=torch.bool)
        z0, mu, logvar = self.encode_baseline(x0, sample)
        dz = self.latent_shift(z0, drug_embeddings, doses_um, drug_mask)
        # Reuse this exact z0 and decoder evaluation for the baseline anchor.
        reconstruction = self.vae.decoder(z0)
        delta = self.vae.decoder(z0 + dz) - reconstruction
        return {"delta": delta, "reconstruction": reconstruction,
                "expression": x0 + delta, "mu": mu, "logvar": logvar,
                "z0": z0, "dz": dz}
