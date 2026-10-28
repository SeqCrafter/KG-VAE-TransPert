import torch.nn as nn

class ResidualProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        transformed = self.mlp(x)
        residual = self.residual_proj(x)
        gate_value = self.gate(x)
        
        return gate_value * transformed + (1 - gate_value) * residual


