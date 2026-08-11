import torch
import torch.nn as nn

class PhysicsGuidedAttention(nn.Module):
    """
    Physics-Informed Attention Gating Module.
    Dynamically modulates data-driven embeddings with the physical Factor of Safety (FOS).
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.physics_gate = nn.Sequential(nn.Linear(1, hidden_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, fos: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        attention_weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
        output = attention_weights @ v
        
        if fos.dim() == 1:
            fos = fos.unsqueeze(-1)
        physics_weights = self.physics_gate(fos)
        return self.layer_norm(physics_weights * output + (1 - physics_weights) * x)
