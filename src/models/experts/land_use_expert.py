import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        output = self.out(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class LandUseExpert(nn.Module):
    """
    Transformer-based expert for contextual and land-use pattern modeling.
    Supports dynamic adaptation if input feature dimension varies.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(LandUseExpert, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        Returns:
            torch.Tensor: [batch_size, hidden_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        actual_input_dim = x.shape[-1]
        if actual_input_dim != self.input_dim:
            if not hasattr(self, '_adapted_embedding') or self._adapted_embedding.in_features != actual_input_dim:
                device = next(self.parameters()).device
                self._adapted_embedding = nn.Linear(actual_input_dim, self.hidden_dim).to(device)
                nn.init.xavier_uniform_(self._adapted_embedding.weight)
                nn.init.zeros_(self._adapted_embedding.bias)
            x = self._adapted_embedding(x)
        else:
            x = self.embedding(x)
            
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = torch.mean(x, dim=1)  # Sequence mean pooling
        output = self.output_layer(x)
        return output
