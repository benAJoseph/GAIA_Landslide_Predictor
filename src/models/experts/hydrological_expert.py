import torch
import torch.nn as nn
import torch.nn.functional as F

class HydrologicalExpert(nn.Module):
    """
    LSTM-based expert model for hydrological patterns in ST-PINO.
    Processes sequential weather/rain/soil-moisture data to capture temporal trends.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(HydrologicalExpert, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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
            x = x.unsqueeze(1)  # Add seq_len dimension if missing
            
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Concatenate last hidden state from both directions
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.output_layer(last_hidden)
        
        return output
