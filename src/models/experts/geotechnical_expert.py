import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class PyGCompatibleGCNConv(nn.Module):
    """GCN Layer with parameter names matching PyTorch Geometric GCNConv ('lin.weight', 'bias')."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PyGCompatibleGCNConv, self).__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        adj = torch.eye(num_nodes, device=x.device)
        if edge_index.size(1) > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            adj[edge_index[1], edge_index[0]] = 1.0
        
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        support = self.lin(x)
        output = torch.matmul(norm_adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class GeotechnicalExpert(nn.Module):
    """
    Graph Neural Network expert for geotechnical relationships & terrain modeling.
    Supports both PyTorch Geometric and standalone PyTorch fallback with identical state_dict key signatures.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super(GeotechnicalExpert, self).__init__()
        self.use_pyg = HAS_PYG
        
        if HAS_PYG:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = PyGCompatibleGCNConv(input_dim, hidden_dim)
            self.conv2 = PyGCompatibleGCNConv(hidden_dim, hidden_dim)
            self.conv3 = PyGCompatibleGCNConv(hidden_dim, hidden_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor, batch: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        edge_index = self.create_graph_from_data(coords, threshold)

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = self.conv3(x, edge_index)

        if HAS_PYG and self.use_pyg:
            x = global_mean_pool(x, batch)
        else:
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            pooled = []
            for b in range(batch_size):
                mask = (batch == b)
                if mask.sum() > 0:
                    pooled.append(x[mask].mean(dim=0))
                else:
                    pooled.append(torch.zeros(x.size(1), device=x.device))
            x = torch.stack(pooled, dim=0)

        x = self.mlp(x)
        return x

    @staticmethod
    def create_graph_from_data(coords: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)
        num_nodes = coords.shape[0]
        if num_nodes <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=coords.device)
            
        dists = torch.cdist(coords, coords, p=2)
        edges = (dists < threshold).nonzero(as_tuple=False).T
        mask = edges[0] != edges[1]
        edge_index = edges[:, mask]
        return edge_index
