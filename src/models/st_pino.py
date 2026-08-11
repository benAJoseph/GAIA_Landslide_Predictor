import torch
import torch.nn as nn
from src.models.experts.hydrological_expert import HydrologicalExpert
from src.models.experts.geotechnical_expert import GeotechnicalExpert
from src.models.experts.land_use_expert import LandUseExpert
from src.models.physics_layer import PhysicsFactorOfSafetyLayer

class STPINOModel(nn.Module):
    """
    Spatio-Temporal Physics-Informed Neural Operator (ST-PINO) with Attention Gating.
    Combines:
    - Hydrological Expert (Bi-LSTM)
    - Geotechnical Expert (GCN)
    - Land Use Expert (Transformer)
    - Differentiable Physics Layer (Factor of Safety FOS)
    """
    def __init__(self, hydro_dim: int = 6, geo_dim: int = 3, land_use_dim: int = 91, hidden_dim: int = 64):
        super(STPINOModel, self).__init__()
        self.hydro_expert = HydrologicalExpert(hydro_dim, hidden_dim)
        self.geo_expert = GeotechnicalExpert(geo_dim, hidden_dim, hidden_dim)
        self.land_use_expert = LandUseExpert(land_use_dim, hidden_dim)
        self.physics_layer = PhysicsFactorOfSafetyLayer()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Binary classification: [No Landslide, Landslide]
        )

    def forward(self, hydro_data: torch.Tensor, geo_features: torch.Tensor, geo_coords: torch.Tensor, land_use_data: torch.Tensor, soil_params: dict):
        batch_size = hydro_data.size(0)
        batch_tensor = torch.arange(batch_size, device=geo_features.device)

        hydro_embeddings = self.hydro_expert(hydro_data)
        geo_embeddings = self.geo_expert(geo_features, geo_coords, batch_tensor)
        land_use_embeddings = self.land_use_expert(land_use_data)

        fos = self.physics_layer(
            soil_params['cohesion'],
            soil_params['slope_angle'],
            soil_params['soil_moisture'],
            soil_params['soil_type_features']
        )

        combined_embeddings = torch.cat([hydro_embeddings, geo_embeddings, land_use_embeddings], dim=1)
        logits = self.classifier(combined_embeddings)

        return {'logits': logits, 'fos': fos}
