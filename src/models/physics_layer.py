import torch
import torch.nn as nn

class PhysicsFactorOfSafetyLayer(nn.Module):
    """
    Differentiable implementation of the Factor of Safety (FOS) equation:
    FOS = (c' + (γh cos²β - u) tanφ') / (γh sinβ cosβ)
    
    Grounds neural network predictions in physical slope stability laws.
    """
    def __init__(self):
        super().__init__()
        self.gamma_h = nn.Parameter(torch.tensor(90.0))
        self.min_cohesion = nn.Parameter(torch.tensor(0.0))
        self.max_cohesion = nn.Parameter(torch.tensor(50.0))
        self.min_friction = nn.Parameter(torch.tensor(20.0))
        self.max_friction = nn.Parameter(torch.tensor(45.0))

    def forward(self, cohesion: torch.Tensor, slope_angle: torch.Tensor, soil_moisture: torch.Tensor, soil_type_features: torch.Tensor) -> torch.Tensor:
        batch_size = cohesion.size(0)

        cohesion = cohesion.view(batch_size)
        slope_angle = slope_angle.view(batch_size)
        soil_moisture = soil_moisture.view(batch_size)
        
        slope_rad = torch.deg2rad(slope_angle)

        if soil_type_features.dim() > 1:
            soil_type_adjustment = torch.mean(soil_type_features, dim=1)
        else:
            soil_type_adjustment = soil_type_features

        phi_rad = torch.deg2rad(self.min_friction + soil_type_adjustment * (self.max_friction - self.min_friction))
        pore_pressure = (soil_moisture * self.gamma_h * 0.8).view(batch_size)

        numerator = cohesion + (self.gamma_h * torch.cos(slope_rad)**2 - pore_pressure) * torch.tan(phi_rad)
        denominator = self.gamma_h * torch.sin(slope_rad) * torch.cos(slope_rad) + 1e-6

        fos = numerator / denominator
        return torch.clamp(fos, min=0.1, max=10.0)
