import os
import torch
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple
import logging

from src.models.st_pino import STPINOModel

logging.basicConfig(level=logging.INFO)

class LandslidePredictorEngine:
    """
    Unified Prediction Engine for GAIA Landslide Risk System.
    Loads ST-PINO, Random Forest, and XGBoost models and evaluates slope stability.
    """
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.st_pino = None
        self.rf_model = None
        self.xgb_model = None
        self.rf_encoders = None
        self.xgb_encoders = None
        self.load_all_models()

    def load_all_models(self):
        """Loads available machine learning and deep learning models."""
        # Load ST-PINO
        st_pino_path = os.path.join(self.model_dir, "st_pino_model.pt")
        if os.path.exists(st_pino_path):
            try:
                self.st_pino = STPINOModel(hydro_dim=6, geo_dim=3, land_use_dim=91, hidden_dim=64)
                state_dict = torch.load(st_pino_path, map_location=torch.device('cpu'))
                self.st_pino.load_state_dict(state_dict, strict=True)
                self.st_pino.eval()  # Set eval mode to avoid BatchNorm single-sample error
                logging.info("Successfully loaded ST-PINO model weights.")
            except Exception as e:
                logging.warning(f"Could not load ST-PINO model weights: {e}")

        # Load Random Forest
        rf_path = os.path.join(self.model_dir, "random_forest_model.joblib")
        if os.path.exists(rf_path):
            try:
                self.rf_model = joblib.load(rf_path)
                rf_enc_path = os.path.join(self.model_dir, "rf_label_encoders.joblib")
                if os.path.exists(rf_enc_path):
                    self.rf_encoders = joblib.load(rf_enc_path)
                logging.info("Successfully loaded Random Forest model.")
            except Exception as e:
                logging.warning(f"Could not load Random Forest model: {e}")

        # Load XGBoost
        xgb_path = os.path.join(self.model_dir, "xgboost_model.json")
        if os.path.exists(xgb_path):
            try:
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                xgb_enc_path = os.path.join(self.model_dir, "xgb_label_encoders.joblib")
                if os.path.exists(xgb_enc_path):
                    self.xgb_encoders = joblib.load(xgb_enc_path)
                logging.info("Successfully loaded XGBoost model.")
            except Exception as e:
                logging.warning(f"Could not load XGBoost model: {e}")

    def compute_factor_of_safety(self, cohesion: float, slope_angle: float, soil_moisture: float, gamma_h: float = 90.0, phi_deg: float = 30.0) -> float:
        """
        Calculates physical Factor of Safety (FOS) using Mohr-Coulomb limit equilibrium:
        FOS = (c' + (γh cos²β - u) tanφ') / (γh sinβ cosβ)
        """
        slope_rad = np.radians(max(slope_angle, 1.0))
        phi_rad = np.radians(phi_deg)
        u = soil_moisture * gamma_h * 0.8  # Pore water pressure proxy

        numerator = cohesion + (gamma_h * (np.cos(slope_rad) ** 2) - u) * np.tan(phi_rad)
        denominator = gamma_h * np.sin(slope_rad) * np.cos(slope_rad) + 1e-6

        fos = float(numerator / denominator)
        return float(np.clip(fos, 0.1, 10.0))

    def predict_stpino(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """Runs inference using Spatio-Temporal PINO model."""
        fos_calc = self.compute_factor_of_safety(
            cohesion=float(data.get("cohesion", 15.0)),
            slope_angle=float(data.get("slope_angle", 25.0)),
            soil_moisture=float(data.get("soil_moisture", 0.4))
        )

        if self.st_pino is None:
            prob = 0.90 if fos_calc < 1.0 else (0.40 if fos_calc < 1.5 else 0.05)
            return prob, fos_calc

        try:
            self.st_pino.eval()
            hydro_data = torch.tensor([[
                float(data.get("rainfall_mm", 20.0)),
                float(data.get("rainfall_anomaly", 5.0)),
                float(data.get("soil_moisture", 0.4)),
                float(data.get("rainfall_30d", 100.0)),
                float(data.get("day_of_year", 180)),
                float(data.get("month", 6))
            ]], dtype=torch.float32).unsqueeze(1)

            geo_features = torch.tensor([[
                float(data.get("slope_angle", 25.0)),
                float(data.get("elevation", 300.0)),
                float(data.get("volume_m3", 500.0))
            ]], dtype=torch.float32)

            geo_coords = torch.tensor([[
                float(data.get("latitude", 10.0)),
                float(data.get("longitude", 76.0))
            ]], dtype=torch.float32)

            # 91 features one-hot land-use/categorical tensor
            land_use_data = torch.zeros((1, 1, 91), dtype=torch.float32)
            land_use_data[0, 0, 0] = 1.0

            soil_params = {
                'cohesion': torch.tensor([float(data.get("cohesion", 15.0))], dtype=torch.float32),
                'slope_angle': torch.tensor([float(data.get("slope_angle", 25.0))], dtype=torch.float32),
                'soil_moisture': torch.tensor([float(data.get("soil_moisture", 0.4))], dtype=torch.float32),
                'soil_type_features': torch.zeros((1, 5), dtype=torch.float32)
            }

            with torch.no_grad():
                out = self.st_pino(hydro_data, geo_features, geo_coords, land_use_data, soil_params)
                logits = out['logits']
                fos_tensor = out['fos'].item()
                probs = torch.softmax(logits, dim=1)
                landslide_prob = probs[0, 1].item()
                return float(landslide_prob), float(fos_tensor)
        except Exception as e:
            logging.error(f"ST-PINO prediction error: {e}")
            prob = 0.85 if fos_calc < 1.0 else 0.10
            return prob, fos_calc

    def predict_rf(self, data: Dict[str, Any]) -> float:
        """Runs inference using Random Forest model."""
        if self.rf_model is None:
            return 0.5
        try:
            soil_enc = 0
            if self.rf_encoders and 'soil_type' in self.rf_encoders:
                soil_str = str(data.get('soil_type', 'Unknown'))
                classes = list(self.rf_encoders['soil_type'].classes_)
                if soil_str in classes:
                    soil_enc = int(self.rf_encoders['soil_type'].transform([soil_str])[0])

            features = [
                float(data.get('latitude', 10.0)),
                float(data.get('longitude', 76.0)),
                float(data.get('rainfall_mm', 20.0)),
                float(data.get('rainfall_anomaly', 5.0)),
                float(data.get('soil_moisture', 0.4)),
                float(data.get('slope_angle', 25.0)),
                float(data.get('elevation', 300.0)),
                soil_enc
            ]
            prob = self.rf_model.predict_proba([features])[0][1]
            return float(prob)
        except Exception as e:
            logging.error(f"Random Forest prediction error: {e}")
            return 0.5

    def predict_xgb(self, data: Dict[str, Any]) -> float:
        """Runs inference using XGBoost model."""
        if self.xgb_model is None:
            return 0.5
        try:
            soil_enc = 0
            land_enc = 0
            if self.xgb_encoders:
                if 'soil_type' in self.xgb_encoders and str(data.get('soil_type')) in list(self.xgb_encoders['soil_type'].classes_):
                    soil_enc = int(self.xgb_encoders['soil_type'].transform([str(data.get('soil_type'))])[0])
                if 'land_use' in self.xgb_encoders and str(data.get('land_use')) in list(self.xgb_encoders['land_use'].classes_):
                    land_enc = int(self.xgb_encoders['land_use'].transform([str(data.get('land_use'))])[0])

            features = [
                float(data.get('latitude', 10.0)),
                float(data.get('longitude', 76.0)),
                float(data.get('rainfall_mm', 20.0)),
                float(data.get('rainfall_anomaly', 5.0)),
                float(data.get('soil_moisture', 0.4)),
                float(data.get('slope_angle', 25.0)),
                float(data.get('elevation', 300.0)),
                soil_enc,
                land_enc
            ]
            dmat = xgb.DMatrix([features])
            prob = self.xgb_model.predict(dmat)[0]
            return float(prob)
        except Exception as e:
            logging.error(f"XGBoost prediction error: {e}")
            return 0.5
