# 🏔️ GAIA: Physics-Informed Spatio-Temporal Landslide Predictor

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**GAIA** is an AI-driven landslide risk prediction and early warning framework combining **Physics-Informed Deep Learning** (Spatio-Temporal Physics-Informed Neural Operator with Attention Gating - **ST-PINO**) alongside traditional machine learning benchmarks (XGBoost & Random Forest).

---

## 🌟 Key Features

1. **Spatio-Temporal Physics-Informed Neural Operator (ST-PINO)**:
   - **Hydrological Expert**: Bi-LSTM for sequential rainfall & soil moisture temporal trends.
   - **Geotechnical Expert**: Graph Convolutional Network (GCN) for spatial terrain topology & proximity modeling.
   - **Land Use Expert**: Transformer Multi-Head Self-Attention for land cover dynamics.
   - **Physics Layer**: Differentiable Factor of Safety ($FOS$) calculation grounding predictions in physical slope stability laws.
2. **Live Environmental API Auto-Fetch**:
   - Geocoding via OpenStreetMap Nominatim.
   - Live & historical rainfall and soil moisture via Open-Meteo API.
   - Elevation data via Open-Elevation API.
3. **Interactive Web Dashboard**:
   - Built with Streamlit, Folium maps, Plotly risk gauges, dark glassmorphism UI, and physical stability indicators.

---

## 📐 Physical Grounding: Factor of Safety ($FOS$)

GAIA enforces physical consistency by integrating the Mohr-Coulomb limit equilibrium equation directly into neural attention gating:

$$FOS = \frac{c' + (\gamma h \cos^2\beta - u) \tan\phi'}{\gamma h \sin\beta \cos\beta}$$

- $FOS < 1.0$: Slope Failure Imminent (High Risk)
- $1.0 \le FOS < 1.5$: Marginally Stable (Moderate Risk)
- $FOS \ge 1.5$: Stable Slope (Low Risk)

---

## 🛠️ Quickstart Installation

1. **Clone Repository & Navigate**:
   ```bash
   git clone https://github.com/benAJoseph/GAIA_Landslide_Predictor.git
   cd GAIA_Landslide_Predictor
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Interactive Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Repository Structure

```
GAIA_Landslide_Predictor/
├── app.py                      # Interactive Web App (Streamlit)
├── requirements.txt            # Dependency specification
├── README.md                   # Project documentation
├── models/                     # Pre-trained model weights
│   ├── st_pino_model.pt        # PyTorch ST-PINO weights
│   ├── xgboost_model.json      # Trained XGBoost model
│   └── random_forest_model.joblib # Trained Random Forest model
└── src/
    ├── predictor.py            # Unified Predictor Engine
    ├── models/                 # Neural architectures & physics layers
    │   ├── st_pino.py          # ST-PINO model wrapper
    │   ├── physics_layer.py    # Differentiable FOS layer
    │   ├── attention_gating.py # Physics-guided attention
    │   └── experts/            # Hydrological, Geotechnical, Land Use experts
    └── data/
        ├── external_api.py     # Weather, elevation & geocoding APIs
        └── data_preprocessor.py# Data cleaning & normalization pipeline
```

---

## 📄 Citation & Publication

*Coming Soon*
