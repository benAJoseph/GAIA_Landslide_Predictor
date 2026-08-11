import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import folium
from streamlit_folium import st_folium
import os
import sys

# Ensure local src directory is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.predictor import LandslidePredictorEngine
from src.data.external_api import (
    get_lat_lon, get_daily_rainfall, calculate_rainfall_anomaly,
    get_hourly_soil_moisture, get_elevation, get_land_use, get_soil_type
)

# Set page configuration with custom title & wide layout
st.set_page_config(
    page_title="GAIA | Physics-Informed Landslide Intelligence",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics: Dark theme, Glassmorphism, Micro-animations, Custom Badges
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0d1322 100%);
        color: #f3f4f6;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(31, 41, 55, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }
    
    /* Header Gradient Banner */
    .header-banner {
        background: linear-gradient(90deg, #0284c7 0%, #3b82f6 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Metric Display Badges */
    .risk-badge-high {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 14px 0 rgba(239, 68, 68, 0.39);
    }
    
    .risk-badge-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 14px 0 rgba(245, 158, 11, 0.39);
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.39);
    }

    /* Custom Streamlit Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.35);
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.5);
        transform: translateY(-1px);
    }
    
    /* Code/Math styling */
    .math-box {
        font-family: 'JetBrains Mono', monospace;
        background: #1e293b;
        padding: 14px;
        border-radius: 8px;
        border-left: 4px solid #38bdf8;
        color: #e2e8f0;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor Engine once using Streamlit resource caching
@st.cache_resource
def get_engine():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    return LandslidePredictorEngine(model_dir=model_dir)

engine = get_engine()

# Sidebar: Controls & Model Settings
st.sidebar.image("https://img.icons8.com/isometric/100/mountain.png", width=70)
st.sidebar.markdown("## ⚙️ Model Settings")
selected_model = st.sidebar.radio(
    "Select Intelligence Engine",
    ["ST-PINO (Physics-Informed Neural Operator)", "XGBoost Classifier", "Random Forest Ensemble"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🌐 Data Auto-Fetch")
use_api_autofill = st.sidebar.checkbox("Auto-fetch Weather & Soil via APIs", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Architecture Info")
st.sidebar.info(
    "**GAIA ST-PINO Architecture:**\n"
    "- 🧠 **Hydrological Expert**: Bi-LSTM\n"
    "- 🗺️ **Geotechnical Expert**: GCN Spatial Model\n"
    "- 🌲 **Land Use Expert**: Transformer Self-Attention\n"
    "- ⚖️ **Physics Layer**: Factor of Safety ($FOS$)"
)

# Header Section
st.markdown("<h1 class='header-banner'>GAIA Landslide Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Physics-Informed Deep Learning & Geotechnical Early Warning System for Slope Stability</p>", unsafe_allow_html=True)

# Layout Columns
col_input, col_results = st.columns([1, 1.2])

with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📍 Location & Date Parameters")
    
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        location_input = st.text_input("Enter Location / District", value="Mundakkai, Wayanad, Kerala", placeholder="e.g. Wayanad, Kerala")
    with loc_col2:
        date_input = st.date_input("Target Date", value=datetime.now().date())
        
    date_str = date_input.strftime("%Y-%m-%d")

    # Fetch API values button or auto-run
    if use_api_autofill and location_input:
        lat, lon = get_lat_lon(location_input)
        if lat is None or lon is None:
            st.warning(f"Could not automatically geocode '{location_input}'. Using default coordinates (Kerala).")
            lat, lon = 11.4628, 76.1342
        
        daily_rain = get_daily_rainfall(lat, lon, date_str)
        anom_data = calculate_rainfall_anomaly(lat, lon, date_str)
        soil_m = get_hourly_soil_moisture(lat, lon, date_str)
        elev = get_elevation(lat, lon)
        land_u = get_land_use(lat, lon)
        soil_t = get_soil_type(lat, lon)
    else:
        lat, lon = 11.4628, 76.1342
        daily_rain = {"recent": 45.0, "sum_7d": 180.0}
        anom_data = {"mean_anomaly": 15.0, "standardized_anomaly": 2.1, "total_intensity": 180.0}
        soil_m = {"recent": 0.45, "trend": 0.08, "max": 0.60}
        elev = 1250.0
        land_u = "Forest / Agricultural"
        soil_t = "Laterite / Clayey"

    st.markdown("---")
    st.markdown("### 🌧️ Environmental & Geotechnical Factors")
    
    g1, g2 = st.columns(2)
    with g1:
        rainfall_val = st.number_input("Recent 24h Rainfall (mm)", value=float(daily_rain["recent"]), min_value=0.0, max_value=800.0, step=5.0)
        rainfall_anom_val = st.number_input("Rainfall Anomaly (mm)", value=float(anom_data["mean_anomaly"]), min_value=-100.0, max_value=500.0, step=5.0)
        soil_moisture_val = st.number_input("Soil Moisture (m³/m³)", value=float(soil_m["recent"]), min_value=0.0, max_value=1.0, step=0.05)
        cohesion_val = st.number_input("Effective Soil Cohesion c' (kPa)", value=15.0, min_value=0.0, max_value=100.0, step=1.0)

    with g2:
        slope_angle_val = st.number_input("Slope Angle (Degrees)", value=32.0, min_value=0.0, max_value=90.0, step=1.0)
        elevation_val = st.number_input("Elevation (Meters)", value=float(elev), min_value=0.0, max_value=8000.0, step=10.0)
        soil_type_val = st.selectbox("Soil Type", ["Laterite / Clayey", "Debris / Granular", "Sand / Loam"], index=0)
        land_use_val = st.selectbox("Land Cover Type", ["Forest / Agricultural", "Deforested / Barren", "Urbanized Infrastructure"], index=0)

    st.markdown("</div>", unsafe_allow_html=True)
    
    predict_btn = st.button("🚀 Predict Landslide Risk & Stability")

# Run prediction
input_payload = {
    "latitude": lat,
    "longitude": lon,
    "date": date_str,
    "location": location_input,
    "rainfall_mm": rainfall_val,
    "rainfall_anomaly": rainfall_anom_val,
    "soil_moisture": soil_moisture_val,
    "slope_angle": slope_angle_val,
    "elevation": elevation_val,
    "cohesion": cohesion_val,
    "soil_type": soil_type_val,
    "land_use": land_use_val
}

with col_results:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Prediction & Stability Assessment")
    
    # Compute ST-PINO and alternatives
    prob_stpino, fos_val = engine.predict_stpino(input_payload)
    prob_rf = engine.predict_rf(input_payload)
    prob_xgb = engine.predict_xgb(input_payload)
    
    if "ST-PINO" in selected_model:
        active_prob = prob_stpino
    elif "XGBoost" in selected_model:
        active_prob = prob_xgb
    else:
        active_prob = prob_rf
        
    # Categorize Risk
    if active_prob > 0.65 or fos_val < 1.0:
        risk_status = "CRITICAL / HIGH RISK"
        risk_class = "risk-badge-high"
    elif active_prob > 0.35 or fos_val < 1.5:
        risk_status = "MODERATE WARNING"
        risk_class = "risk-badge-moderate"
    else:
        risk_status = "LOW RISK / STABLE"
        risk_class = "risk-badge-low"
        
    st.markdown(f"<div class='{risk_class}'>{risk_status}</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Landslide Risk Prob", f"{active_prob * 100:.1f}%")
    with m2:
        st.metric("Factor of Safety (FOS)", f"{fos_val:.2f}", delta="Unstable (<1.0)" if fos_val < 1.0 else "Stable (≥1.0)")
    with m3:
        st.metric("Slope Angle", f"{slope_angle_val}°")
        
    # Gauge Chart for Probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = active_prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Landslide Hazard Probability (%)", 'font': {'size': 18, 'color': '#f3f4f6'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9ca3af"},
            'bar': {'color': "#ef4444" if active_prob > 0.65 else ("#f59e0b" if active_prob > 0.35 else "#10b981")},
            'bgcolor': "#1f2937",
            'borderwidth': 2,
            'bordercolor': "#374151",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ]
        }
    ))
    fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Lower Visualizations: Map and Model Comparison
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### 🌍 Geospatial Hazard & Multi-Model Comparison")

map_col, chart_col = st.columns([1, 1])

with map_col:
    st.markdown("#### Interactive Terrain Map")
    m = folium.Map(location=[lat, lon], zoom_start=11, tiles="OpenStreetMap")
    
    popup_text = f"<b>{location_input}</b><br>Risk Prob: {active_prob*100:.1f}%<br>FOS: {fos_val:.2f}"
    color = "red" if active_prob > 0.65 or fos_val < 1.0 else ("orange" if active_prob > 0.35 else "green")
    
    folium.Marker(
        [lat, lon],
        popup=popup_text,
        tooltip=location_input,
        icon=folium.Icon(color=color, icon="exclamation-triangle" if color=="red" else "info-sign", prefix="fa")
    ).add_to(m)
    
    folium.Circle(
        [lat, lon],
        radius=3000,
        color=color,
        fill=True,
        fill_opacity=0.3
    ).add_to(m)
    
    st_folium(m, height=320, width=None)

with chart_col:
    st.markdown("#### Model Probability Comparison")
    df_models = pd.DataFrame({
        "Model": ["ST-PINO (Physics)", "XGBoost", "Random Forest"],
        "Probability (%)": [prob_stpino * 100, prob_xgb * 100, prob_rf * 100]
    })
    
    fig_bar = px.bar(
        df_models,
        x="Model",
        y="Probability (%)",
        color="Model",
        color_discrete_sequence=["#38bdf8", "#818cf8", "#f472b6"],
        text_auto=".1f"
    )
    fig_bar.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f3f4f6"},
        showlegend=False,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# Physics Grounding Section
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("### ⚖️ Geotechnical Physics Grounding (Factor of Safety)")
st.write(
    "Unlike purely black-box machine learning models, **GAIA ST-PINO** enforces physical consistency by integrating "
    "the **Mohr-Coulomb Limit Equilibrium Equation** into neural network feature attention weights:"
)
st.markdown(
    "<div class='math-box'>"
    "$$FOS = \\frac{c' + (\\gamma h \\cos^2\\beta - u) \\tan\\phi'}{\\gamma h \\sin\\beta \\cos\\beta}$$"
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    f"- **Computed $FOS$ Value:** `{fos_val:.3f}`\n"
    f"- **Interpretation:** {'⚠️ **Slope failure imminent ($FOS < 1.0$)**. Driving shear force exceeds resisting shear strength.' if fos_val < 1.0 else '✅ **Slope is in equilibrium ($FOS \\ge 1.0$)**. Resisting shear strength satisfies physical stability threshold.'}"
)
st.markdown("</div>", unsafe_allow_html=True)
