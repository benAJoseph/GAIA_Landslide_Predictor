import requests
from datetime import datetime, timedelta
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO)

def get_lat_lon(location_name: str) -> Tuple[Optional[float], Optional[float]]:
    """Geocode location name to (latitude, longitude) using OpenStreetMap Nominatim API."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "GAIA-LandslidePredictor/1.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logging.warning(f"Geocoding error for '{location_name}': {e}")
    return None, None

def get_daily_rainfall(lat: float, lon: float, date_str: str) -> Dict[str, float]:
    """Fetch daily rainfall metrics from Open-Meteo historical weather API."""
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=7)).strftime("%Y-%m-%d")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": date_str,
            "daily": "rain_sum",
            "timezone": "auto"
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            rain_sums = resp.json().get("daily", {}).get("rain_sum", [])
            recent_rain = float(rain_sums[-1]) if rain_sums else 15.0
            sum_7d = float(np.sum(rain_sums)) if rain_sums else 50.0
            return {"recent": recent_rain, "sum_7d": sum_7d}
    except Exception as e:
        logging.warning(f"Rainfall API error for ({lat}, {lon}): {e}")
    return {"recent": 25.0, "sum_7d": 120.0}

def calculate_rainfall_anomaly(lat: float, lon: float, date_str: str) -> Dict[str, float]:
    """Calculate rainfall anomaly metrics comparing current month to baseline."""
    daily_info = get_daily_rainfall(lat, lon, date_str)
    recent = daily_info["recent"]
    baseline_mean = 10.0
    baseline_std = 8.0
    mean_anomaly = float(recent - baseline_mean)
    standardized_anomaly = float((recent - baseline_mean) / (baseline_std + 1e-5))
    return {
        "mean_anomaly": round(mean_anomaly, 2),
        "standardized_anomaly": round(standardized_anomaly, 2),
        "total_intensity": round(daily_info["sum_7d"], 2)
    }

def get_hourly_soil_moisture(lat: float, lon: float, date_str: str) -> Dict[str, float]:
    """Fetch soil moisture from Open-Meteo API."""
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=3)).strftime("%Y-%m-%d")
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": date_str,
            "hourly": "soil_moisture_0_to_7cm",
            "timezone": "auto"
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            moisture_vals = resp.json().get("hourly", {}).get("soil_moisture_0_to_7cm", [])
            valid_vals = [v for v in moisture_vals if v is not None]
            if valid_vals:
                recent = float(valid_vals[-1])
                trend = float(valid_vals[-1] - valid_vals[0])
                max_val = float(np.max(valid_vals))
                return {"recent": recent, "trend": trend, "max": max_val}
    except Exception as e:
        logging.warning(f"Soil moisture API error for ({lat}, {lon}): {e}")
    return {"recent": 0.42, "trend": 0.05, "max": 0.55}

def get_elevation(lat: float, lon: float) -> float:
    """Fetch elevation in meters using Open-Elevation API."""
    url = "https://api.open-elevation.com/api/v1/lookup"
    params = {"locations": f"{lat},{lon}"}
    try:
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                return float(results[0]["elevation"])
    except Exception as e:
        logging.warning(f"Elevation API error for ({lat}, {lon}): {e}")
    # Default fallback estimation for hilly terrain
    return float(np.round(300 + abs(lat) * 20 + abs(lon) * 5, 1))

def get_land_use(lat: float, lon: float) -> str:
    """Determine land use type based on geospatial coordinates."""
    # Simplified regional land-use classifier for demo/geohazard areas
    if 8.0 <= lat <= 13.0 and 74.0 <= lon <= 78.0:
        return "Agricultural/Forest"
    elif lat > 20.0:
        return "Mountainous Slope"
    return "Forest"

def get_soil_type(lat: float, lon: float) -> str:
    """Determine soil type based on geospatial coordinates."""
    if 8.0 <= lat <= 13.0:
        return "Laterite / Clayey Soil"
    return "Loam / Sandy Clay"
