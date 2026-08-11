import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Any

class LandslideDataPreprocessor:
    """Preprocesses input tabular data for machine learning and deep learning models."""
    def __init__(self):
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')

        self.numerical_features = [
            'rainfall_mm', 'rainfall_anomaly', 'soil_moisture', 
            'slope_angle', 'elevation'
        ]
        self.categorical_features = ['land_use', 'soil_type']
        self.spatial_features = ['latitude', 'longitude']

    def preprocess_single_entry(self, data_dict: Dict[str, Any]) -> pd.DataFrame:
        """Takes raw user input dictionary and returns a preprocessed pandas DataFrame."""
        df = pd.DataFrame([data_dict])
        
        # Default missing values
        for feat in self.numerical_features:
            if feat not in df.columns or pd.isna(df[feat].iloc[0]):
                df[feat] = 0.0
                
        for feat in self.categorical_features:
            if feat not in df.columns or pd.isna(df[feat].iloc[0]):
                df[feat] = "Unknown"

        if 'date' in df.columns and pd.notna(df['date'].iloc[0]):
            dt = pd.to_datetime(df['date'].iloc[0])
            df['day_of_year'] = dt.dayofyear
            df['month'] = dt.month
        else:
            df['day_of_year'] = 180
            df['month'] = 6

        df['rainfall_30d'] = df['rainfall_mm'] * 3.5  # 30-day cumulative estimate
        return df
