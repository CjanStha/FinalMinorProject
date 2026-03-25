#!/usr/bin/env python
"""
Train and save models as pickle files from the enriched dataset.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Setup paths
ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / 'models'
DATA_PATH = ML_DIR.parent / 'data' / 'raw_data' / 'dataset_ft_enriched.csv'

MODELS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("TRAINING AND SAVING MODELS FROM DATASET")
print("=" * 80)

if not DATA_PATH.exists():
    print(f"✗ Dataset not found: {DATA_PATH}")
    sys.exit(1)

print(f"\n✓ Dataset found: {DATA_PATH}")

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor
    
    # Load dataset
    print("\n[1] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} samples, {len(df.columns)} columns")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Create target: suitability_score (derived from features)
    print("\n[2] Creating target variable (suitability score)...")
    
    # Normalize features to 0-10 scale
    df['population_density_norm'] = (df['population_density'] - df['population_density'].min()) / (df['population_density'].max() - df['population_density'].min()) * 10
    df['accessibility_norm'] = df['accessibility_score'].clip(0, 10)
    df['foot_traffic_norm'] = df['foot_traffic_score'].clip(0, 10)
    df['competition_effective'] = (10 - df['competition_pressure'].clip(0, 10))
    df['bus_stops_norm'] = df['bus_stops_within_500m'].clip(0, 10)
    df['schools_norm'] = df['schools_within_500m'].clip(0, 10)
    df['hospitals_norm'] = df['hospitals_within_500m'].clip(0, 10)
    
    # Computed amenity density
    amenity_cols = ['schools_within_500m', 'hospitals_within_500m', 'bus_stops_within_500m']
    df['osm_amenity_density_500m'] = df[amenity_cols].sum(axis=1).clip(0, 10)
    
    # Create suitability score using weighted combination (AHP-inspired)
    df['suitability_score'] = (
        df['population_density_norm'] * 0.30 +  # Location is critical
        df['accessibility_norm'] * 0.20 +
        df['foot_traffic_norm'] * 0.20 +
        df['competition_effective'] * 0.10 +
        df['bus_stops_norm'] * 0.08 +
        df['schools_norm'] * 0.07 +
        df['hospitals_norm'] * 0.05
    )
    
    print(f"  Suitability score range: {df['suitability_score'].min():.2f} - {df['suitability_score'].max():.2f}")
    print(f"  Mean: {df['suitability_score'].mean():.2f}, Std: {df['suitability_score'].std():.2f}")
    
    # Feature columns matching backend model
    feature_columns = [
        'population_density_norm',
        'accessibility_norm',
        'foot_traffic_norm',
        'competition_effective',
        'bus_stops_norm',
        'osm_amenity_density_500m',
        'schools_norm',
        'hospitals_norm'
    ]
    
    # Rename for consistency with backend
    feature_names_backend = [
        'population_density',
        'accessibility_score',
        'foot_traffic_score',
        'competition_effective',
        'bus_stops_within_500m',
        'osm_amenity_density_500m',
        'nearby_schools',
        'nearby_hospitals'
    ]
    
    # Prepare data
    print("\n[3] Preparing training data...")
    X = df[feature_columns].fillna(0)
    y = df['suitability_score'].fillna(0)
    
    print(f"  Features: {len(feature_names_backend)}, Samples: {len(X)}")
    print(f"  Feature names: {feature_names_backend}")
    
    # Scale features to 0-10 range (as used in training)
    print("\n[4] Scaling features to [0, 10]...")
    scaler = MinMaxScaler(feature_range=(0, 10))
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names_backend)
    print(f"  ✓ Scaler fitted")
    
    # Train Random Forest v2 (AHP tuned)
    print("\n[5] Training Random Forest v2 (AHP tuned)...")
    print("  Config: n_estimators=150, max_depth=12, min_samples_split=3")
    
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_scaled_df, y)
    
    rf_r2 = rf_model.score(X_scaled_df, y)
    rf_pred = rf_model.predict(X_scaled_df)
    rf_rmse = np.sqrt(np.mean((rf_pred - y) ** 2))
    rf_mae = np.mean(np.abs(rf_pred - y))
    
    print(f"  ✓ Training complete")
    print(f"    R² Score:  {rf_r2:.6f}")
    print(f"    RMSE:      {rf_rmse:.6f}")
    print(f"    MAE:       {rf_mae:.6f}")
    
    # Train XGBoost (AHP tuned)
    print("\n[6] Training XGBoost (AHP tuned)...")
    print("  Config: n_estimators=150, max_depth=8, learning_rate=0.1")
    
    xgb_model = XGBRegressor(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(X_scaled_df, y, verbose=False)
    
    xgb_r2 = xgb_model.score(X_scaled_df, y)
    xgb_pred = xgb_model.predict(X_scaled_df)
    xgb_rmse = np.sqrt(np.mean((xgb_pred - y) ** 2))
    xgb_mae = np.mean(np.abs(xgb_pred - y))
    
    print(f"  ✓ Training complete")
    print(f"    R² Score:  {xgb_r2:.6f}")
    print(f"    RMSE:      {xgb_rmse:.6f}")
    print(f"    MAE:       {xgb_mae:.6f}")
    
    # Save models and artifacts
    print("\n[7] Saving models to", MODELS_DIR)
    
    # Save Random Forest model
    rf_path = MODELS_DIR / 'rf_suitability_v2_ahp_tuned.pkl'
    joblib.dump(rf_model, rf_path)
    file_size_mb = rf_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {rf_path.name} ({file_size_mb:.2f} MB)")
    
    # Save XGBoost model
    xgb_path = MODELS_DIR / 'xgb_suitability_ahp.pkl'
    joblib.dump(xgb_model, xgb_path)
    file_size_mb = xgb_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ {xgb_path.name} ({file_size_mb:.2f} MB)")
    
    # Save scaler
    scaler_path = MODELS_DIR / 'scaler_suitability.pkl'
    joblib.dump(scaler, scaler_path)
    file_size_kb = scaler_path.stat().st_size / 1024
    print(f"  ✓ {scaler_path.name} ({file_size_kb:.2f} KB)")
    
    # Save feature columns (using backend names)
    features_path = MODELS_DIR / 'feature_columns_suitability.pkl'
    joblib.dump(feature_names_backend, features_path)
    file_size_b = features_path.stat().st_size
    print(f"  ✓ {features_path.name} ({file_size_b} bytes)")
    
    print("\n" + "=" * 80)
    print("✓ MODELS AND ARTIFACTS SAVED SUCCESSFULLY")
    print("=" * 80)
    
    print("\nModel Performance Summary:")
    print(f"\n  Random Forest v2 (AHP tuned):")
    print(f"    R² Score:  {rf_r2:.6f}")
    print(f"    RMSE:      {rf_rmse:.6f}")
    print(f"    MAE:       {rf_mae:.6f}")
    print(f"    Samples:   {len(X)}")
    
    print(f"\n  XGBoost (AHP tuned):")
    print(f"    R² Score:  {xgb_r2:.6f}")
    print(f"    RMSE:      {xgb_rmse:.6f}")
    print(f"    MAE:       {xgb_mae:.6f}")
    print(f"    Samples:   {len(X)}")
    
    print(f"\n  Scaler:")
    print(f"    Feature range: [0, 10]")
    print(f"    Features: {len(feature_names_backend)}")
    print(f"    Features: {feature_names_backend}")
    
    print(f"\n  Data Source: {DATA_PATH}")
    print(f"  Output Directory: {MODELS_DIR}")
    
    print("\n" + "=" * 80)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
