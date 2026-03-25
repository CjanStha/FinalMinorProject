#!/usr/bin/env python
"""
Train and save models as pickle files.
"""

import sys
from pathlib import Path

# Setup paths
ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / 'models'
DATA_PATH = ML_DIR.parent / 'data' / 'raw_data' / 'dataset_ft_enriched.csv'

MODELS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("TRAINING AND SAVING MODELS")
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
    print(f"  Loaded {len(df)} samples")
    
    # Feature columns used by the backend
    # Note: competition_pressure is in the dataset, but backend uses competition_effective (10 - competition_pressure)
    available_features = {
        'population_density': 'population_density',
        'accessibility_score': 'accessibility_score',
        'foot_traffic_score': 'foot_traffic_score',
        'bus_stops_within_500m': 'bus_stops_within_500m',
        'nearby_schools': 'schools_within_500m',
        'nearby_hospitals': 'hospitals_within_500m',
    }
    
    # Build feature list from dataset
    feature_columns = []
    derived_features = {}
    
    for model_feat, dataset_feat in available_features.items():
        if dataset_feat in df.columns:
            feature_columns.append(dataset_feat)
    
    # Add derived feature: competition_effective (inverse of competition_pressure)
    if 'competition_pressure' in df.columns:
        df['competition_effective'] = 10 - df['competition_pressure'].clip(0, 10)
        feature_columns.insert(3, 'competition_effective')  # Insert after foot_traffic
        derived_features['competition_effective'] = True
    
    # Add derived feature: osm_amenity_density (approximate from schools + hospitals + bus_stops)
    if all(col in df.columns for col in ['schools_within_500m', 'hospitals_within_500m', 'bus_stops_within_500m']):
        df['osm_amenity_density_500m'] = (
            df['schools_within_500m'] + 
            df['hospitals_within_500m'] + 
            df['bus_stops_within_500m']
        ) / 3  # Average amenity count
        feature_columns.append('osm_amenity_density_500m')
        derived_features['osm_amenity_density_500m'] = True
    
    # Reorder to match backend model expectations
    desired_order = [
        'population_density', 'accessibility_score', 'foot_traffic_score',
        'competition_effective', 'bus_stops_within_500m',
        'schools_within_500m', 'hospitals_within_500m', 'osm_amenity_density_500m'
    ]
    feature_columns = [f for f in desired_order if f in df.columns]
    
    # Rename dataset columns to match backend expectations
    column_mapping = {
        'schools_within_500m': 'nearby_schools',
        'hospitals_within_500m': 'nearby_hospitals',
    }
    
    # Create final feature dataframe
    X_features = []
    for feat in feature_columns:
        if feat in column_mapping:
            X_features.append(df[feat].copy())
            X_features[-1].name = column_mapping[feat]
        else:
            X_features.append(df[feat].copy())
    
    # Find target column (suitability_score or similar)
    target_options = ['suitability_score', 'score', 'target', 'y']
    target_column = None
    for col in target_options:
        if col in df.columns:
            target_column = col
            break
    
    if not target_column:
        print(f"  ⚠ No pre-computed target column found")
        print(f"  Available columns: {df.columns.tolist()}")
        print(f"\n  Deriving suitability score from features...")
        
        # Derive suitability score from available features
        # Using a weighted combination similar to AHP methodology
        df['suitability_score'] = (
            (df['population_density'] / df['population_density'].max()) * 0.25 +
            (df['accessibility_score'] / 10.0) * 0.20 +
            (df['foot_traffic_score'] / 10.0) * 0.20 +
            (1.0 - df['competition_pressure'] / 10.0) * 0.15 +
            (df['bus_stops_within_500m'] / df['bus_stops_within_500m'].max()) * 0.10 +
            (df['schools_within_500m'] / df['schools_within_500m'].max()) * 0.05 +
            (df['hospitals_within_500m'] / df['hospitals_within_500m'].max()) * 0.05
        ) * 10  # Scale to 0-10
        
        target_column = 'suitability_score'
        print(f"  ✓ Derived suitability scores (range: {df['suitability_score'].min():.2f} - {df['suitability_score'].max():.2f})")
    
    print(f"  Target column: {target_column}")
    
    # Check if all features exist
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        print(f"  ⚠ Missing features: {missing_features}")
        print(f"  Available features: {df.columns.tolist()}")
        # Use available features
        feature_columns = [f for f in feature_columns if f in df.columns]
        print(f"  Using {len(feature_columns)} features")
    
    # Prepare data
    X = df[feature_columns].fillna(0)
    y = df[target_column].fillna(0)
    
    print(f"  Features: {len(feature_columns)}, Samples: {len(X)}")
    
    # Scale features to 0-10 range (as per training)
    print("\n[2] Scaling features to [0, 10]...")
    scaler = MinMaxScaler(feature_range=(0, 10))
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    print(f"  ✓ Scaler fitted")
    
    # Train Random Forest v2 (AHP tuned)
    print("\n[3] Training Random Forest v2 (AHP tuned)...")
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
    print(f"    R² Score: {rf_r2:.6f}")
    print(f"    RMSE: {rf_rmse:.6f}")
    print(f"    MAE: {rf_mae:.6f}")
    
    # Train XGBoost (AHP tuned)
    print("\n[4] Training XGBoost (AHP tuned)...")
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
    print(f"    R² Score: {xgb_r2:.6f}")
    print(f"    RMSE: {xgb_rmse:.6f}")
    print(f"    MAE: {xgb_mae:.6f}")
    
    # Save models and artifacts
    print("\n[5] Saving models to", MODELS_DIR)
    
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
    
    # Save feature columns
    features_path = MODELS_DIR / 'feature_columns_suitability.pkl'
    joblib.dump(feature_columns, features_path)
    file_size_b = features_path.stat().st_size
    print(f"  ✓ {features_path.name} ({file_size_b} bytes)")
    
    print("\n" + "=" * 80)
    print("✓ MODELS SAVED SUCCESSFULLY")
    print("=" * 80)
    print("\nModel Performance Summary:")
    print(f"\n  Random Forest v2 (AHP tuned):")
    print(f"    R² Score:  {rf_r2:.6f}")
    print(f"    RMSE:      {rf_rmse:.6f}")
    print(f"    MAE:       {rf_mae:.6f}")
    print(f"\n  XGBoost (AHP tuned):")
    print(f"    R² Score:  {xgb_r2:.6f}")
    print(f"    RMSE:      {xgb_rmse:.6f}")
    print(f"    MAE:       {xgb_mae:.6f}")
    print(f"\n  Scaler:")
    print(f"    Feature range: [0, 10]")
    print(f"    Features: {len(feature_columns)}")
    
    print(f"\n✓ All files saved to: {MODELS_DIR}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
