#!/usr/bin/env python
"""
Train and save models as pickle files from the notebook data.
"""

import sys
from pathlib import Path

# Directories
ML_DIR = Path(__file__).resolve().parent
MODELS_DIR = ML_DIR / 'models'

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("TRAINING AND SAVING MODELS")
print("=" * 80)

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBRegressor
    
    # Try to locate the enriched dataset
    backend_data = ML_DIR.parent / 'cafelocate' / 'backend' / 'data'
    main_data = ML_DIR / 'data'
    
    dataset_path = None
    for path in [main_data / 'dataset_ft_enriched.csv', backend_data / 'dataset_ft_enriched.csv']:
        if path.exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("⚠ Dataset not found. Searched in:")
        print(f"  - {main_data / 'dataset_ft_enriched.csv'}")
        print(f"  - {backend_data / 'dataset_ft_enriched.csv'}")
        sys.exit(1)
    
    print(f"✓ Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Feature columns
    feature_columns = [
        'population_density', 'accessibility_score', 'foot_traffic_score',
        'competition_effective', 'bus_stops_within_500m',
        'osm_amenity_density_500m', 'nearby_schools', 'nearby_hospitals'
    ]
    
    # Target column
    target_column = 'suitability_score' if 'suitability_score' in df.columns else 'target'
    
    # Prepare data
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"✓ Loaded {len(X)} samples with {len(feature_columns)} features")
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 10))
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # Train Random Forest
    print("\n[4] Training Random Forest v2 (AHP tuned)...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_scaled_df, y)
    
    # Evaluate RF
    rf_r2 = rf_model.score(X_scaled_df, y)
    rf_pred = rf_model.predict(X_scaled_df)
    rf_rmse = np.sqrt(np.mean((rf_pred - y) ** 2))
    print(f"  R² Score: {rf_r2:.6f}")
    print(f"  RMSE: {rf_rmse:.6f}")
    
    # Train XGBoost
    print("\n[5] Training XGBoost (AHP tuned)...")
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
    xgb_model.fit(X_scaled_df, y)
    
    # Evaluate XGB
    xgb_r2 = xgb_model.score(X_scaled_df, y)
    xgb_pred = xgb_model.predict(X_scaled_df)
    xgb_rmse = np.sqrt(np.mean((xgb_pred - y) ** 2))
    print(f"  R² Score: {xgb_r2:.6f}")
    print(f"  RMSE: {xgb_rmse:.6f}")
    
    # Save models and preprocessing artifacts
    print("\n[6] Saving models and artifacts to", MODELS_DIR)
    
    # Save Random Forest model
    rf_path = MODELS_DIR / 'rf_suitability_v2_ahp_tuned.pkl'
    joblib.dump(rf_model, rf_path)
    print(f"  ✓ Saved Random Forest: {rf_path}")
    
    # Save XGBoost model
    xgb_path = MODELS_DIR / 'xgb_suitability_ahp.pkl'
    joblib.dump(xgb_model, xgb_path)
    print(f"  ✓ Saved XGBoost: {xgb_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / 'scaler_suitability.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved Scaler: {scaler_path}")
    
    # Save feature columns
    features_path = MODELS_DIR / 'feature_columns_suitability.pkl'
    joblib.dump(feature_columns, features_path)
    print(f"  ✓ Saved Feature Columns: {features_path}")
    
    print("\n" + "=" * 80)
    print("✓ ALL MODELS SAVED SUCCESSFULLY")
    print("=" * 80)
    print("\nModel Summary:")
    print(f"  Random Forest v2:")
    print(f"    - R²: {rf_r2:.4f}")
    print(f"    - RMSE: {rf_rmse:.4f}")
    print(f"  XGBoost:")
    print(f"    - R²: {xgb_r2:.4f}")
    print(f"    - RMSE: {xgb_rmse:.4f}")
    print(f"\nFiles saved to: {MODELS_DIR}")
    
except Exception as e:
    print(f"\n✗ Error during model training and saving: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
