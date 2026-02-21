"""
Train ML model for café location suitability prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_training_data():
    """
    Load the real training data for location suitability prediction
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'cafe_location_training_dataset.csv')

    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        print("Falling back to synthetic data generation...")
        return create_synthetic_training_data()

    print(f"Loading training data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Training data shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("Suitability distribution:")
    print(df['suitability'].value_counts())

    return df

def create_synthetic_training_data():
    """
    Create synthetic training data for location suitability prediction
    Based on the features from the real training dataset
    """
    np.random.seed(42)

    # Create 1574 synthetic data points (same as real dataset)
    n_samples = 1574

    data = []

    for i in range(n_samples):
        # Generate features similar to real data
        competitors_within_500m = np.random.randint(0, 20)
        competitors_within_200m = np.random.randint(0, competitors_within_500m + 1)
        competitors_min_distance = np.random.uniform(25, 500)
        competitors_avg_distance = np.random.uniform(competitors_min_distance, 500)

        roads_within_500m = np.random.randint(0, 15)
        roads_avg_distance = np.random.uniform(50, 500)

        schools_within_500m = np.random.randint(0, 20)
        schools_within_200m = np.random.randint(0, schools_within_500m + 1)
        schools_min_distance = np.random.uniform(25, 500)

        hospitals_within_500m = np.random.randint(0, 5)
        hospitals_min_distance = np.random.uniform(100, 500)

        bus_stops_within_500m = np.random.randint(0, 10)
        bus_stops_min_distance = np.random.uniform(50, 500)

        population_density_proxy = np.random.uniform(0, 100)
        accessibility_score = np.random.uniform(0, 5)
        foot_traffic_score = np.random.uniform(0, 10)
        competition_pressure = np.random.uniform(0, 10)

        # Determine suitability based on features
        score = (
            (population_density_proxy * 0.2) +
            (accessibility_score * 0.15) +
            (foot_traffic_score * 0.15) +
            (schools_within_500m * 0.1) +
            (bus_stops_within_500m * 0.1) -
            (competition_pressure * 0.2) -
            (competitors_within_200m * 0.1)
        )

        if score > 15:
            suitability = 'High'
        elif score > 8:
            suitability = 'Medium'
        else:
            suitability = 'Low'

        data.append({
            'competitors_within_500m': competitors_within_500m,
            'competitors_within_200m': competitors_within_200m,
            'competitors_min_distance': competitors_min_distance,
            'competitors_avg_distance': competitors_avg_distance,
            'roads_within_500m': roads_within_500m,
            'roads_avg_distance': roads_avg_distance,
            'schools_within_500m': schools_within_500m,
            'schools_within_200m': schools_within_200m,
            'schools_min_distance': schools_min_distance,
            'hospitals_within_500m': hospitals_within_500m,
            'hospitals_min_distance': hospitals_min_distance,
            'bus_stops_within_500m': bus_stops_within_500m,
            'bus_stops_min_distance': bus_stops_min_distance,
            'population_density_proxy': population_density_proxy,
            'accessibility_score': accessibility_score,
            'foot_traffic_score': foot_traffic_score,
            'competition_pressure': competition_pressure,
            'suitability': suitability
        })

    return pd.DataFrame(data)

def train_suitability_model():
    """
    Train and save the location suitability ML model
    """
    print("Loading training data...")
    df = load_training_data()

    # Prepare features and target
    feature_cols = [
        'competitors_within_500m', 'competitors_within_200m', 'competitors_min_distance', 'competitors_avg_distance',
        'roads_within_500m', 'roads_avg_distance',
        'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
        'hospitals_within_500m', 'hospitals_min_distance',
        'bus_stops_within_500m', 'bus_stops_min_distance',
        'population_density_proxy', 'accessibility_score', 'foot_traffic_score', 'competition_pressure'
    ]

    X = df[feature_cols]
    y = df['suitability']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance on Test Set:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save model, encoder, and scaler
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'suitability_rf_model.pkl')
    encoder_path = os.path.join(models_dir, 'suitability_label_encoder.pkl')
    scaler_path = os.path.join(models_dir, 'suitability_scaler.pkl')

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Scaler saved to: {scaler_path}")

    return model, label_encoder, scaler

if __name__ == "__main__":
    train_suitability_model()