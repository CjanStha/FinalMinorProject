"""
Train ML model for café type prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def create_synthetic_training_data():
    """
    Create synthetic training data for café type prediction
    Based on the features we use in the API: competitor_count, avg_rating, road_length_m, population_density
    """
    np.random.seed(42)

    # Create 1000 synthetic data points
    n_samples = 1000

    data = []

    for i in range(n_samples):
        # Generate features
        competitor_count = np.random.randint(0, 30)  # 0-29 competitors
        avg_rating = np.random.uniform(3.0, 5.0)  # 3.0-5.0 rating
        road_length_m = np.random.uniform(0, 5000)  # 0-5000m road length
        population_density = np.random.uniform(1000, 20000)  # 1000-20000 people/km²

        # Determine café type based on features (simplified logic)
        if population_density > 12000 and competitor_count < 10:
            # High density, low competition -> Coffee Shop
            cafe_type = 'coffee_shop'
        elif road_length_m > 2500 and avg_rating > 4.0:
            # Good road access, high ratings -> Bakery
            cafe_type = 'bakery'
        elif competitor_count > 15 and population_density > 8000:
            # High competition, decent density -> Dessert Shop
            cafe_type = 'dessert_shop'
        else:
            # Default -> Restaurant
            cafe_type = 'restaurant'

        data.append({
            'competitor_count': competitor_count,
            'avg_competitor_rating': avg_rating,
            'road_length_m': road_length_m,
            'population_density': population_density,
            'cafe_type': cafe_type
        })

    return pd.DataFrame(data)

def train_model():
    """
    Train and save the ML model
    """
    print("Creating synthetic training data...")
    df = create_synthetic_training_data()

    print(f"Training data shape: {df.shape}")
    print("Class distribution:")
    print(df['cafe_type'].value_counts())

    # Prepare features and target
    feature_cols = ['competitor_count', 'avg_competitor_rating', 'road_length_m', 'population_density']
    X = df[feature_cols]
    y = df['cafe_type']

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nLabel mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save model and encoder
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'rf_model.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")

    return model, label_encoder

if __name__ == "__main__":
    train_model()