import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

def preprocess_training_data():
    """
    Preprocess the training dataset by:
    1. Removing low-impact features
    2. Handling class imbalance with SMOTE
    3. Removing outliers
    4. Scaling features
    """

    # Load data
    data_path = os.path.join('..', 'data', 'cafe_location_training_dataset.csv')
    df = pd.read_csv(data_path)

    print(f"Original dataset shape: {df.shape}")
    print("Original class distribution:")
    print(df['suitability'].value_counts())

    # Features to keep (remove low-impact ones)
    low_impact_features = [
        'competitors_within_500m',
        'competitors_within_200m',
        'competitors_avg_distance',
        'bus_stops_within_500m',
        'bus_stops_min_distance',
        'competition_pressure'
    ]

    keep_features = [
        'competitors_min_distance',
        'roads_within_500m', 'roads_avg_distance',
        'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
        'hospitals_within_500m', 'hospitals_min_distance',
        'population_density_proxy', 'accessibility_score', 'foot_traffic_score'
    ]

    # Prepare features and target
    X = df[keep_features]
    y = df['suitability']

    # Remove outliers using IQR method
    def remove_outliers_iqr(data, factor=1.5):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]

    print(f"\nBefore outlier removal: {X.shape}")
    X_clean = remove_outliers_iqr(X)
    y_clean = y.loc[X_clean.index]
    print(f"After outlier removal: {X_clean.shape}")

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_clean)

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=42, k_neighbors=3)  # k_neighbors=3 for minority class
    X_balanced, y_balanced = smote.fit_resample(X_clean, y_encoded)

    print(f"\nAfter SMOTE balancing: {X_balanced.shape}")
    print("Balanced class distribution:")
    unique, counts = np.unique(y_balanced, return_counts=True)
    for label, count in zip(label_encoder.inverse_transform(unique), counts):
        print(f"{label}: {count}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    # Create preprocessed dataframe
    preprocessed_df = pd.DataFrame(X_scaled, columns=keep_features)
    preprocessed_df['suitability'] = label_encoder.inverse_transform(y_balanced)

    # Duplicate coordinates to match the balanced dataset size
    n_original = len(X_clean)
    n_balanced = len(X_balanced)
    coords_repeated = pd.concat([df.loc[X_clean.index, ['latitude', 'longitude']]] * (n_balanced // n_original), ignore_index=True)
    coords_extra = df.loc[X_clean.index, ['latitude', 'longitude']].head(n_balanced % n_original)
    coords_final = pd.concat([coords_repeated, coords_extra], ignore_index=True)

    preprocessed_df['latitude'] = coords_final['latitude'].values
    preprocessed_df['longitude'] = coords_final['longitude'].values

    # Save preprocessed dataset
    output_path = os.path.join('..', 'data', 'preprocessed_training_dataset.csv')
    preprocessed_df.to_csv(output_path, index=False)

    print(f"\nPreprocessed dataset saved to: {output_path}")
    print(f"Final shape: {preprocessed_df.shape}")
    print(f"Features kept: {keep_features}")

    return preprocessed_df

if __name__ == "__main__":
    preprocess_training_data()