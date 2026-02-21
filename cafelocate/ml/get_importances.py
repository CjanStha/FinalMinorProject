import joblib

model = joblib.load('models/suitability_rf_model.pkl')
feature_cols = [
    'competitors_within_500m', 'competitors_within_200m', 'competitors_min_distance', 'competitors_avg_distance',
    'roads_within_500m', 'roads_avg_distance',
    'schools_within_500m', 'schools_within_200m', 'schools_min_distance',
    'hospitals_within_500m', 'hospitals_min_distance',
    'bus_stops_within_500m', 'bus_stops_min_distance',
    'population_density_proxy', 'accessibility_score', 'foot_traffic_score', 'competition_pressure'
]
importances = model.feature_importances_
print('Feature Importances (sorted by importance):')
for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
    print(f'{name}: {imp:.4f}')

print('\nFeatures with importance < 0.01 (low impact):')
low_impact = [name for name, imp in zip(feature_cols, importances) if imp < 0.01]
print(low_impact)