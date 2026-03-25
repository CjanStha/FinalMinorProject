import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from django.test import TestCase

from .suitability_predictor import (
    get_suitability_prediction,
    _build_feature_array,
    _fallback_score,
    _score_to_level,
    _load_models,
    DEFAULT_FEATURES,
)


class SuitabilityPredictorUnitTests(TestCase):
    """
    Unit tests for the ML engine suitability predictor module.
    Tests model loading, feature engineering, and prediction logic.
    """

    def setUp(self):
        """Reset global model state before each test."""
        import ml_engine.suitability_predictor as predictor_module
        predictor_module._rf_model = None
        predictor_module._xgb_model = None
        predictor_module._scaler = None
        predictor_module._feature_columns = None
        predictor_module._active_model_name = None

    # ─────────────────────────────────────────────
    # Test: Score to Level Classification
    # ─────────────────────────────────────────────

    def test_score_to_level_high_suitability(self):
        """Test high suitability classification (score >= 7)."""
        self.assertEqual(_score_to_level(7.0), 'High Suitability')
        self.assertEqual(_score_to_level(9.5), 'High Suitability')
        self.assertEqual(_score_to_level(10.0), 'High Suitability')

    def test_score_to_level_medium_suitability(self):
        """Test medium suitability classification (4 <= score < 7)."""
        self.assertEqual(_score_to_level(4.0), 'Medium Suitability')
        self.assertEqual(_score_to_level(5.5), 'Medium Suitability')
        self.assertEqual(_score_to_level(6.9), 'Medium Suitability')

    def test_score_to_level_low_suitability(self):
        """Test low suitability classification (score < 4)."""
        self.assertEqual(_score_to_level(0.0), 'Low Suitability')
        self.assertEqual(_score_to_level(3.0), 'Low Suitability')
        self.assertEqual(_score_to_level(3.99), 'Low Suitability')

    # ─────────────────────────────────────────────
    # Test: Feature Array Building
    # ─────────────────────────────────────────────

    def test_build_feature_array_with_complete_features(self):
        """Test feature array construction with all required features."""
        features_dict = {
            'population_density': 5.0,
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        array, columns = _build_feature_array(features_dict)
        
        self.assertIsInstance(array, pd.DataFrame)
        self.assertEqual(len(columns), len(DEFAULT_FEATURES))
        self.assertEqual(array.shape[0], 1)  # Single sample
        self.assertEqual(array.shape[1], len(DEFAULT_FEATURES))

    def test_build_feature_array_with_missing_features(self):
        """Test feature array with missing features (should default to 0.0)."""
        features_dict = {
            'population_density': 5.0,
            'accessibility_score': 6.0,
        }
        
        array, columns = _build_feature_array(features_dict)
        
        self.assertEqual(array.shape[0], 1)
        self.assertEqual(array.shape[1], len(DEFAULT_FEATURES))
        # Check that missing features are set to 0.0
        self.assertTrue((array.iloc[0, 2:] == 0.0).all())

    def test_build_feature_array_empty_dict(self):
        """Test feature array with empty features dictionary."""
        array, columns = _build_feature_array({})
        
        self.assertEqual(array.shape[0], 1)
        self.assertEqual(array.shape[1], len(DEFAULT_FEATURES))
        self.assertTrue((array.iloc[0] == 0.0).all())

    # ─────────────────────────────────────────────
    # Test: Fallback Scoring
    # ─────────────────────────────────────────────

    def test_fallback_score_high_population_low_competition(self):
        """Test fallback score with high population and low competition."""
        features_dict = {
            'population_density': 8.0,
            'accessibility_score': 7.0,
            'foot_traffic_score': 6.0,
            'competition_effective': 8.0,  # Low competition (high effective value)
            'bus_stops_within_500m': 5.0,
            'osm_amenity_density_500m': 4.0,
            'nearby_schools': 6.0,
            'nearby_hospitals': 5.0,
        }
        
        score = _fallback_score(features_dict)
        
        # Should be reasonably high
        self.assertGreater(score, 5.0)
        self.assertLessEqual(score, 10.0)

    def test_fallback_score_bounds_between_0_and_10(self):
        """Test that fallback score is always between 0 and 10."""
        test_cases = [
            {i: v for i, v in enumerate(range(20))},  # Very high values
            {i: v for i, v in enumerate(range(-20, 0))},  # Negative values
            {'population_density': 100, 'competition_effective': -50},  # Mixed extremes
        ]
        
        for features in test_cases:
            # Convert to proper feature names
            features_dict = {
                'population_density': features.get('population_density', 5.0),
                'accessibility_score': features.get('accessibility_score', 5.0),
                'foot_traffic_score': features.get('foot_traffic_score', 5.0),
                'competition_effective': features.get('competition_effective', 5.0),
                'bus_stops_within_500m': features.get('bus_stops_within_500m', 5.0),
                'osm_amenity_density_500m': features.get('osm_amenity_density_500m', 5.0),
                'nearby_schools': features.get('nearby_schools', 5.0),
                'nearby_hospitals': features.get('nearby_hospitals', 5.0),
            }
            score = _fallback_score(features_dict)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 10.0)

    def test_fallback_score_zero_features(self):
        """Test fallback score with all zero features."""
        features_dict = {
            'population_density': 0,
            'accessibility_score': 0,
            'foot_traffic_score': 0,
            'competition_effective': 0,
            'bus_stops_within_500m': 0,
            'osm_amenity_density_500m': 0,
            'nearby_schools': 0,
            'nearby_hospitals': 0,
        }
        
        score = _fallback_score(features_dict)
        
        # With all zeros and zero competition, score should be very low
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 3.0)

    # ─────────────────────────────────────────────
    # Test: Model Prediction (No Models Loaded)
    # ─────────────────────────────────────────────

    def test_prediction_fallback_when_no_models_available(self):
        """Test prediction falls back when no ML models are loaded."""
        features_dict = {
            'population_density': 5.0,
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        result = get_suitability_prediction(features_dict)
        
        self.assertIn('predicted_score', result)
        self.assertIn('predicted_suitability', result)
        self.assertIn('model_type', result)
        self.assertEqual(result['model_type'], 'regression_fallback')
        self.assertIsNone(result['random_forest_score'])
        self.assertIsNone(result['xgboost_score'])
        self.assertGreaterEqual(result['predicted_score'], 0.0)
        self.assertLessEqual(result['predicted_score'], 10.0)

    # ─────────────────────────────────────────────
    # Test: Model Prediction (Ensemble Mock)
    # ─────────────────────────────────────────────

    @patch('ml_engine.suitability_predictor.joblib.load')
    def test_prediction_ensemble_when_both_models_loaded(self, mock_joblib_load):
        """Test ensemble prediction when both RF and XGBoost models are available."""
        # Mock models
        mock_rf = MagicMock()
        mock_rf.predict.return_value = np.array([7.5])
        mock_rf.n_jobs = 1
        
        mock_xgb = MagicMock()
        mock_xgb.predict.return_value = np.array([6.5])
        mock_xgb.n_jobs = 1
        
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[5, 6, 4, 3, 2, 7, 8, 5.5]])
        
        mock_feature_cols = ['population_density', 'accessibility_score', 'foot_traffic_score',
                            'competition_effective', 'bus_stops_within_500m',
                            'osm_amenity_density_500m', 'nearby_schools', 'nearby_hospitals']
        
        # Setup joblib.load to return mocks in the right order
        def load_side_effect(path):
            path_str = str(path)
            if 'scaler' in path_str:
                return mock_scaler
            elif 'feature_columns' in path_str:
                return mock_feature_cols
            elif 'rf_' in path_str:
                return mock_rf
            elif 'xgb_' in path_str:
                return mock_xgb
            raise FileNotFoundError(f"Mock file not found: {path}")
        
        mock_joblib_load.side_effect = load_side_effect
        
        features_dict = {
            'population_density': 5.0,
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        result = get_suitability_prediction(features_dict)
        
        # Verify ensemble prediction
        self.assertEqual(result['model_type'], 'regression_ensemble_v3')
        self.assertEqual(result['random_forest_score'], 7.5)
        self.assertEqual(result['xgboost_score'], 6.5)
        self.assertEqual(result['ensemble_score'], 7.0)  # (7.5 + 6.5) / 2
        self.assertGreater(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)

    # ─────────────────────────────────────────────
    # Test: Edge Cases and Error Handling
    # ─────────────────────────────────────────────

    def test_prediction_handles_invalid_feature_types(self):
        """Test that prediction handles non-numeric feature values gracefully."""
        features_dict = {
            'population_density': 'invalid',  # String instead of number
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        # Should handle gracefully by falling back
        result = get_suitability_prediction(features_dict)
        self.assertIn('predicted_score', result)
        self.assertGreaterEqual(result['predicted_score'], 0.0)
        self.assertLessEqual(result['predicted_score'], 10.0)

    def test_prediction_handles_nan_values(self):
        """Test that prediction handles NaN values in features."""
        features_dict = {
            'population_density': float('nan'),
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        result = get_suitability_prediction(features_dict)
        self.assertIn('predicted_score', result)
        # Should be a valid prediction result
        self.assertTrue(isinstance(result['predicted_score'], (int, float)))

    def test_prediction_with_empty_features(self):
        """Test prediction with empty features dictionary."""
        result = get_suitability_prediction({})
        
        self.assertIn('predicted_score', result)
        self.assertIn('predicted_suitability', result)
        self.assertGreaterEqual(result['predicted_score'], 0.0)
        self.assertLessEqual(result['predicted_score'], 10.0)

    # ─────────────────────────────────────────────
    # Test: Response Structure
    # ─────────────────────────────────────────────

    def test_prediction_response_structure(self):
        """Test that prediction response has all required fields."""
        features_dict = {
            'population_density': 5.0,
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        result = get_suitability_prediction(features_dict)
        
        required_fields = [
            'predicted_score',
            'predicted_suitability',
            'confidence',
            'model_type',
            'features_used',
            'random_forest_score',
            'xgboost_score',
            'ensemble_score',
        ]
        
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")

    def test_prediction_score_is_rounded(self):
        """Test that prediction scores are properly rounded."""
        features_dict = {
            'population_density': 5.0,
            'accessibility_score': 6.0,
            'foot_traffic_score': 4.0,
            'competition_effective': 3.0,
            'bus_stops_within_500m': 2.0,
            'osm_amenity_density_500m': 7.0,
            'nearby_schools': 8.0,
            'nearby_hospitals': 5.5,
        }
        
        result = get_suitability_prediction(features_dict)
        
        # Check that scores are rounded to 2 decimal places
        predicted = result['predicted_score']
        ensemble = result['ensemble_score']
        confidence = result['confidence']
        
        # Convert to string and check decimal places
        self.assertLessEqual(len(str(predicted).split('.')[-1]) if '.' in str(predicted) else 0, 2)
        self.assertLessEqual(len(str(ensemble).split('.')[-1]) if '.' in str(ensemble) else 0, 2)
        self.assertLessEqual(len(str(confidence).split('.')[-1]) if '.' in str(confidence) else 0, 3)
