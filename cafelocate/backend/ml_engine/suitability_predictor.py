import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / 'ml' / 'models'

MODEL_CANDIDATES = [
    {
        'name': 'ahp_tuned_v2',
        'rf_path': MODELS_DIR / 'rf_suitability_v2_ahp_tuned.pkl',
        'xgb_path': MODELS_DIR / 'xgb_suitability_ahp.pkl',
        'scaler_path': MODELS_DIR / 'scaler_suitability.pkl',
        'features_path': MODELS_DIR / 'feature_columns_suitability.pkl',
    },
]

DEFAULT_FEATURES = [
    'population_density', 'accessibility_score', 'foot_traffic_score',
    'competition_effective', 'bus_stops_within_500m',
    'osm_amenity_density_500m', 'nearby_schools', 'nearby_hospitals'
]

_rf_model = None
_xgb_model = None
_scaler = None
_feature_columns = None
_active_model_name = None


def _load_models():
    global _rf_model, _xgb_model, _scaler, _feature_columns, _active_model_name
    if _scaler is not None and _feature_columns is not None and (_rf_model is not None or _xgb_model is not None):
        return

    for candidate in MODEL_CANDIDATES:
        try:
            scaler = joblib.load(candidate['scaler_path'])
            feature_columns = joblib.load(candidate['features_path'])
        except FileNotFoundError:
            continue

        rf_model = None
        xgb_model = None

        try:
            rf_model = joblib.load(candidate['rf_path'])
            if hasattr(rf_model, 'n_jobs'):
                rf_model.n_jobs = 1
        except Exception as exc:
            logger.warning(f"{candidate['name']} Random Forest regressor could not be loaded: {exc}")

        try:
            xgb_model = joblib.load(candidate['xgb_path'])
            if hasattr(xgb_model, 'n_jobs'):
                xgb_model.n_jobs = 1
        except Exception as exc:
            logger.warning(f"{candidate['name']} XGBoost regressor could not be loaded: {exc}")

        if rf_model is None and xgb_model is None:
            continue

        _scaler = scaler
        _feature_columns = feature_columns
        _rf_model = rf_model
        _xgb_model = xgb_model
        _active_model_name = candidate['name']

        if _rf_model is not None and _xgb_model is not None:
            logger.info(f'Regression suitability ensemble loaded successfully: {_active_model_name}')
        elif _rf_model is not None:
            logger.info(f'Regression suitability Random Forest model loaded successfully: {_active_model_name}')
        else:
            logger.info(f'Regression suitability XGBoost model loaded successfully: {_active_model_name}')
        return

    logger.warning('Regression preprocessing artifacts not found. Using fallback scoring.')
    _rf_model = None
    _xgb_model = None
    _scaler = None
    _feature_columns = DEFAULT_FEATURES
    _active_model_name = None


def _score_to_level(score):
    if score >= 7:
        return 'High Suitability'
    if score >= 4:
        return 'Medium Suitability'
    return 'Low Suitability'


def _build_feature_array(features_dict):
    feature_columns = _feature_columns or DEFAULT_FEATURES
    values = [float(features_dict.get(feature, 0.0)) for feature in feature_columns]
    return pd.DataFrame([values], columns=feature_columns, dtype=float), feature_columns


def _fallback_score(features_dict):
    population_component = min(10.0, float(features_dict.get('population_density', 0.0))) * 0.25
    accessibility_component = min(10.0, float(features_dict.get('accessibility_score', 0.0))) * 0.20
    foot_traffic_component = min(10.0, float(features_dict.get('foot_traffic_score', 0.0))) * 0.20
    competition_penalty = min(10.0, float(features_dict.get('competition_effective', 0.0))) * 0.15
    bus_component = min(10.0, float(features_dict.get('bus_stops_within_500m', 0.0)) * 0.1) * 0.10
    amenity_component = min(10.0, float(features_dict.get('osm_amenity_density_500m', 0.0)) * 0.01) * 0.05
    schools_component = min(10.0, float(features_dict.get('nearby_schools', 0.0)) * 0.1) * 0.03
    hospitals_component = min(10.0, float(features_dict.get('nearby_hospitals', 0.0)) * 0.1) * 0.02

    score = (
        population_component +
        accessibility_component +
        foot_traffic_component +
        bus_component +
        amenity_component +
        schools_component +
        hospitals_component -
        competition_penalty
    )
    return float(max(0.0, min(10.0, score)))


def get_suitability_prediction(features_dict):
    """
    Predict suitability scores using both Random Forest and XGBoost models.
    Returns individual model predictions for frontend comparison.
    """
    _load_models()

    try:
        features_array, feature_columns = _build_feature_array(features_dict)

        if (_rf_model is None and _xgb_model is None) or _scaler is None:
            score = _fallback_score(features_dict)
            return {
                'predicted_score': score,
                'predicted_suitability': _score_to_level(score),
                'confidence': 0.0,
                'model_type': 'regression_fallback',
                'features_used': len(feature_columns),
                'random_forest_score': None,
                'xgboost_score': None,
                'ensemble_score': score,
            }

        features_scaled = pd.DataFrame(
            _scaler.transform(features_array),
            columns=feature_columns,
        )

        rf_score = None
        xgb_score = None
        ensemble_score = None

        if _rf_model is not None:
            rf_score = float(_rf_model.predict(features_scaled)[0])
        if _xgb_model is not None:
            xgb_score = float(_xgb_model.predict(features_scaled)[0])

        # Calculate ensemble if both models are available
        if rf_score is not None and xgb_score is not None:
            ensemble_score = float(np.clip((rf_score + xgb_score) / 2, 0.0, 10.0))
            confidence = float(max(0.0, min(1.0, 1.0 - abs(rf_score - xgb_score) / 10.0)))
            model_type = 'regression_ensemble_v3'
        elif rf_score is not None:
            ensemble_score = rf_score
            confidence = 0.75
            model_type = 'regression_rf_only_v3'
        elif xgb_score is not None:
            ensemble_score = xgb_score
            confidence = 0.75
            model_type = 'regression_xgb_only_v3'
        else:
            ensemble_score = _fallback_score(features_dict)
            confidence = 0.0
            model_type = 'regression_fallback'

        return {
            'predicted_score': round(ensemble_score, 2),  # Keep for backward compatibility
            'predicted_suitability': _score_to_level(ensemble_score),
            'confidence': round(confidence, 3),
            'model_type': model_type,
            'model_variant': _active_model_name,
            'features_used': len(feature_columns),
            'random_forest_score': round(rf_score, 2) if rf_score is not None else None,
            'xgboost_score': round(xgb_score, 2) if xgb_score is not None else None,
            'ensemble_score': round(ensemble_score, 2),
        }
    except Exception as exc:
        logger.error(f'Error in regression suitability prediction: {exc}')
        score = _fallback_score(features_dict)
        return {
            'predicted_score': round(score, 2),
            'predicted_suitability': _score_to_level(score),
            'confidence': 0.0,
            'model_type': 'regression_error_fallback',
            'features_used': len(_feature_columns or DEFAULT_FEATURES),
            'random_forest_score': None,
            'xgboost_score': None,
            'ensemble_score': round(score, 2),
            'error': str(exc),
        }
