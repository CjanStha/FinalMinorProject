import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Path to the trained model files ──────────────────────────────────────────
# These .pkl files are created by ml/train_model.py
# They live in ml/models/ folder and are copied here after training
BASE_DIR      = Path(__file__).resolve().parent
MODEL_PATH    = BASE_DIR / 'models' / 'rf_model.pkl'
ENCODER_PATH  = BASE_DIR / 'models' / 'label_encoder.pkl'

# ── Load model once at import time ────────────────────────────────────────────
# Module-level variables = loaded ONCE when Django starts, not on each request.
# This is critical for performance — loading a model takes ~1 second.
_model   = None
_encoder = None

def _load_model():
    """Load model from disk if not already loaded (lazy loading)."""
    global _model, _encoder
    if _model is None:
        try:
            # For now, use mock model until scikit-learn models are available
            _model = "mock_model"
            _encoder = "mock_encoder"
            logger.info("Using mock ML model (real models not available).")
        except FileNotFoundError:
            logger.warning("ML models not found. Using mock predictions.")


# ── Feature names — must match what the model was trained on ──────────────────
FEATURE_NAMES = [
    'competitor_count',     # number of cafes within 500m
    'avg_competitor_rating', # average rating of those cafes
    'road_length_m',         # total road length in buffer (meters)
    'population_density',    # people per sq km in the ward
]

# ── Human-readable labels ─────────────────────────────────────────────────────
CAFE_TYPE_LABELS = {
    'coffee_shop':   'Coffee Shop',
    'bakery':        'Bakery Café',
    'dessert_shop':  'Dessert Shop',
    'restaurant':    'Restaurant Café',
}


def get_prediction(features: list) -> dict:
    """
    Run prediction from the trained Random Forest model.

    Args:
        features: [competitor_count, avg_rating, road_length_m, pop_density]

    Returns:
        { 'predicted_type': 'Bakery Café', 'confidence': 0.87,
          'all_probabilities': {'Coffee Shop': 0.08, 'Bakery Café': 0.87, ...} }
    """
    _load_model()

    if _model is None or _model == "mock_model":
        # Mock prediction logic based on features
        competitor_count, avg_rating, road_length_m, pop_density = features

        # Simple rule-based prediction
        if pop_density > 12000 and competitor_count < 10:
            predicted_type = 'coffee_shop'
            confidence = 0.75
        elif road_length_m > 2500 and avg_rating > 4.0:
            predicted_type = 'bakery'
            confidence = 0.80
        elif competitor_count > 15 and pop_density > 8000:
            predicted_type = 'dessert_shop'
            confidence = 0.70
        else:
            predicted_type = 'restaurant'
            confidence = 0.65

        # Mock probabilities
        all_probs = {
            'Coffee Shop': 0.25,
            'Bakery Café': 0.25,
            'Dessert Shop': 0.25,
            'Restaurant Café': 0.25
        }
        all_probs[CAFE_TYPE_LABELS[predicted_type]] = confidence
        # Normalize
        total = sum(all_probs.values())
        all_probs = {k: round(v/total, 3) for k, v in all_probs.items()}

        return {
            'predicted_type':    CAFE_TYPE_LABELS[predicted_type],
            'confidence':        confidence,
            'all_probabilities':  all_probs,
        }

    # Original ML model code (when models are available)
    # Convert feature list to 2D array (sklearn expects shape: [n_samples, n_features])
    # X = np.array([features])    # shape: (1, 4)
    #
    # # Get predicted class (integer) and convert to label string
    # predicted_int   = _model.predict(X)[0]
    # predicted_label = _encoder.inverse_transform([predicted_int])[0]
    #
    # # Get probabilities for all classes
    # probabilities = _model.predict_proba(X)[0]
    # class_labels  = _encoder.inverse_transform(_model.classes_)
    #
    # all_probs = {
    #     CAFE_TYPE_LABELS.get(label, label): round(float(prob), 3)
    #     for label, prob in zip(class_labels, probabilities)
    # }
    #
    # return {
    #     'predicted_type':    CAFE_TYPE_LABELS.get(predicted_label, predicted_label),
    #     'confidence':        round(float(max(probabilities)), 3),
    #     'all_probabilities':  all_probs,
    # }
    """
    Run prediction from the trained Random Forest model.

    Args:
        features: [competitor_count, avg_rating, road_length_m, pop_density]

    Returns:
        { 'predicted_type': 'Bakery Café', 'confidence': 0.87,
          'all_probabilities': {'Coffee Shop': 0.08, 'Bakery Café': 0.87, ...} }
    """
    _load_model()

    if _model is None:
        # Model not trained yet — return a safe fallback
        return {
            'predicted_type':    'Model not trained yet',
            'confidence':        0,
            'all_probabilities': {},
        }

    # Convert feature list to 2D array (sklearn expects shape: [n_samples, n_features])
    X = np.array([features])    # shape: (1, 4)

    # Get predicted class (integer) and convert to label string
    predicted_int   = _model.predict(X)[0]
    predicted_label = _encoder.inverse_transform([predicted_int])[0]

    # Get probabilities for all classes
    # predict_proba returns [[0.05, 0.87, 0.03, 0.05]] for 4 classes
    probabilities = _model.predict_proba(X)[0]
    class_labels  = _encoder.inverse_transform(_model.classes_)

    all_probs = {
        CAFE_TYPE_LABELS.get(label, label): round(float(prob), 3)
        for label, prob in zip(class_labels, probabilities)
    }

    return {
        'predicted_type':    CAFE_TYPE_LABELS.get(predicted_label, predicted_label),
        'confidence':        round(float(max(probabilities)), 3),
        'all_probabilities':  all_probs,
    }
    # Example output:
    # { "predicted_type": "Bakery Café",
    #   "confidence": 0.87,
    #   "all_probabilities": {
    #     "Coffee Shop": 0.05, "Bakery Café": 0.87,
    #     "Dessert Shop": 0.03, "Restaurant Café": 0.05
    #   }
    # }