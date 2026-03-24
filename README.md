# CafeLocate

CafeLocate is a student project for analyzing potential cafe locations in Kathmandu. It combines geospatial data, demographic signals, nearby amenities, and machine learning to estimate a suitability score for a selected map location.

The main project lives in [`MP/cafelocate`](c:/Users/v15/Desktop/CJN3/MP/cafelocate).

## What It Does

- Lets a user pick a cafe type and pin a location on an interactive map
- Analyzes nearby cafes, roads, population density, bus stops, schools, and hospitals
- Computes a suitability score on a `0-10` scale
- Shows Random Forest, XGBoost, and ensemble model outputs
- Recommends the best cafe type to open for that location
- Supports login, demo mode, and analysis history

## Repository Layout

- [`MP/cafelocate/frontend`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/frontend): static frontend with map UI
- [`MP/cafelocate/backend`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/backend): Django + DRF backend
- [`MP/cafelocate/ml`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/ml): notebooks, trained models, and ML documentation
- [`MP/cafelocate/data/raw_data`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/data/raw_data): project datasets

## Tech Stack

- Frontend: HTML, CSS, JavaScript, Leaflet
- Backend: Django, Django REST Framework
- ML: scikit-learn, XGBoost, pandas, NumPy
- Spatial / GIS: Shapely, GeoPandas, GDAL, Fiona, PyProj
- Database support: PostgreSQL / PostGIS-oriented dependencies

## How Scoring Works

At a high level, the backend:

1. Computes location features such as accessibility, foot-traffic proxy, amenity density, and competition pressure.
2. Normalizes those features into the range expected by the trained models.
3. Predicts suitability using Random Forest and XGBoost.
4. Averages the model outputs into one final ensemble suitability score.
5. Returns the score and label to the frontend, which displays it in the score circle and model comparison cards.

Detailed documentation is available in:

- [`MP/cafelocate/ml/suitability_score_documentation.ipynb`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/ml/suitability_score_documentation.ipynb)

## Running the Project

### Backend

From `MP/cafelocate/backend`:

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

The backend will usually run at `http://127.0.0.1:8000/`.

### Frontend

Open the frontend from [`MP/cafelocate/frontend/index.html`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/frontend/index.html) using a local static server or your editor's live server.

If you use VS Code Live Server, the frontend commonly runs at something like:

```text
http://127.0.0.1:5500/MP/cafelocate/frontend/
```

## Data Files

Some important raw datasets used in the project:

- [`dataset_ft_enriched.csv`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/data/raw_data/dataset_ft_enriched.csv)
- [`kathmandu_census.csv`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/data/raw_data/kathmandu_census.csv)
- [`kathmandu_cafes.csv`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/data/raw_data/kathmandu_cafes.csv)
- [`osm_roads_kathmandu.csv`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/data/raw_data/osm_roads_kathmandu.csv)
- [`osm_amenities_kathmandu.csv`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/data/raw_data/osm_amenities_kathmandu.csv)

## Models

Saved model artifacts are stored in:

- [`MP/cafelocate/ml/models`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/ml/models)

These include:

- Random Forest suitability model
- XGBoost suitability model
- feature column metadata
- scaler used before prediction
- tuned AHP-related weights

## Notes

- This is a student project, so some scoring rules are literature-inspired heuristics rather than formulas copied directly from one paper.
- GIS-related dependencies such as GDAL can need extra setup, especially on Windows.
- The project includes both heuristic scoring logic and ML-based scoring, but the frontend mainly shows the ML ensemble result.

## Testing

From `MP/cafelocate/backend`:

```bash
python manage.py test api.tests
```

## Authoring Tip

If you are extending this repository, start with:

- [`MP/cafelocate/backend/api/views.py`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/backend/api/views.py)
- [`MP/cafelocate/backend/ml_engine/suitability_predictor.py`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/backend/ml_engine/suitability_predictor.py)
- [`MP/cafelocate/frontend/js/map.js`](c:/Users/v15/Desktop/CJN3/MP/cafelocate/frontend/js/map.js)

