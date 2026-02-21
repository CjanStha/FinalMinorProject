# CafeLocate ML

A machine learning-powered café location recommendation system for Kathmandu, Nepal. Uses spatial data analysis, Google Places API, and Random Forest classification to suggest optimal café locations based on competitor density, road access, and population demographics.

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- PostgreSQL 16 with PostGIS
- Google Cloud Console account (for API keys)

### Setup
1. **Clone and navigate:**
   ```bash
   git clone <your-repo-url>
   cd cafelocate
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Set up database:**
   - Install PostgreSQL and PostGIS
   - Create database `cafelocate_db` and user `cafelocate_user`
   - Enable PostGIS extension

5. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and database credentials
   ```

6. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

7. **Start the servers:**
   ```bash
   # Terminal 1: Backend
   python manage.py runserver

   # Terminal 2: Frontend (use VS Code Live Server or Python server)
   cd ../frontend
   python -m http.server 5500
   ```

8. **Open in browser:**
   - Frontend: http://localhost:5500/map.html
   - Backend API: http://localhost:8000/

## 📁 Project Structure

```
cafelocate/
├── backend/                 # Django backend
│   ├── api/                # REST API endpoints
│   ├── ml_engine/          # ML prediction service
│   ├── cafelocate/         # Django settings
│   └── requirements.txt
├── frontend/               # HTML/CSS/JS map interface
├── ml/                     # ML training scripts
├── data/                   # Raw datasets
└── README.md
```

## 🛠 Development Workflow

- **Backend API:** Django REST Framework at `localhost:8000`
- **Frontend:** Vanilla JS with Leaflet maps at `localhost:5500`
- **Database:** PostgreSQL with PostGIS for spatial queries
- **ML Model:** Random Forest classifier saved as `.pkl`

## 👥 Team

- **Santosh Mahato Koiri** - Backend API Development
- **Sijan Shrestha** - Frontend & UI/UX
- **Upendra Dhungana** - Machine Learning & Data Science

## 📋 Phases

- **Phase 1:** Environment setup (✅ Complete)
- **Phase 2:** Data collection & preprocessing
- **Phase 3:** ML model training
- **Phase 4:** API development
- **Phase 5:** Frontend integration
- **Phase 6:** Testing & deployment

## 🔧 Tech Stack

- **Backend:** Django 4.2, Django REST Framework
- **Database:** PostgreSQL + PostGIS
- **ML:** scikit-learn, pandas, numpy
- **Frontend:** HTML5, CSS3, JavaScript, Leaflet.js
- **APIs:** Google Places API, Google OAuth
- **Deployment:** Docker (planned)

## 📊 Data Sources

- Kathmandu café locations (Google Places API)
- Road network data (OpenStreetMap)
- Population demographics (Kathmandu census data)
- Ward boundaries (GeoJSON)