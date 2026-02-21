# CafeLocate ML

A machine learning-powered café location recommendation system for Kathmandu, Nepal. Uses real spatial data analysis, Mapbox Geocoding API, OpenStreetMap, and Random Forest classification to suggest optimal café locations based on competitor density, road access, and population demographics.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (3.11 recommended)
- Git
- Web browser (Chrome, Firefox, Safari, or Edge)
- Internet connection (for API calls and data fetching)

### Automated Setup (Recommended)

For quick setup, use the provided setup scripts:

**Windows:**
```bash
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Create and activate a virtual environment
- Install all required dependencies
- Set up the database
- Create the `.env` file from template

### Manual Setup

If you prefer to set up manually, follow these steps:
   ```bash
   git clone <your-github-repo-url>
   cd MP2  # Navigate to the project directory
   ```

2. **Set up Python virtual environment:**
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install backend dependencies:**
   ```bash
   cd cafelocate/backend
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cd cafelocate
   copy .env.example .env  # Windows
   # cp .env.example .env  # macOS/Linux
   ```

   Edit the `.env` file and add your Mapbox API token:
   ```
   SECRET_KEY=your-django-secret-key-here
   DEBUG=True
   ALLOWED_HOSTS=localhost,127.0.0.1
   MAPBOX_ACCESS_TOKEN=your-mapbox-token-here
   ```

5. **Run database migrations:**
   ```bash
   cd backend
   python manage.py migrate
   ```

6. **Load initial data (optional - for testing):**
   ```bash
   # Load café data
   python manage.py shell -c "from ml.load_cafes import load_cafes; load_cafes()"

   # Load census data
   python manage.py shell -c "from ml.load_census import load_census; load_census()"

   # Load ward boundaries
   python manage.py shell -c "from ml.load_wards import load_wards; load_wards()"
   ```

7. **Start the application:**
   ```bash
   # Windows
   run.bat

   # macOS/Linux
   ./run.sh
   ```

   This will start both the backend (Django) and frontend (HTTP server) simultaneously.

8. **Open your browser and navigate to:**
   - **Main Application:** http://localhost:5500
   - **API Documentation:** http://localhost:8000/

### Alternative Manual Startup

If you prefer to start servers manually:api/

## 📁 Project Structure

```
MP2/
├── cafelocate/              # Main project directory
│   ├── backend/            # Django REST API backend
│   │   ├── api/           # API endpoints and models
│   │   ├── ml_engine/     # ML prediction service
│   │   ├── cafelocate/    # Django settings
│   │   ├── db.sqlite3     # SQLite database
│   │   └── requirements.txt
│   ├── frontend/          # HTML/CSS/JS frontend
│   │   ├── index.html    # Main application
│   │   ├── css/          # Stylesheets
│   │   └── js/           # JavaScript modules
│   ├── ml/               # ML training and data processing
│   ├── data/             # Raw datasets and processed data
│   ├── .env.example      # Environment template
│   └── system_analysis.ipynb  # System documentation
├── mp_env/               # Legacy virtual environment (can be ignored)
├── notesForMP/           # Project documentation and notes
├── test_osm.py          # OSM testing script
├── test_auth.py         # Authentication testing script
├── auth_test.html       # Authentication test page
└── README.md
```

## 🔑 Getting a Mapbox API Token

1. Go to [Mapbox.com](https://account.mapbox.com/) and create a free account
2. Navigate to the [Access Tokens](https://account.mapbox.com/access-tokens/) page
3. Copy your default public token
4. Add it to your `.env` file as `MAPBOX_ACCESS_TOKEN`

## 🎯 How to Use the Application

1. **Register/Login:** Create an account or login with existing credentials
2. **Guest Access:** Click "Continue as Guest" to explore without registration
3. **Pin Location:** Click anywhere on the Kathmandu map
4. **Select Café Type:** Choose from Coffee Shop, Bakery, Restaurant, etc.
5. **Get Analysis:** View competitor analysis, population data, and ML recommendations
6. **Explore Results:** See top 5 nearby cafés and suitability score

## 🛠 Development Workflow

### Running Tests
```bash
cd cafelocate/backend
python manage.py test
```

### Training ML Model
```bash
cd cafelocate/ml
python train_model.py
```

### Evaluating Model Performance
```bash
cd cafelocate/ml
python evaluate.py
```

### Data Collection Scripts
```bash
cd cafelocate/ml
python collect_data.py    # Collect café data via Mapbox
python download_census.py # Process census PDF data
python download_roads.py  # Download OSM road data
```

## 👥 Team

- **Santosh Mahato Koiri** - Backend API Development
- **Sijan Shrestha** - Frontend & UI/UX
- **Upendra Dhungana** - Machine Learning & Data Science

## 📋 Features

- ✅ **Real Data Integration:** Mapbox API, OpenStreetMap, Nepal Census 2021
- ✅ **Custom Authentication:** Username/email + password registration/login
- ✅ **Guest Access:** Use the app without creating an account
- ✅ **Interactive Map:** Leaflet.js powered Kathmandu map
- ✅ **ML Predictions:** Random Forest model for location suitability
- ✅ **Spatial Analysis:** Population density, competitor analysis, road access
- ✅ **Responsive Design:** Works on desktop and mobile devices

## 🔧 Tech Stack

- **Backend:** Django 4.2, Django REST Framework, SQLite
- **ML:** scikit-learn, pandas, numpy, PyPDF2
- **Frontend:** HTML5, CSS3, JavaScript, Leaflet.js
- **APIs:** Mapbox Geocoding API, OpenStreetMap
- **Authentication:** JWT (JSON Web Tokens)
- **Data Processing:** GeoPandas, GeoJSON

## 📊 Data Sources

- **Café Locations:** Mapbox Geocoding API (1,072+ locations)
- **Road Network:** OpenStreetMap data
- **Population Data:** Nepal Census 2021 (862,400 total population)
- **Ward Boundaries:** Kathmandu administrative divisions (32 wards)

## 🐛 Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt` in backend directory

2. **Database errors:**
   - Run `python manage.py migrate` to set up database
   - Check `.env` file has correct settings

3. **Map not loading:**
   - Check internet connection
   - Verify Mapbox token is correct in `.env`

4. **API errors:**
   - Ensure backend server is running on port 8000
   - Check CORS settings if accessing from different port

5. **Authentication issues:**
   - Clear browser localStorage
   - Check backend server logs for errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 for Python code
- Use meaningful commit messages
- Test your changes before submitting
- Update documentation for new features
- Ensure all dependencies are properly listed

## 🚀 Deployment

### Local Development
The setup scripts and instructions above are for local development.

### Production Deployment (Future)
- Use PostgreSQL instead of SQLite
- Set `DEBUG=False` in environment
- Use a production WSGI server (Gunicorn)
- Set up proper static file serving
- Configure HTTPS
- Set up monitoring and logging

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the `system_analysis.ipynb` notebook
3. Check the `notesForMP/` folder for additional documentation
4. Open an issue on GitHub with detailed error messages

## 🙏 Acknowledgments

- **Mapbox** for providing excellent mapping services
- **OpenStreetMap** contributors for road network data
- **Nepal Census 2021** for population demographics
- **Django** and **Leaflet.js** communities for amazing tools

---

**Developed by:** Santosh Mahato Koiri, Sijan Shrestha, Upendra Dhungana
**Institution:** [Your Institution Name]
**Date:** February 21, 2026