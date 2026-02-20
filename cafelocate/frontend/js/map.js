// CafeLocate - Map Module
// Handles Leaflet map initialization, markers, and user interactions

class MapManager {
    constructor() {
        this.map = null;
        this.marker = null;
        this.circle = null;
        this.cafeMarkers = [];
        this.selectedLocation = null;
        this.selectedCafeType = null;
        this.init();
    }

    init() {
        if (!document.getElementById('map')) return;

        this.initializeMap();
        this.setupEventListeners();
        this.loadKathmanduBounds();
    }

    initializeMap() {
        // Initialize Leaflet map centered on Kathmandu
        this.map = L.map('map').setView([27.7172, 85.3240], 12);

        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18,
        }).addTo(this.map);

        // Set bounds to Kathmandu Metropolitan City
        const kathmanduBounds = L.latLngBounds(
            [27.6, 85.2], // Southwest
            [27.8, 85.4]  // Northeast
        );
        this.map.setMaxBounds(kathmanduBounds);
        this.map.on('drag', () => {
            this.map.panInsideBounds(kathmanduBounds, { animate: false });
        });
    }

    setupEventListeners() {
        // Map click event
        this.map.on('click', (e) => {
            this.handleMapClick(e.latlng);
        });

        // Cafe type selector
        const cafeTypeSelect = document.getElementById('cafe-type-select');
        if (cafeTypeSelect) {
            cafeTypeSelect.addEventListener('change', (e) => {
                this.selectedCafeType = e.target.value;
                if (this.selectedLocation) {
                    this.analyzeLocation();
                }
            });
        }
    }

    handleMapClick(latlng) {
        const { lat, lng } = latlng;

        // Remove existing marker and circle
        this.clearMarkerAndCircle();

        // Add new marker
        this.marker = L.marker([lat, lng]).addTo(this.map);
        this.marker.bindPopup(`<b>Location Selected</b><br>Lat: ${lat.toFixed(6)}<br>Lng: ${lng.toFixed(6)}`).openPopup();

        // Add 500m radius circle
        this.circle = L.circle([lat, lng], {
            color: '#e74c3c',
            fillColor: '#e74c3c',
            fillOpacity: 0.1,
            radius: 500
        }).addTo(this.map);

        // Store selected location
        this.selectedLocation = { lat, lng };

        // Update coordinates display
        this.updateCoordinatesDisplay(lat, lng);

        // Analyze location if cafe type is selected
        if (this.selectedCafeType) {
            this.analyzeLocation();
        } else {
            this.showCafeTypePrompt();
        }
    }

    async analyzeLocation() {
        if (!this.selectedLocation || !this.selectedCafeType) return;

        const { lat, lng } = this.selectedLocation;

        // Show loading indicator
        this.showLoading(true);

        try {
            // Get nearby cafes
            const nearbyData = await window.apiManager.getNearbyCafes(lat, lng, 1000);
            this.displayNearbyCafes(nearbyData.cafes);

            // Get full analysis
            const analysisData = await window.apiManager.getSuitabilityAnalysis(lat, lng, this.selectedCafeType);
            this.displayAnalysisResults(analysisData);

        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(window.apiManager.formatError(error));
        } finally {
            this.showLoading(false);
        }
    }

    displayNearbyCafes(cafes) {
        // Clear existing cafe markers
        this.clearCafeMarkers();

        // Add markers for each cafe
        cafes.forEach(cafe => {
            const marker = L.marker([cafe.latitude, cafe.longitude])
                .addTo(this.map)
                .bindPopup(`
                    <b>${cafe.name}</b><br>
                    Type: ${cafe.cafe_type}<br>
                    Rating: ${cafe.rating || 'N/A'} ⭐<br>
                    Reviews: ${cafe.review_count || 0}
                `);

            this.cafeMarkers.push(marker);
        });
    }

    displayAnalysisResults(data) {
        // Update suitability score
        const scoreEl = document.getElementById('suitability-score');
        if (scoreEl) {
            scoreEl.textContent = data.suitability.score;
            this.updateScoreCircle(data.suitability.score);
        }

        // Update prediction
        const predictionCard = document.getElementById('prediction-card');
        if (predictionCard && data.prediction) {
            predictionCard.querySelector('.prediction-type').textContent = data.prediction.predicted_type;
            predictionCard.querySelector('.prediction-confidence').textContent =
                `Confidence: ${(data.prediction.confidence * 100).toFixed(1)}%`;
        }

        // Update top 5 cafes
        const top5List = document.getElementById('top5-list');
        if (top5List && data.top5) {
            top5List.innerHTML = data.top5.map(cafe => `
                <div class="cafe-item">
                    <div class="cafe-name">${cafe.name}</div>
                    <div class="cafe-rating">${cafe.rating || 'N/A'} ⭐</div>
                </div>
            `).join('');
        }

        // Update detailed metrics
        this.updateMetrics(data.suitability);
    }

    updateScoreCircle(score) {
        const scoreCircle = document.querySelector('.score-circle');
        if (scoreCircle) {
            const percentage = (score / 100) * 360;
            scoreCircle.style.background = `conic-gradient(var(--success-color) 0% ${score}%, #e0e0e0 ${score}% 100%)`;
        }
    }

    updateMetrics(suitability) {
        const updates = {
            'competitor-count': suitability.competitor_count,
            'road-length': `${suitability.road_length_m}m`,
            'population-density': `${suitability.population_density}/km²`
        };

        Object.entries(updates).forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        });
    }

    updateCoordinatesDisplay(lat, lng) {
        const coordsEl = document.getElementById('location-coords');
        if (coordsEl) {
            coordsEl.textContent = `${lat.toFixed(6)}, ${lng.toFixed(6)}`;
        }
    }

    showCafeTypePrompt() {
        // Highlight the cafe type selector
        const selectEl = document.getElementById('cafe-type-select');
        if (selectEl) {
            selectEl.style.borderColor = '#e74c3c';
            selectEl.style.boxShadow = '0 0 0 2px rgba(231, 76, 60, 0.2)';

            setTimeout(() => {
                selectEl.style.borderColor = '';
                selectEl.style.boxShadow = '';
            }, 3000);
        }
    }

    showLoading(show) {
        const loadingEl = document.getElementById('loading-indicator');
        if (loadingEl) {
            loadingEl.style.display = show ? 'block' : 'none';
        }
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast notification
        alert(`Error: ${message}`);
    }

    clearMarkerAndCircle() {
        if (this.marker) {
            this.map.removeLayer(this.marker);
            this.marker = null;
        }
        if (this.circle) {
            this.map.removeLayer(this.circle);
            this.circle = null;
        }
    }

    clearCafeMarkers() {
        this.cafeMarkers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.cafeMarkers = [];
    }

    clearMap() {
        this.clearMarkerAndCircle();
        this.clearCafeMarkers();
        this.selectedLocation = null;
        this.selectedCafeType = null;

        // Clear results display
        this.clearResultsDisplay();
    }

    clearResultsDisplay() {
        const elements = [
            'suitability-score',
            'prediction-card',
            'top5-list',
            'competitor-count',
            'road-length',
            'population-density',
            'location-coords'
        ];

        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (id === 'suitability-score' || id.includes('count') || id.includes('length') || id.includes('density')) {
                    el.textContent = '-';
                } else if (id === 'prediction-card') {
                    el.querySelector('.prediction-type').textContent = '-';
                    el.querySelector('.prediction-confidence').textContent = '-';
                } else if (id === 'top5-list') {
                    el.innerHTML = '<p class="no-data">Click on the map to see results</p>';
                } else if (id === 'location-coords') {
                    el.textContent = '';
                }
            }
        });
    }

    loadKathmanduBounds() {
        // This could load GeoJSON boundaries for more precise bounds
        // For now, using simple rectangular bounds set in initializeMap()
    }
}

// Initialize map manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.mapManager = new MapManager();
});
