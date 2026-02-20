// CafeLocate - API Module
// Handles all communication with the Django backend

class APIManager {
    constructor() {
        this.baseURL = 'http://localhost:8000/api';
        this.init();
    }

    init() {
        // Set up any initial API configurations
    }

    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Add auth token if available
        if (window.authManager && window.authManager.getToken()) {
            config.headers['Authorization'] = `Bearer ${window.authManager.getToken()}`;
        }

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Get nearby cafes within radius
    async getNearbyCafes(lat, lng, radius = 1000) {
        return this.makeRequest(`/cafes/nearby/?lat=${lat}&lng=${lng}&radius=${radius}`);
    }

    // Get full suitability analysis
    async getSuitabilityAnalysis(lat, lng, cafeType) {
        return this.makeRequest('/analyze/', {
            method: 'POST',
            body: JSON.stringify({
                lat: lat,
                lng: lng,
                cafe_type: cafeType
            })
        });
    }

    // Google OAuth authentication (mock)
    async authenticateWithGoogle(token) {
        return this.makeRequest('/auth/google/', {
            method: 'POST',
            body: JSON.stringify({ token: token })
        });
    }

    // Format API errors for user display
    formatError(error) {
        if (error.message.includes('Failed to fetch')) {
            return 'Unable to connect to the server. Please check your internet connection.';
        }

        if (error.message.includes('401')) {
            return 'Authentication required. Please log in again.';
        }

        if (error.message.includes('500')) {
            return 'Server error. Please try again later.';
        }

        return 'An unexpected error occurred. Please try again.';
    }
}

// Initialize API manager
window.apiManager = new APIManager();
