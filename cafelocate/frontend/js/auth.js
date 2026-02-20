// CafeLocate - Authentication Module
// Handles Google OAuth and user session management

class AuthManager {
    constructor() {
        this.user = null;
        this.token = null;
        this.init();
    }

    init() {
        // Check for existing session
        const savedToken = localStorage.getItem('auth_token');
        const savedUser = localStorage.getItem('user_data');

        if (savedToken && savedUser) {
            this.token = savedToken;
            this.user = JSON.parse(savedUser);
            this.showMapPage();
        }

        this.setupGoogleSignIn();
        this.setupEventListeners();
    }

    setupGoogleSignIn() {
        // Initialize Google Sign-In (mock implementation for demo)
        const signInButton = document.getElementById('google-signin-button');

        if (window.google && window.google.accounts) {
            window.google.accounts.id.initialize({
                client_id: 'your-google-client-id.apps.googleusercontent.com',
                callback: this.handleGoogleSignIn.bind(this)
            });

            window.google.accounts.id.renderButton(signInButton, {
                theme: 'outline',
                size: 'large',
                text: 'signin_with'
            });
        } else {
            // Fallback for demo
            signInButton.innerHTML = '<button class="btn btn-primary" onclick="authManager.mockGoogleSignIn()">Sign in with Google</button>';
        }
    }

    async handleGoogleSignIn(response) {
        try {
            // In real implementation, send token to backend
            // const backendResponse = await fetch('/api/auth/google/', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify({ token: response.credential })
            // });
            // const data = await backendResponse.json();

            // Mock response for demo
            const mockUser = {
                id: 1,
                email: 'demo@example.com',
                name: 'Demo User',
                picture_url: 'https://via.placeholder.com/40'
            };

            this.user = mockUser;
            this.token = 'mock-jwt-token-' + Date.now();

            // Save to localStorage
            localStorage.setItem('auth_token', this.token);
            localStorage.setItem('user_data', JSON.stringify(this.user));

            this.showMapPage();
        } catch (error) {
            console.error('Authentication failed:', error);
            alert('Authentication failed. Please try again.');
        }
    }

    mockGoogleSignIn() {
        // Demo sign-in without actual Google OAuth
        const mockUser = {
            id: 1,
            email: 'guest@cafelocate.com',
            name: 'Guest User',
            picture_url: 'https://via.placeholder.com/40'
        };

        this.user = mockUser;
        this.token = 'demo-token-' + Date.now();

        localStorage.setItem('auth_token', this.token);
        localStorage.setItem('user_data', JSON.stringify(this.user));

        this.showMapPage();
    }

    setupEventListeners() {
        // Guest login button
        const guestBtn = document.getElementById('guest-login');
        if (guestBtn) {
            guestBtn.addEventListener('click', () => this.mockGoogleSignIn());
        }

        // Logout button
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.logout());
        }
    }

    showMapPage() {
        document.getElementById('login-page').classList.remove('active');
        document.getElementById('map-page').classList.add('active');

        // Update user info in header
        const userNameEl = document.getElementById('user-name');
        if (userNameEl && this.user) {
            userNameEl.textContent = this.user.name;
        }

        // Initialize map if not already done
        if (window.mapManager) {
            window.mapManager.init();
        }
    }

    logout() {
        this.user = null;
        this.token = null;

        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');

        document.getElementById('map-page').classList.remove('active');
        document.getElementById('login-page').classList.add('active');

        // Clear map if it exists
        if (window.mapManager) {
            window.mapManager.clearMap();
        }
    }

    isAuthenticated() {
        return this.token !== null;
    }

    getToken() {
        return this.token;
    }

    getUser() {
        return this.user;
    }
}

// Initialize auth manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.authManager = new AuthManager();
});
