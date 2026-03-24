import jwt
from django.test import TestCase
from django.urls import reverse
from django.conf import settings
from unittest.mock import patch

from .models import AnalysisHistory, Amenity, Cafe, UserProfile, Ward


class CafeApiTests(TestCase):
    def setUp(self):
        self.user = UserProfile.objects.create_user(
            username='historyuser',
            email='history@example.com',
            password='secret123'
        )
        self.auth_token = jwt.encode(
            {'user_id': self.user.id, 'username': self.user.username, 'email': self.user.email},
            settings.SECRET_KEY,
            algorithm='HS256'
        )
        Ward.objects.create(
            ward_number=1,
            population=1000,
            households=250,
            area_sqkm=1.0,
            population_density=1000.0,
            boundary={
                'type': 'Polygon',
                'coordinates': [[
                    [85.30, 27.70],
                    [85.35, 27.70],
                    [85.35, 27.75],
                    [85.30, 27.75],
                    [85.30, 27.70],
                ]]
            }
        )

    def test_cafe_stats_empty(self):
        response = self.client.get('/api/cafes/stats/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['total_cafes'], 0)
        self.assertEqual(data['open_cafes'], 0)
        self.assertIsNone(data['avg_rating'])
        self.assertEqual(data['avg_review_count'], 0)
        self.assertEqual(data['type_counts'], {})
        self.assertEqual(data['top_type_ranking'], [])

    def test_cafe_stats_aggregation(self):
        Cafe.objects.create(
            place_id='test-1',
            name='Cafe One',
            cafe_type='coffee_shop',
            latitude=27.70,
            longitude=85.30,
            location={'type': 'Point', 'coordinates': [85.30, 27.70]},
            rating=4.5,
            review_count=120,
            is_open=True
        )

        Cafe.objects.create(
            place_id='test-2',
            name='Cafe Two',
            cafe_type='bakery',
            latitude=27.71,
            longitude=85.31,
            location={'type': 'Point', 'coordinates': [85.31, 27.71]},
            rating=4.0,
            review_count=80,
            is_open=False
        )

        Cafe.objects.create(
            place_id='test-3',
            name='Cafe Three',
            cafe_type='coffee_shop',
            latitude=27.72,
            longitude=85.32,
            location={'type': 'Point', 'coordinates': [85.32, 27.72]},
            rating=None,
            review_count=0,
            is_open=True
        )

        response = self.client.get('/api/cafes/stats/')
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data['total_cafes'], 3)
        self.assertEqual(data['open_cafes'], 2)
        self.assertAlmostEqual(data['avg_rating'], 4.25)
        self.assertAlmostEqual(data['avg_review_count'], 66.7, places=1)
        self.assertEqual(data['type_counts']['coffee_shop'], 2)
        self.assertEqual(data['type_counts']['bakery'], 1)
        self.assertEqual(data['top_type_ranking'][0]['type'], 'coffee_shop')

    def test_nearby_cafes_by_distance(self):
        Cafe.objects.create(
            place_id='nearby-1',
            name='Nearby Cafe',
            cafe_type='coffee_shop',
            latitude=27.7172,
            longitude=85.3240,
            location={'type': 'Point', 'coordinates': [85.3240, 27.7172]},
            rating=4.4,
            review_count=50,
            is_open=True
        )

        Cafe.objects.create(
            place_id='far-1',
            name='Far Cafe',
            cafe_type='bakery',
            latitude=27.80,
            longitude=85.40,
            location={'type': 'Point', 'coordinates': [85.40, 27.80]},
            rating=3.5,
            review_count=10,
            is_open=True
        )

        response = self.client.get('/api/cafes/nearby/', {'lat': 27.7172, 'lng': 85.3240, 'radius': 1000})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['cafes'][0]['place_id'], 'nearby-1')

        # with big radius includes far cafe
        response2 = self.client.get('/api/cafes/nearby/', {'lat': 27.7172, 'lng': 85.3240, 'radius': 15000})
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        self.assertEqual(data2['count'], 2)

    def test_validate_location_inside_metropolitan_boundary(self):
        response = self.client.get('/api/validate-location/', {'lat': 27.7172, 'lng': 85.3240})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['is_valid'])

    def test_validate_location_outside_metropolitan_boundary(self):
        response = self.client.get('/api/validate-location/', {'lat': 27.80, 'lng': 85.50})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['is_valid'])

    def test_analyze_rejects_location_outside_metropolitan_boundary(self):
        response = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.80,
                'lng': 85.50,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn('lat', data)

    @patch('api.views.get_suitability_prediction')
    def test_analyze_uses_regression_score_in_response(self, mock_get_suitability_prediction):
        mock_get_suitability_prediction.return_value = {
            'predicted_score': 77.7,
            'predicted_suitability': 'High Suitability',
            'confidence': 0.91,
            'model_type': 'regression_ensemble_v3',
            'model_breakdown': {
                'random_forest_v3_score': 76.8,
                'xgboost_v3_score': 78.6,
            },
        }

        response = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['suitability']['score'], 77.7)
        self.assertEqual(data['suitability']['level'], 'High Suitability')
        self.assertEqual(data['prediction']['model_type'], 'regression_ensemble_v3')
        self.assertIn('recommended_cafe_type', data['prediction'])
        self.assertIn('recommended_cafe_type_confidence', data['prediction'])

    @patch('api.views.get_suitability_prediction')
    def test_analyze_returns_best_cafe_type_recommendation(self, mock_get_suitability_prediction):
        mock_get_suitability_prediction.return_value = {
            'predicted_score': 8.2,
            'predicted_suitability': 'High Suitability',
            'confidence': 0.88,
            'model_type': 'regression_ensemble_v3',
            'model_breakdown': {},
        }

        Cafe.objects.create(
            place_id='dessert-rival-1',
            name='Dessert Rival',
            cafe_type='dessert_shop',
            latitude=27.71725,
            longitude=85.32405,
            location={'type': 'Point', 'coordinates': [85.32405, 27.71725]},
            rating=4.0,
            review_count=30,
            is_open=True
        )

        response = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data['prediction']['recommended_cafe_type'], {'Coffee Shop', 'Bakery Cafe', 'Dessert Shop', 'Restaurant Cafe'})
        self.assertGreaterEqual(data['prediction']['recommended_cafe_type_confidence'], 0.0)
        self.assertIn('cafe_type_probabilities', data['prediction'])

    @patch('api.views.get_suitability_prediction')
    def test_analyze_builds_real_amenity_features(self, mock_get_suitability_prediction):
        captured = {}

        def capture_features(features_dict):
            captured.update(features_dict)
            return {
                'predicted_score': 55.0,
                'predicted_suitability': 'Medium Suitability',
                'confidence': 0.5,
                'model_type': 'regression_ensemble_v3',
                'model_breakdown': {},
            }

        mock_get_suitability_prediction.side_effect = capture_features

        Amenity.objects.create(
            osm_id=1,
            amenity_type='school',
            name='Nearby School',
            latitude=27.7174,
            longitude=85.3242,
            location={'type': 'Point', 'coordinates': [85.3242, 27.7174]},
        )
        Amenity.objects.create(
            osm_id=2,
            amenity_type='bus_stop',
            name='Nearby Stop',
            latitude=27.7173,
            longitude=85.3241,
            location={'type': 'Point', 'coordinates': [85.3241, 27.7173]},
        )

        response = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        self.assertGreaterEqual(captured['nearby_schools'], 1)
        self.assertGreaterEqual(captured['bus_stops_within_500m'], 1)
        self.assertGreater(captured['accessibility_score'], 0)
        self.assertGreater(captured['foot_traffic_score'], 0)

    @patch('api.views.get_suitability_prediction')
    def test_analyze_scales_regression_features_to_training_range(self, mock_get_suitability_prediction):
        captured = {}

        def capture_features(features_dict):
            captured.update(features_dict)
            return {
                'predicted_score': 5.5,
                'predicted_suitability': 'Medium Suitability',
                'confidence': 0.5,
                'model_type': 'regression_ensemble_v3',
                'model_breakdown': {},
            }

        mock_get_suitability_prediction.side_effect = capture_features

        Amenity.objects.create(
            osm_id=10,
            amenity_type='bus_stop',
            name='Nearby Stop',
            latitude=27.7173,
            longitude=85.3241,
            location={'type': 'Point', 'coordinates': [85.3241, 27.7173]},
        )
        Amenity.objects.create(
            osm_id=11,
            amenity_type='school',
            name='Nearby School',
            latitude=27.7174,
            longitude=85.3242,
            location={'type': 'Point', 'coordinates': [85.3242, 27.7174]},
        )

        response = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(captured['population_density'], 1.0)
        self.assertGreaterEqual(captured['competition_effective'], 0.0)
        self.assertLessEqual(captured['competition_effective'], 10.0)

        for feature_name, value in captured.items():
            self.assertGreaterEqual(value, 0.0, msg=feature_name)
            self.assertLessEqual(value, 10.0, msg=feature_name)

    @patch('api.views.get_suitability_prediction')
    def test_analyze_weights_same_type_competitors_more_heavily(self, mock_get_suitability_prediction):
        captured_calls = []

        def capture_features(features_dict):
            captured_calls.append(features_dict.copy())
            return {
                'predicted_score': 50.0,
                'predicted_suitability': 'Medium Suitability',
                'confidence': 0.5,
                'model_type': 'regression_ensemble_v3',
                'model_breakdown': {},
            }

        mock_get_suitability_prediction.side_effect = capture_features

        Cafe.objects.create(
            place_id='same-type-1',
            name='Coffee Rival',
            cafe_type='coffee_shop',
            latitude=27.71725,
            longitude=85.32405,
            location={'type': 'Point', 'coordinates': [85.32405, 27.71725]},
            rating=4.7,
            review_count=120,
            is_open=True
        )
        Cafe.objects.create(
            place_id='other-type-1',
            name='Bakery Rival',
            cafe_type='bakery',
            latitude=27.71728,
            longitude=85.32408,
            location={'type': 'Point', 'coordinates': [85.32408, 27.71728]},
            rating=4.0,
            review_count=60,
            is_open=True
        )
        Cafe.objects.create(
            place_id='same-type-2',
            name='Coffee Rival Two',
            cafe_type='coffee_shop',
            latitude=27.71732,
            longitude=85.32412,
            location={'type': 'Point', 'coordinates': [85.32412, 27.71732]},
            rating=4.5,
            review_count=80,
            is_open=True
        )

        response_coffee = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json'
        )
        self.assertEqual(response_coffee.status_code, 200)

        response_bakery = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'bakery',
                'radius': 500,
            },
            content_type='application/json'
        )
        self.assertEqual(response_bakery.status_code, 200)

        coffee_features = captured_calls[0]
        bakery_features = captured_calls[1]

        self.assertLess(coffee_features['competition_effective'], bakery_features['competition_effective'])

    @patch('api.views.get_suitability_prediction')
    def test_logged_in_analyze_does_not_persist_history_entries(self, mock_get_suitability_prediction):
        mock_get_suitability_prediction.return_value = {
            'predicted_score': 66.5,
            'predicted_suitability': 'Medium Suitability',
            'confidence': 0.8,
            'model_type': 'regression_ensemble_v3',
            'model_breakdown': {},
        }

        response = self.client.post(
            '/api/analyze/',
            data={
                'lat': 27.7172,
                'lng': 85.3240,
                'cafe_type': 'coffee_shop',
                'radius': 500,
            },
            content_type='application/json',
            HTTP_AUTHORIZATION=f'Bearer {self.auth_token}'
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(AnalysisHistory.objects.filter(user=self.user).count(), 0)

    def test_history_endpoint_requires_authentication(self):
        response = self.client.get('/api/history/')
        self.assertEqual(response.status_code, 401)

    def test_history_endpoint_returns_user_history_for_same_cafe_type(self):
        AnalysisHistory.objects.create(
            user=self.user,
            latitude=27.7172,
            longitude=85.3240,
            cafe_type='coffee_shop',
            radius=500,
            suitability_score=61.5,
            suitability_level='Medium Suitability'
        )
        AnalysisHistory.objects.create(
            user=self.user,
            latitude=27.7190,
            longitude=85.3250,
            cafe_type='bakery',
            radius=500,
            suitability_score=73.0,
            suitability_level='High Suitability'
        )

        response = self.client.get(
            '/api/history/',
            {'cafe_type': 'coffee_shop'},
            HTTP_AUTHORIZATION=f'Bearer {self.auth_token}'
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['history'][0]['cafe_type'], 'coffee_shop')
