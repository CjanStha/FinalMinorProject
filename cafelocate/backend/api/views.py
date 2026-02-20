from django.conf import settings
from django.db.models import F, Value, FloatField
from django.db.models.functions import Sqrt, Power

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response

# Temporarily commented out Google OAuth imports
# from google.oauth2 import id_token
# from google.auth.transport import requests as google_requests
import jwt, math, logging, json

from .models import Cafe, Ward, Road, UserProfile
from .serializers import CafeSerializer, SuitabilityRequestSerializer, UserProfileSerializer
from ml_engine.predictor import get_prediction

logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r * 1000  # Return meters


def point_in_polygon(point_lng, point_lat, polygon_geojson):
    """
    Check if a point is inside a GeoJSON polygon using ray casting algorithm
    """
    if not polygon_geojson or polygon_geojson.get('type') != 'Polygon':
        return False

    # Get the exterior ring coordinates
    coordinates = polygon_geojson.get('coordinates', [])
    if not coordinates or len(coordinates) == 0:
        return False

    exterior_ring = coordinates[0]  # First array is the exterior ring

    # Ray casting algorithm
    n = len(exterior_ring)
    inside = False

    p1x, p1y = exterior_ring[0]
    for i in range(1, n + 1):
        p2x, p2y = exterior_ring[i % n]
        if point_lat > min(p1y, p2y):
            if point_lat <= max(p1y, p2y):
                if point_lng <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point_lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point_lng <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


# ═══════════════════════════════════════════════════════════════════
# VIEW 1: Google OAuth Login
# POST /api/auth/google/
# Frontend sends Google's ID token → we validate it → return our JWT
# ═══════════════════════════════════════════════════════════════════
class GoogleLoginView(APIView):
    authentication_classes = []  # no auth required to log in
    permission_classes = []     # open to anyone

    def post(self, request):
        # TEMPORARY: Mock login for testing
        return Response({
            'token': 'mock-jwt-token-for-testing',
            'user': {
                'id': 1,
                'email': 'test@example.com',
                'name': 'Test User',
                'picture_url': ''
            }
        })

        # Original Google OAuth code (commented out temporarily)
        # token = request.data.get('token')
        # if not token:
        #     return Response({'error': 'Token required'}, status=400)
        #
        # try:
        #     # Verify with Google — confirms the token is legitimate
        #     idinfo = id_token.verify_oauth2_token(
        #         token,
        #         google_requests.Request(),
        #         settings.GOOGLE_CLIENT_ID
        #     )
        #
        #     # Get or create user in our database
        #     user, created = UserProfile.objects.get_or_create(
        #         google_id=idinfo['sub'],
        #         defaults={
        #             'email':       idinfo['email'],
        #             'name':        idinfo.get('name', ''),
        #             'picture_url': idinfo.get('picture', ''),
        #         }
        #     )
        #
        #     # Create our own JWT so frontend doesn't need Google's token again
        #     our_token = jwt.encode(
        #         {'user_id': user.id, 'email': user.email},
        #         settings.SECRET_KEY,
        #         algorithm='HS256'
        #     )
        #     return Response({
        #         'token': our_token,
        #         'user':  UserProfileSerializer(user).data
        #     })
        #
        # except ValueError as e:
        #     return Response({'error': 'Invalid Google token'}, status=401)


# ═══════════════════════════════════════════════════════════════════
# VIEW 2: Nearby Cafés
# GET /api/cafes/nearby/?lat=27.71&lng=85.32&radius=500
# Returns all cafés within 'radius' meters of the given point
# ═══════════════════════════════════════════════════════════════════
class NearbyCafesView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))  # default 500m
        except (TypeError, ValueError):
            return Response({'error': 'lat and lng are required numbers'}, status=400)

        # Get all open cafes and calculate distances
        cafes = []
        for cafe in Cafe.objects.filter(is_open=True):
            if cafe.location and isinstance(cafe.location, dict):
                cafe_lng = cafe.location.get('coordinates', [None, None])[0]
                cafe_lat = cafe.location.get('coordinates', [None, None])[1]

                if cafe_lng is not None and cafe_lat is not None:
                    distance = haversine_distance(lat, lng, cafe_lat, cafe_lng)
                    if distance <= radius:
                        # Add distance to cafe object for sorting
                        cafe.distance = distance
                        cafes.append(cafe)

        # Sort by distance
        cafes.sort(key=lambda c: c.distance)

        serializer = CafeSerializer(cafes, many=True)
        return Response({
            'count':  len(cafes),
            'cafes':  serializer.data,
            'center': {'lat': lat, 'lng': lng}
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: Full Suitability Analysis
# POST /api/analyze/
# The main endpoint — takes a pinned location + cafe type
# and returns: nearby cafes, top 5, suitability score, prediction
# ═══════════════════════════════════════════════════════════════════
class SuitabilityAnalysisView(APIView):

    def post(self, request):
        # Step 1: Validate incoming data
        serializer = SuitabilityRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        lat       = serializer.validated_data['lat']
        lng       = serializer.validated_data['lng']
        cafe_type = serializer.validated_data['cafe_type']

        # Step 2: Get all cafes within 500m using haversine distance
        nearby_cafes = []
        for cafe in Cafe.objects.filter(is_open=True):
            if cafe.location and isinstance(cafe.location, dict):
                cafe_lng = cafe.location.get('coordinates', [None, None])[0]
                cafe_lat = cafe.location.get('coordinates', [None, None])[1]

                if cafe_lng is not None and cafe_lat is not None:
                    distance = haversine_distance(lat, lng, cafe_lat, cafe_lng)
                    if distance <= 500:
                        cafe.distance = distance
                        nearby_cafes.append(cafe)

        total_competitors = len(nearby_cafes)

        # Step 4: Get Top 5 cafés ranked by our score formula
        top5_qs = sorted(
            nearby_cafes,
            key=lambda c: (c.rating or 0) * math.log(c.review_count + 1),
            reverse=True
        )[:5]

        # Step 5: Get population density from the ward containing this point
        pop_density = 5000  # fallback
        for ward in Ward.objects.all():
            if ward.boundary and isinstance(ward.boundary, dict):
                if point_in_polygon(lng, lat, ward.boundary):
                    pop_density = ward.population_density
                    break

        # Step 6: Compute road length within 500m
        # For now, count road segments within radius (simplified approach)
        road_segments_nearby = 0
        for road in Road.objects.all():
            if road.geometry and isinstance(road.geometry, dict):
                geom_type = road.geometry.get('type')
                coordinates = road.geometry.get('coordinates', [])

                if geom_type == 'LineString' and coordinates:
                    # Check if any point on the line is within 500m
                    for coord in coordinates:
                        if len(coord) >= 2:
                            road_lng, road_lat = coord[0], coord[1]
                            distance = haversine_distance(lat, lng, road_lat, road_lng)
                            if distance <= 500:
                                road_segments_nearby += 1
                                break
                elif geom_type == 'MultiLineString' and coordinates:
                    # Check each LineString in the MultiLineString
                    for linestring in coordinates:
                        for coord in linestring:
                            if len(coord) >= 2:
                                road_lng, road_lat = coord[0], coord[1]
                                distance = haversine_distance(lat, lng, road_lat, road_lng)
                                if distance <= 500:
                                    road_segments_nearby += 1
                                    break
                        else:
                            continue
                        break

        # Estimate road length based on number of segments (rough approximation)
        road_m = road_segments_nearby * 100  # Assume 100m per segment on average

        # Step 7: Calculate suitability score (0–100)
        # w1=competition penalty, w2=road access bonus, w3=population bonus
        competitor_score = max(0, 1 - (total_competitors / 20)) * 40
        road_score        = min(1, road_m / 3000) * 30
        pop_score         = min(1, pop_density / 15000) * 30
        suitability_score = round(competitor_score + road_score + pop_score)

        # Step 8: Get ML prediction
        # Build the feature vector the model expects
        ratings = [c.rating for c in nearby_cafes if c.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        features = [
            total_competitors,
            avg_rating,
            road_m,
            pop_density,
        ]
        prediction = get_prediction(features)  # from ml_engine/predictor.py

        # Step 9: Build and return the full response
        return Response({
            'location':  {'lat': lat, 'lng': lng},
            'nearby_count': total_competitors,
            'top5':       CafeSerializer(top5_qs, many=True).data,
            'suitability': {
                'score':           suitability_score,
                'competitor_count': total_competitors,
                'road_length_m':   round(road_m),
                'population_density': pop_density,
            },
            'prediction': prediction,  # { type: "Bakery Café", confidence: 0.87 }
        })