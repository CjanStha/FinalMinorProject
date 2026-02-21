from django.conf import settings
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
import jwt, math, logging, json

from .models import Cafe, Ward, Road, UserProfile, Amenity
from .serializers import CafeSerializer, SuitabilityRequestSerializer, UserProfileSerializer, AmenitySerializer
from ml_engine.predictor import get_suitability_prediction

try:
    from shapely.wkt import loads as wkt_loads
    from shapely.geometry import Point
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in meters between two points.
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371 * 1000  # metres


def point_in_polygon(point_lng, point_lat, polygon_geojson):
    """
    Ray-casting algorithm to check if a point is inside a GeoJSON polygon.
    """
    if not polygon_geojson:
        return False

    geom_type = polygon_geojson.get('type')
    coordinates = polygon_geojson.get('coordinates', [])

    if not coordinates:
        return False

    # Handle both Polygon and MultiPolygon
    rings = []
    if geom_type == 'Polygon':
        rings = [coordinates[0]] if coordinates else []
    elif geom_type == 'MultiPolygon':
        for polygon in coordinates:
            if polygon:
                rings.append(polygon[0])
    else:
        return False

    for exterior_ring in rings:
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
        if inside:
            return True

    return False


# ═══════════════════════════════════════════════════════════════════
# VIEW 1: User Registration
# POST /api/auth/register/
# ═══════════════════════════════════════════════════════════════════
class UserRegistrationView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        username = request.data.get('username', '').strip()
        email    = request.data.get('email', '').strip()
        password = request.data.get('password', '')

        if not username or not email or not password:
            return Response(
                {'error': 'Username, email, and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if len(password) < 6:
            return Response(
                {'error': 'Password must be at least 6 characters'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if UserProfile.objects.filter(username=username).exists():
            return Response(
                {'error': 'Username already taken'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if UserProfile.objects.filter(email=email).exists():
            return Response(
                {'error': 'An account with this email already exists'},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = UserProfile.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        token = jwt.encode(
            {'user_id': user.id, 'username': user.username, 'email': user.email},
            settings.SECRET_KEY,
            algorithm='HS256'
        )

        return Response({
            'token': token,
            'user': UserProfileSerializer(user).data
        }, status=status.HTTP_201_CREATED)


# ═══════════════════════════════════════════════════════════════════
# VIEW 2: User Login
# POST /api/auth/login/
# ═══════════════════════════════════════════════════════════════════
class UserLoginView(APIView):
    authentication_classes = []
    permission_classes = []

    def post(self, request):
        login_credential = request.data.get('username', '').strip()
        password = request.data.get('password', '')

        if not login_credential or not password:
            return Response(
                {'error': 'Username/email and password are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        user = None
        if '@' in login_credential:
            try:
                user = UserProfile.objects.get(email=login_credential)
            except UserProfile.DoesNotExist:
                pass
        else:
            try:
                user = UserProfile.objects.get(username=login_credential)
            except UserProfile.DoesNotExist:
                pass

        if user is None or not user.check_password(password):
            return Response(
                {'error': 'Invalid username/email or password'},
                status=status.HTTP_401_UNAUTHORIZED
            )

        if not user.is_active:
            return Response(
                {'error': 'Account is disabled'},
                status=status.HTTP_403_FORBIDDEN
            )

        token = jwt.encode(
            {'user_id': user.id, 'username': user.username, 'email': user.email},
            settings.SECRET_KEY,
            algorithm='HS256'
        )

        return Response({
            'token': token,
            'user': UserProfileSerializer(user).data
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 3: Nearby Cafés
# GET /api/cafes/nearby/?lat=27.71&lng=85.32&radius=500
# ═══════════════════════════════════════════════════════════════════
class NearbyCafesView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng query parameters are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        cafes = []
        for cafe in Cafe.objects.filter(is_open=True):
            if not (cafe.location and isinstance(cafe.location, dict)):
                # Fall back to latitude/longitude fields
                if cafe.latitude and cafe.longitude:
                    distance = haversine_distance(lat, lng, cafe.latitude, cafe.longitude)
                    if distance <= radius:
                        cafe.distance = distance
                        cafes.append(cafe)
            else:
                coords = cafe.location.get('coordinates', [None, None])
                cafe_lng, cafe_lat = coords[0], coords[1]
                if cafe_lng is not None and cafe_lat is not None:
                    distance = haversine_distance(lat, lng, cafe_lat, cafe_lng)
                    if distance <= radius:
                        cafe.distance = distance
                        cafes.append(cafe)

        cafes.sort(key=lambda c: getattr(c, 'distance', 0))

        serializer = CafeSerializer(cafes, many=True)
        return Response({
            'count':  len(cafes),
            'cafes':  serializer.data,
            'center': {'lat': lat, 'lng': lng}
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 4: Full Suitability Analysis
# POST /api/analyze/
# ═══════════════════════════════════════════════════════════════════
class SuitabilityAnalysisView(APIView):

    def post(self, request):
        serializer = SuitabilityRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        lat       = serializer.validated_data['lat']
        lng       = serializer.validated_data['lng']
        cafe_type = serializer.validated_data['cafe_type']
        radius    = serializer.validated_data.get('radius', 500)

        # Step 1: Find nearby cafes using haversine distance
        nearby_cafes = []
        for cafe in Cafe.objects.filter(is_open=True):
            cafe_lat, cafe_lng = None, None

            if cafe.location and isinstance(cafe.location, dict):
                coords = cafe.location.get('coordinates', [None, None])
                if len(coords) >= 2:
                    cafe_lng, cafe_lat = coords[0], coords[1]
            elif cafe.latitude and cafe.longitude:
                cafe_lat, cafe_lng = cafe.latitude, cafe.longitude

            if cafe_lat is not None and cafe_lng is not None:
                distance = haversine_distance(lat, lng, cafe_lat, cafe_lng)
                if distance <= radius:
                    cafe.distance = distance
                    nearby_cafes.append(cafe)

        total_competitors = len(nearby_cafes)

        # Step 2: Top 5 cafes by score
        top5_qs = sorted(
            nearby_cafes,
            key=lambda c: (c.rating or 0) * math.log(max(c.review_count, 1) + 1),
            reverse=True
        )[:5]

        # Step 3: Population density from ward
        pop_density = 10000  # fallback
        for ward in Ward.objects.all():
            if ward.boundary and isinstance(ward.boundary, dict):
                if point_in_polygon(lng, lat, ward.boundary):
                    pop_density = ward.population_density
                    break

        # Step 4: Road length estimate within radius
        road_segments_nearby = 0
        for road in Road.objects.all():
            if not (road.geometry and isinstance(road.geometry, dict)):
                continue

            geom_type = road.geometry.get('type')
            coordinates = road.geometry.get('coordinates', [])

            if geom_type == 'LineString' and coordinates:
                for coord in coordinates:
                    if len(coord) >= 2:
                        d = haversine_distance(lat, lng, coord[1], coord[0])
                        if d <= radius:
                            road_segments_nearby += 1
                            break

            elif geom_type == 'MultiLineString' and coordinates:
                found = False
                for linestring in coordinates:
                    if found:
                        break
                    for coord in linestring:
                        if len(coord) >= 2:
                            d = haversine_distance(lat, lng, coord[1], coord[0])
                            if d <= radius:
                                road_segments_nearby += 1
                                found = True
                                break

        road_m = road_segments_nearby * 100  # ~100m per segment estimate

        # Step 5: Compute suitability score (0-100)
        competitor_score = max(0, 1 - (total_competitors / 20)) * 40
        road_score       = min(1, road_m / 3000) * 30
        pop_score        = min(1, pop_density / 15000) * 30
        suitability_score = round(competitor_score + road_score + pop_score)

        # Step 6: ML prediction
        ratings = [c.rating for c in nearby_cafes if c.rating is not None]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0

        features = [
            total_competitors,
            avg_rating,
            road_m,
            pop_density,
        ]

        prediction = get_suitability_prediction(features)

        return Response({
            'location':     {'lat': lat, 'lng': lng},
            'nearby_count': total_competitors,
            'top5':         CafeSerializer(top5_qs, many=True).data,
            'suitability': {
                'score':              suitability_score,
                'level':              prediction.get('predicted_suitability', 'Unknown'),
                'confidence':         prediction.get('confidence', 0),
                'competitor_count':   total_competitors,
                'road_length_m':      round(road_m),
                'population_density': pop_density,
            },
            'prediction': prediction,
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 5: Amenities by Type and Radius
# GET /api/amenities/?lat=27.71&lng=85.32&radius=500&type=school
# Returns all amenities of a specific type within a radius
# ═══════════════════════════════════════════════════════════════════
class AmenitiesView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))
            amenity_type = request.GET.get('type', '').strip()
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat, lng (required) and type, radius (optional) must be valid'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Query amenities by type
        query = Amenity.objects.all()
        if amenity_type:
            query = query.filter(amenity_type__icontains=amenity_type)

        amenities = []
        for amenity in query:
            distance = haversine_distance(lat, lng, amenity.latitude, amenity.longitude)
            if distance <= radius:
                amenity.distance = distance
                amenities.append(amenity)

        # Sort by distance
        amenities.sort(key=lambda a: getattr(a, 'distance', 0))

        serializer = AmenitySerializer(amenities, many=True)
        return Response({
            'count': len(amenities),
            'amenities': serializer.data,
            'center': {'lat': lat, 'lng': lng},
            'amenity_type': amenity_type if amenity_type else 'All',
            'radius': radius
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 6: Multiple Amenity Types Report
# POST /api/amenities-report/
# Returns count of different amenity types within a radius
# ═══════════════════════════════════════════════════════════════════
class AmenitiesReportView(APIView):

    def post(self, request):
        try:
            lat    = float(request.data.get('lat'))
            lng    = float(request.data.get('lng'))
            radius = float(request.data.get('radius', 500))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Key amenity types we want to report
        key_amenity_types = ['school', 'hospital', 'bus_station', 'cafe', 'health_post', 'pharmacy']

        report = {}
        for amenity_type in key_amenity_types:
            amenities = []
            for amenity in Amenity.objects.filter(amenity_type__icontains=amenity_type):
                distance = haversine_distance(lat, lng, amenity.latitude, amenity.longitude)
                if distance <= radius:
                    amenity.distance = distance
                    amenities.append(amenity)

            report[amenity_type] = {
                'count': len(amenities),
                'amenities': AmenitySerializer(amenities[:20], many=True).data  # Top 20
            }

        return Response({
            'location': {'lat': lat, 'lng': lng},
            'radius': radius,
            'amenities_report': report
        })


# ═══════════════════════════════════════════════════════════════════
# VIEW 7: Area Population Calculation
# GET /api/area-population/?lat=27.71&lng=85.32&radius=500
# Calculates exact population within the area based on ward boundaries
# ═══════════════════════════════════════════════════════════════════
class AreaPopulationView(APIView):

    def get(self, request):
        try:
            lat    = float(request.GET.get('lat'))
            lng    = float(request.GET.get('lng'))
            radius = float(request.GET.get('radius', 500))
        except (TypeError, ValueError):
            return Response(
                {'error': 'lat and lng query parameters are required numbers'},
                status=status.HTTP_400_BAD_REQUEST
            )

        total_population = 0
        affected_wards = []

        # Check which wards the location falls in or intersects with
        for ward in Ward.objects.all():
            if not (ward.boundary and isinstance(ward.boundary, dict)):
                continue

            is_affected = False

            # Try to use Shapely if available for more accurate geometry handling
            if SHAPELY_AVAILABLE and ward.boundary.get('type') == 'wkt':
                try:
                    wkt_str = ward.boundary.get('wkt')
                    if wkt_str:
                        ward_polygon = wkt_loads(wkt_str)
                        test_point = Point(lng, lat)

                        # Check if point is within ward or within radius of ward
                        if ward_polygon.contains(test_point):
                            is_affected = True
                        elif ward_polygon.distance(test_point) * 111000 <= radius:  # ~111km per degree
                            is_affected = True
                except Exception as e:
                    logger.warning(f'Error parsing WKT for ward {ward.ward_number}: {e}')
                    continue
            else:
                # Fallback to checking if point is within or near the ward's geometric bounds
                # This is a simplified approach - just consider wards near the location
                try:
                    # Extract first set of coordinates if possible for boundary check
                    coords = ward.boundary.get('coordinates', [])
                    if coords and len(coords) > 0:
                        # Simple check: if we have coordinates, assume some overlap within radius
                        for coord_set in coords:
                            if isinstance(coord_set, (list, tuple)) and len(coord_set) > 0:
                                for coord in coord_set:
                                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                                        d = haversine_distance(lat, lng, coord[1], coord[0])
                                        if d <= radius * 2:  # generous bounds
                                            is_affected = True
                                            break
                                if is_affected:
                                    break
                except:
                    continue

            if is_affected:
                total_population += ward.population
                affected_wards.append({
                    'ward_number': ward.ward_number,
                    'population': ward.population,
                    'population_density': ward.population_density,
                    'area_sqkm': ward.area_sqkm,
                })

        return Response({
            'location': {'lat': lat, 'lng': lng},
            'radius': radius,
            'total_population': total_population,
            'affected_wards': affected_wards,
            'affected_ward_count': len(affected_wards),
        })

