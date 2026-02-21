from django.db import models
# from django.contrib.gis.db import models as gis_models  # GDAL not available yet


# ═══════════════════════════════════════════════════════════════════
# TABLE 1: Cafe
# Stores every café in Kathmandu collected from Google Places API
# ═══════════════════════════════════════════════════════════════════
class Cafe(models.Model):

    # Google's unique ID for this place (e.g. "ChIJN1t_tDeuEmsRUsoyG83frY4")
    place_id     = models.CharField(max_length=100, unique=True)

    # Name of the café
    name         = models.CharField(max_length=255)

    # Type: "coffee_shop", "bakery", "dessert_shop", "restaurant"
    cafe_type    = models.CharField(max_length=50)

    # Latitude and longitude stored as regular numbers (for reference)
    latitude     = models.FloatField()
    longitude    = models.FloatField()

    # PointField: stores the location as a PostGIS geometry object
    # srid=4326 means WGS84 coordinate system (standard GPS coordinates)
    # This field enables spatial queries like ST_DWithin()
    # Temporarily using JSONField until GDAL is properly configured
    location     = models.JSONField(null=True, blank=True)  # Will store {'type': 'Point', 'coordinates': [lng, lat]}

    # Rating 1.0–5.0 from Google
    rating       = models.FloatField(null=True, blank=True)

    # Number of reviews on Google Maps
    review_count = models.IntegerField(default=0)

    # Is it still open? Helps filter out closed businesses
    is_open      = models.BooleanField(default=True)

    # When we collected this data from Google
    collected_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cafes'   # actual SQL table name
        ordering = ['-rating']  # default order: highest rated first

    def __str__(self):
        return f"{self.name} ({self.cafe_type}) — ⭐{self.rating}"
        # shown in admin panel and shell: "Morning Brew (coffee_shop) — ⭐4.5"


# ═══════════════════════════════════════════════════════════════════
# TABLE 2: Ward
# Kathmandu's 32 administrative wards with population data
# from Nepal Census 2021
# ═══════════════════════════════════════════════════════════════════
class Ward(models.Model):

    ward_number       = models.IntegerField(unique=True)
    population        = models.IntegerField()
    households        = models.IntegerField()
    area_sqkm         = models.FloatField()

    # Population density = population / area (calculated field)
    population_density = models.FloatField()

    # The actual boundary polygon of this ward
    # MultiPolygonField handles wards with non-contiguous boundaries
    # Temporarily using JSONField until GDAL is properly configured
    boundary          = models.JSONField(null=True, blank=True)  # Will store GeoJSON MultiPolygon

    class Meta:
        db_table = 'wards'

    def __str__(self):
        return f"Ward {self.ward_number} — Pop: {self.population:,}"


# ═══════════════════════════════════════════════════════════════════
# TABLE 3: Road
# Road network of Kathmandu from OpenStreetMap
# Used to compute road accessibility score in buffer area
# ═══════════════════════════════════════════════════════════════════
class Road(models.Model):

    osm_id      = models.BigIntegerField(unique=True)
    # Type: "primary", "secondary", "residential", "footway", etc.
    road_type   = models.CharField(max_length=50, blank=True)
    # The road geometry as a line
    # Temporarily using JSONField until GDAL is properly configured
    geometry    = models.JSONField()  # Will store GeoJSON LineString or MultiLineString

    class Meta:
        db_table = 'roads'


from django.contrib.auth.models import AbstractUser
from django.db import models


# ═══════════════════════════════════════════════════════════════════
# TABLE 4: Amenity
# OpenStreetMap amenities: schools, hospitals, bus stops, etc.
# ═══════════════════════════════════════════════════════════════════
class Amenity(models.Model):

    osm_id        = models.BigIntegerField(unique=True)
    amenity_type  = models.CharField(max_length=100)  # "school", "hospital", "bus_station", etc.
    name          = models.CharField(max_length=255, null=True, blank=True)
    latitude      = models.FloatField()
    longitude     = models.FloatField()

    # GeoJSON point: {'type': 'Point', 'coordinates': [lng, lat]}
    location      = models.JSONField(null=True, blank=True)

    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'amenities'
        indexes = [
            models.Index(fields=['amenity_type']),
            models.Index(fields=['latitude', 'longitude']),
        ]

    def __str__(self):
        return f"{self.amenity_type}: {self.name or 'Unnamed'}"


# ═══════════════════════════════════════════════════════════════════
# TABLE 5: UserProfile
# Custom user model with username, email, password authentication
# ═══════════════════════════════════════════════════════════════════
class UserProfile(AbstractUser):

    # Email is already included in AbstractUser, but we make it required
    email = models.EmailField(unique=True)

    # Additional fields if needed
    joined_at = models.DateTimeField(auto_now_add=True)

    # Remove Google OAuth fields
    # google_id and picture_url are no longer needed

    class Meta:
        db_table = 'user_profiles'

    def __str__(self):
        return f"{self.username} ({self.email})"