from django.db import models
# from django.contrib.gis.db import models  # Temporarily disabled due to GDAL installation issues on Windows


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
    # null=True because we might add data without geometry first
    # Temporarily changed to CharField due to GDAL issues
    location     = models.CharField(max_length=255, null=True, blank=True)

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
    # Temporarily changed to TextField due to GDAL issues
    boundary          = models.TextField(null=True, blank=True)

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
    # Temporarily changed to TextField due to GDAL issues
    geometry    = models.TextField()

    class Meta:
        db_table = 'roads'


# ═══════════════════════════════════════════════════════════════════
# TABLE 4: UserProfile
# Stores users who log in via Google OAuth
# ═══════════════════════════════════════════════════════════════════
class UserProfile(models.Model):

    # Google's unique ID for this user
    google_id    = models.CharField(max_length=100, unique=True)
    email        = models.EmailField(unique=True)
    name         = models.CharField(max_length=200)
    picture_url  = models.URLField(blank=True)
    joined_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'user_profiles'

    def __str__(self):
        return f"{self.name} ({self.email})"