from django.contrib import admin  # Temporarily changed from GIS admin due to GDAL issues
from .models import Cafe, Ward, Road, UserProfile


# Register Cafe with a customized display
@admin.register(Cafe)
class CafeAdmin(admin.ModelAdmin):  # Temporarily changed from GeoModelAdmin
    # Columns shown in the list view
    list_display  = ['name', 'cafe_type', 'rating', 'review_count', 'is_open']
    # Filter sidebar options
    list_filter   = ['cafe_type', 'is_open']
    # Search box — searches these fields
    search_fields = ['name', 'place_id']
    # How many per page
    list_per_page = 50


@admin.register(Ward)
class WardAdmin(admin.ModelAdmin):  # Temporarily changed from GeoModelAdmin
    list_display  = ['ward_number', 'population', 'area_sqkm', 'population_density']
    ordering      = ['ward_number']


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display  = ['name', 'email', 'joined_at']
    search_fields = ['name', 'email']
    readonly_fields = ['google_id', 'joined_at']

# Road is not registered — too many records to browse in admin