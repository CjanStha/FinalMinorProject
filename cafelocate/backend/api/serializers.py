from rest_framework import serializers
from .models import Cafe, Ward, UserProfile


# ═══════════════════════════════════════════════════════════════════
# CafeSerializer
# Converts a Cafe database object → JSON for the frontend
# ═══════════════════════════════════════════════════════════════════
class CafeSerializer(serializers.ModelSerializer):

    # Extra field not in the model — calculated from rating × log(reviews)
    # SerializerMethodField = computed field, not directly from DB column
    score = serializers.SerializerMethodField()

    class Meta:
        model  = Cafe      # which model to serialize
        fields = [         # which fields to include in JSON output
            'id',
            'place_id',
            'name',
            'cafe_type',
            'latitude',
            'longitude',
            'rating',
            'review_count',
            'is_open',
            'score',        # our calculated field
        ]
        # NOTE: 'location' (PointField geometry) is intentionally excluded
        #       because it produces complex GeoJSON that the frontend doesn't need.
        #       We expose latitude/longitude instead (simpler for Leaflet.js).

    def get_score(self, obj):
        """Weighted score used to rank Top 5 cafés.
        Formula: rating × log(review_count + 1)
        Higher rating AND more reviews = higher score.
        The +1 avoids log(0) error for cafes with 0 reviews.
        """
        import math
        if obj.rating is None:
            return 0
        return round(obj.rating * math.log(obj.review_count + 1), 2)


# ═══════════════════════════════════════════════════════════════════
# SuitabilityRequestSerializer
# Validates the incoming POST data when user pins a location
# ═══════════════════════════════════════════════════════════════════
class SuitabilityRequestSerializer(serializers.Serializer):
    # Frontend sends: { "lat": 27.7172, "lng": 85.3240, "cafe_type": "bakery", "radius": 500 }

    lat       = serializers.FloatField(
        min_value=27.6, max_value=27.8,  # Kathmandu latitude range
        error_messages={'min_value': 'Location must be within Kathmandu.'}
    )
    lng       = serializers.FloatField(
        min_value=85.2, max_value=85.5,  # Kathmandu longitude range
    )
    cafe_type = serializers.ChoiceField(
        choices=['coffee_shop', 'bakery', 'dessert_shop', 'restaurant']
    )
    radius    = serializers.IntegerField(
        min_value=100, max_value=2000, default=500,  # 100m to 2km radius
        required=False
    )


# ═══════════════════════════════════════════════════════════════════
# UserProfileSerializer
# ═══════════════════════════════════════════════════════════════════
class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model  = UserProfile
        fields = ['id', 'email', 'name', 'picture_url', 'joined_at']
        read_only_fields = ['joined_at']  # can't be set by the client