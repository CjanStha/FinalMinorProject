"""
Load road network data into Django/SQLite database
"""

import os
import sys
import django
import json

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
django.setup()

from api.models import Road

def load_road_network():
    """
    Load road network data from GeoJSON into the database
    """
    geojson_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_roads.geojson')

    if not os.path.exists(geojson_path):
        print(f"Error: Road GeoJSON not found at {geojson_path}")
        print("Run download_roads.py first to get the data")
        return

    print("Loading road network data...")

    # Load GeoJSON
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson = json.load(f)

    print(f"Found {len(geojson['features'])} road features in GeoJSON")

    # Clear existing data
    Road.objects.all().delete()
    print("Cleared existing road data")

    loaded_count = 0
    for feature in geojson['features']:
        try:
            properties = feature['properties']
            geometry = feature['geometry']

            osm_id = properties['osm_id']
            road_type = properties['highway']
            name = properties.get('name', '')

            # Store geometry as JSON for SQLite compatibility
            # (instead of using GeoDjango's spatial fields)
            geometry_json = json.dumps(geometry)

            # Create Road object
            road = Road(
                osm_id=osm_id,
                road_type=road_type,
                geometry=geometry_json  # Store as JSON string
            )
            road.save()
            loaded_count += 1

        except Exception as e:
            print(f"Error loading road {properties.get('osm_id', 'unknown')}: {e}")
            continue

    print(f"Successfully loaded {loaded_count} road segments")

if __name__ == "__main__":
    load_road_network()