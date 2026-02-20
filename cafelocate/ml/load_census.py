"""
Load census data into Ward models
"""

import os
import django
import pandas as pd

# Setup Django environment
import sys
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cafelocate.settings')
django.setup()

from api.models import Ward

def load_census_data():
    """
    Load census data from CSV and update Ward objects
    """
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'kathmandu_census.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Census CSV not found at {csv_path}")
        print("Run download_census.py first to get the data")
        return

    print("Loading census data...")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} wards in census data")

    updated_count = 0
    for _, row in df.iterrows():
        try:
            ward_number = int(row['ward_no'])

            # Find existing ward
            ward = Ward.objects.filter(ward_number=ward_number).first()
            if not ward:
                print(f"Warning: Ward {ward_number} not found in database")
                continue

            # Update with census data
            ward.population = int(row['population'])
            ward.households = int(row['households'])
            ward.area_sqkm = float(row['area_sqkm'])
            ward.population_density = float(row['population_density'])

            ward.save()
            updated_count += 1

        except Exception as e:
            print(f"Error updating ward {row.get('ward_no', 'unknown')}: {e}")
            continue

    print(f"Successfully updated {updated_count} wards with census data")

if __name__ == "__main__":
    load_census_data()