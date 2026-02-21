import csv
import os
from django.core.management.base import BaseCommand
from django.db import transaction
from api.models import Amenity


class Command(BaseCommand):
    help = 'Load amenities from osm_amenities_kathmandu.csv'

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv',
            type=str,
            default='cafelocate/data/osm_amenities_kathmandu.csv',
            help='Path to the CSV file'
        )

    @transaction.atomic
    def handle(self, *args, **options):
        csv_path = options['csv']

        # Check if file exists
        if not os.path.exists(csv_path):
            self.stdout.write(self.style.ERROR(f'File not found: {csv_path}'))
            return

        loaded_count = 0
        skipped_count = 0
        updated_count = 0

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    osm_id = int(row['osm_id'])
                    amenity_type = row['amenity_type'].strip()
                    name = row.get('name', '').strip() or None
                    latitude = float(row['latitude'])
                    longitude = float(row['longitude'])

                    # Check if already exists
                    amenity, created = Amenity.objects.update_or_create(
                        osm_id=osm_id,
                        defaults={
                            'amenity_type': amenity_type,
                            'name': name,
                            'latitude': latitude,
                            'longitude': longitude,
                            'location': {
                                'type': 'Point',
                                'coordinates': [longitude, latitude]
                            }
                        }
                    )

                    if created:
                        loaded_count += 1
                    else:
                        updated_count += 1

                except (ValueError, KeyError) as e:
                    skipped_count += 1
                    continue

        self.stdout.write(self.style.SUCCESS(
            f'✅ Amenities loaded: {loaded_count}, Updated: {updated_count}, Skipped: {skipped_count}'
        ))
