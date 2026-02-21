import pandas as pd
import os

# Directory containing the CSV files
data_dir = 'd:/vscode/MP2/cafelocate/data'

# List of CSV files to process
csv_files = [
    'kathmandu_cafes.csv',
    'kathmandu_census.csv',
    'kathmandu_wards_boundary_sorted.csv',
    'combined_amenities_clean.csv',
    'osm_amenities_kathmandu.csv',
    'osm_roads_kathmandu.csv',
    'cafe_location_training_dataset.csv'
]

# Function to read CSV and remove duplicates
def read_and_dedup(file_path):
    df = pd.read_csv(file_path)
    original_count = len(df)
    df_dedup = df.drop_duplicates()
    dedup_count = len(df_dedup)
    print(f"{os.path.basename(file_path)}: {original_count} -> {dedup_count} rows after deduplication")
    return df_dedup

# Read all CSV files and remove duplicates
dfs = []
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    if os.path.exists(file_path):
        df = read_and_dedup(file_path)
        df['source_file'] = os.path.basename(csv_file)
        dfs.append(df)
    else:
        print(f"File not found: {file_path}")

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True, sort=False)

# Save the combined CSV
combined_file_path = os.path.join(data_dir, 'combined_all_datasets.csv')
combined_df.to_csv(combined_file_path, index=False)

print(f"Combined CSV created at: {combined_file_path}")
print(f"Total rows in combined CSV: {len(combined_df)}")
print(f"Columns in combined CSV: {list(combined_df.columns)}")