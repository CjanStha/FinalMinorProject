import pandas as pd

df1 = pd.read_csv('cafelocate/data/osm_amenities_kathmandu.csv')
df2 = pd.read_csv('cafelocate/data/combined_amenities_clean.csv')

print('osm_amenities_kathmandu.csv shape:', df1.shape)
print('combined_amenities_clean.csv shape:', df2.shape)

if 'osm_id' in df1.columns and 'osm_id' in df2.columns:
    common_ids = set(df1['osm_id']).intersection(set(df2['osm_id']))
    print('Common osm_id count:', len(common_ids))
    print('Sample common ids:', list(common_ids)[:5] if common_ids else 'None')
else:
    print('osm_id not in both files')