import pandas as pd

# Load the combined dataset
df = pd.read_csv('cafelocate/data/combined_all_datasets.csv', low_memory=False)

print('Combined CSV shape:', df.shape)
print('\nMissing values per column:')
missing = df.isnull().sum()
print(missing[missing > 0])

print('\nData types:')
print(df.dtypes.value_counts())

print('\nColumns with most missing values:')
print(missing.nlargest(10))

print('\nSource file distribution:')
print(df['source_file'].value_counts())

# Check for columns that are mostly empty
mostly_empty = missing[missing > len(df) * 0.9]
print(f'\nColumns that are >90% empty ({len(mostly_empty)} columns):')
print(mostly_empty)