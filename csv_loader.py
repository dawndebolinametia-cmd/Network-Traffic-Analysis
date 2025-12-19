import pandas as pd
import os

# Ensure 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

# Check CSV exists
if not os.path.exists('data/ASNM-NBPOv2.csv'):
    raise FileNotFoundError("CSV file data/ASNM-NBPOv2.csv not found")

# Load with chunking
chunksize = 10000
total_rows = 0
first_chunk = None
num_columns = None

for chunk in pd.read_csv('data/ASNM-NBPOv2.csv', sep=';', chunksize=chunksize):
    if first_chunk is None:
        first_chunk = chunk
        num_columns = len(chunk.columns)
    else:
        if len(chunk.columns) != num_columns:
            raise ValueError("Column count mismatch in chunk")
    total_rows += len(chunk)

# Print first 5 rows
print("First 5 rows:")
print(first_chunk.head(5))

# Print total rows and columns
print(f"Total number of rows: {total_rows}")
print(f"Total number of columns: {num_columns}")
