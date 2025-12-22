import pandas as pd
import os

csv_path = 'data/ASNM-NBPOv2.csv'

if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path, sep=';')
    print(f"Row count: {len(df)}")
    print(f"Column names: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
except Exception as e:
    print(f"Error loading CSV: {e}")
