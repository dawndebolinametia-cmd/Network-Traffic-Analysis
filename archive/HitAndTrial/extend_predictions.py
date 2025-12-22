import pandas as pd
from datetime import datetime, timedelta

# Load the predictions CSV
df = pd.read_csv('reports/predictions.csv')

# Convert log_time to datetime
df['log_time'] = pd.to_datetime(df['log_time'])

# Sort by log_time
df = df.sort_values('log_time').reset_index(drop=True)

# Set new log_time starting from 2025-12-10 00:00:00 at hourly intervals
start = datetime(2025, 12, 10, 0, 0, 0)
df['log_time'] = [start + timedelta(hours=i) for i in range(len(df))]

# Save back to CSV
df.to_csv('reports/predictions.csv', index=False)

print(f"Extended predictions.csv with {len(df)} rows at hourly intervals starting from {start}")
