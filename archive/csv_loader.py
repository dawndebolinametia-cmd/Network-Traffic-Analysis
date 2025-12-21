import pandas as pd
import os
from datetime import datetime

def load_and_preprocess_csv(csv_path):
    """
    Load and preprocess the CSV file with semicolon separator.
    Standardize column names, parse timestamps, handle missing values,
    and compute derived features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} not found")

    # Load CSV with semicolon separator
    df = pd.read_csv(csv_path, sep=';')

    # Standardize column names (remove spaces, special chars, make lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9]', '_', regex=True)

    # Parse timestamps (assuming TSesStart and TSesEnd are present)
    if 'tsesstart' in df.columns:
        df['tsesstart'] = pd.to_datetime(df['tsesstart'], unit='s', errors='coerce')
    if 'tsesend' in df.columns:
        df['tsesend'] = pd.to_datetime(df['tsesend'], unit='s', errors='coerce')

    # Handle missing values (fill with 0 for numeric, 'unknown' for categorical)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('unknown')

    # Compute derived features
    if 'tsesstart' in df.columns and 'tsesend' in df.columns:
        df['session_duration'] = (df['tsesend'] - df['tsesstart']).dt.total_seconds()
    if 'cntpktsfrom' in df.columns and 'cntpktssto' in df.columns and 'session_duration' in df.columns:
        df['packet_rate'] = (df['cntpktsfrom'] + df['cntpktssto']) / df['session_duration'].replace(0, 1)
    if 'bytes_tcp_overhead' in df.columns and 'session_duration' in df.columns:
        df['bytes_per_session'] = df['bytes_tcp_overhead'] / df['session_duration'].replace(0, 1)

    # Add an id column if not present
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    return df
