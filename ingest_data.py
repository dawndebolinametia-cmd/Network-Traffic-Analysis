import pandas as pd
import pymysql
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import logging
import os
from config import (
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    NETWORK_TRAFFIC_TABLE
)
from logging_utils import logger

def get_db_connection():
    """Establish database connection."""
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

def preprocess_data(df):
    """Preprocess the data: timestamps, missing values, encoding, scaling."""
    df = df.copy()

    # Handle timestamps - assuming there's a timestamp column, convert to datetime
    if 'TSesStart' in df.columns:
        df['log_time'] = pd.to_datetime(df['TSesStart'], unit='s', errors='coerce')
    else:
        df['log_time'] = pd.Timestamp.now()

    # Handle missing values - fill numeric with mean, categorical with mode
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')

    # Encode categorical features
    categorical_cols = ['srcIP', 'dstIP', 'srcMAC', 'dstMAC', 'label_2', 'label_poly', 'label_poly_o']
    encoders = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude encoded columns and timestamp
    numeric_cols = [col for col in numeric_cols if not col.endswith('_encoded') and col != 'log_time']

    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, encoders, scaler

def insert_data_to_mysql(df, table_name):
    """Insert preprocessed data into MySQL table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Prepare columns for insertion
    columns = ['log_time'] + [col for col in df.columns if col != 'log_time']
    placeholders = ', '.join(['%s'] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    # Convert to list of tuples for insertion
    data_tuples = []
    for _, row in df.iterrows():
        row_data = []
        for col in columns:
            val = row[col]
            if pd.isna(val):
                row_data.append(None)
            elif isinstance(val, pd.Timestamp):
                row_data.append(val.to_pydatetime())
            else:
                row_data.append(val)
        data_tuples.append(tuple(row_data))

    # Insert in batches
    batch_size = 1000
    total_inserted = 0
    for i in range(0, len(data_tuples), batch_size):
        batch = data_tuples[i:i+batch_size]
        try:
            cursor.executemany(sql, batch)
            conn.commit()
            total_inserted += len(batch)
            logger.info(f"Inserted {len(batch)} rows (total: {total_inserted})")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            conn.rollback()

    cursor.close()
    conn.close()
    return total_inserted

def main():
    """Main ingestion function."""
    csv_path = 'data/ASNM-NBPOv2.csv'

    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        return

    logger.info("Starting data ingestion...")

    # Read CSV with chunking and validate column count
    chunksize = 10000
    total_rows = 0
    first_chunk = None
    num_columns = None

    for chunk in pd.read_csv(csv_path, sep=';', chunksize=chunksize):
        if first_chunk is None:
            first_chunk = chunk
            num_columns = len(chunk.columns)
            logger.info(f"Detected {num_columns} columns in CSV")
        else:
            if len(chunk.columns) != num_columns:
                logger.error(f"Column count mismatch in chunk: expected {num_columns}, got {len(chunk.columns)}")
                return

        # Preprocess chunk
        processed_chunk, _, _ = preprocess_data(chunk)

        # Insert chunk
        inserted = insert_data_to_mysql(processed_chunk, NETWORK_TRAFFIC_TABLE)
        total_rows += inserted

    logger.info(f"Data ingestion completed. Total rows ingested: {total_rows}")

if __name__ == "__main__":
    import numpy as np
    main()
