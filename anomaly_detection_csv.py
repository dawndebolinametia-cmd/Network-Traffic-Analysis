import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pymysql
import logging
import os
from datetime import datetime
from config import (
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    PREDICTIONS_TABLE
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

def create_hourly_predictions_table():
    """Create hourly_predictions table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        log_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        prediction TINYINT(1),
        anomaly_score FLOAT,
        traffic_type VARCHAR(50),
        is_anomaly TINYINT(1) GENERATED ALWAYS AS (CASE WHEN prediction = 1 THEN 1 ELSE 0 END) STORED,
        session_id BIGINT,
        model_type VARCHAR(50),
        confidence_score FLOAT
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    cursor.execute(create_sql)
    conn.commit()
    cursor.close()
    conn.close()
    logger.info(f"Ensured table {PREDICTIONS_TABLE} exists")

def preprocess_chunk(chunk):
    """Preprocess a chunk of data."""
    chunk = chunk.copy()

    # Handle timestamps
    if 'TSesStart' in chunk.columns:
        chunk['log_time'] = pd.to_datetime(chunk['TSesStart'], unit='s', errors='coerce')
        chunk['hour'] = chunk['log_time'].dt.floor('H')  # Aggregate by hour
    else:
        chunk['log_time'] = pd.Timestamp.now()
        chunk['hour'] = chunk['log_time'].dt.floor('H')

    # Handle NaNs
    numeric_cols = chunk.select_dtypes(include=[np.number]).columns
    chunk[numeric_cols] = chunk[numeric_cols].fillna(chunk[numeric_cols].mean())

    categorical_cols = ['srcIP', 'dstIP', 'srcMAC', 'dstMAC', 'label_2', 'label_poly', 'label_poly_o']
    for col in categorical_cols:
        if col in chunk.columns:
            chunk[col] = chunk[col].fillna('unknown')

    return chunk

def aggregate_hourly(chunk):
    """Aggregate chunk data by hour."""
    # Select numeric columns for aggregation
    numeric_cols = ['BPerSecIn', 'BPerSecOut', 'PktPerSIn', 'PktPerSOut', 'CntResendPktsIn', 'CntResendPktsOut']

    # Ensure columns exist
    available_numeric = [col for col in numeric_cols if col in chunk.columns]

    if not available_numeric:
        # Fallback to any numeric columns
        available_numeric = chunk.select_dtypes(include=[np.number]).columns.tolist()
        available_numeric = [col for col in available_numeric if col not in ['id', 'TSesStart', 'TSesEnd', 'SessDuration']]

    agg_dict = {col: ['mean', 'sum', 'count'] for col in available_numeric}
    agg_dict['id'] = 'count'  # Session count

    hourly = chunk.groupby('hour').agg(agg_dict).reset_index()

    # Flatten column names
    hourly.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in hourly.columns]

    # Rename session count
    session_count_col = [col for col in hourly.columns if 'id_count' in col]
    if session_count_col:
        hourly = hourly.rename(columns={session_count_col[0]: 'session_count'})

    return hourly

def perform_anomaly_detection(hourly_data):
    """Perform anomaly detection on hourly aggregated data."""
    # Select features for anomaly detection
    feature_cols = [col for col in hourly_data.columns if col not in ['hour', 'log_time'] and hourly_data[col].dtype in ['int64', 'float64']]

    if not feature_cols:
        logger.error("No suitable numeric features found for anomaly detection")
        return hourly_data

    X = hourly_data[feature_cols].fillna(0)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)

    # Convert predictions: -1 (anomaly) to 1, 1 (normal) to 0
    hourly_data['prediction'] = (predictions == -1).astype(int)
    hourly_data['anomaly_score'] = -anomaly_scores  # Make higher scores more anomalous

    logger.info(f"Anomaly detection completed. Detected {sum(hourly_data['prediction'])} anomalies out of {len(hourly_data)} hours")

    return hourly_data

def insert_predictions(hourly_data):
    """Insert predictions into MySQL table."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Prepare data for insertion
    data_to_insert = []
    for _, row in hourly_data.iterrows():
        data_to_insert.append((
            row['hour'],
            int(row['prediction']),
            float(row['anomaly_score']),
            int(row.get('session_count', 0)),
            float(row.get('BPerSecIn_mean', 0)),
            float(row.get('BPerSecOut_sum', 0)),
            float(row.get('BPerSecIn_sum', 0))
        ))

    sql = f"""
    INSERT INTO {PREDICTIONS_TABLE}
    (log_time, prediction, anomaly_score, session_count, avg_packet_size, total_bytes_in, total_bytes_out)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    # Insert in batches
    batch_size = 1000
    for i in range(0, len(data_to_insert), batch_size):
        batch = data_to_insert[i:i+batch_size]
        try:
            cursor.executemany(sql, batch)
            conn.commit()
            logger.info(f"Inserted {len(batch)} hourly predictions")
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            conn.rollback()

    cursor.close()
    conn.close()

def main():
    """Main function."""
    csv_path = 'data/ASNM-NBPOv2.csv'

    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        return

    logger.info("Starting anomaly detection on CSV data")

    # Create table if needed
    create_hourly_predictions_table()

    # Process CSV in chunks
    chunksize = 10000
    all_hourly = []

    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, sep=';', chunksize=chunksize)):
        logger.info(f"Processing chunk {chunk_num + 1}")

        try:
            # Preprocess chunk
            processed_chunk = preprocess_chunk(chunk)

            # Aggregate by hour
            hourly_chunk = aggregate_hourly(processed_chunk)

            all_hourly.append(hourly_chunk)

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num + 1}: {e}")
            continue

    if not all_hourly:
        logger.error("No data processed")
        return

    # Combine all hourly data
    full_hourly = pd.concat(all_hourly, ignore_index=True)

    # Group again in case same hours across chunks
    final_agg = full_hourly.groupby('hour').agg({
        col: 'sum' if 'sum' in col else 'mean' for col in full_hourly.columns if col != 'hour'
    }).reset_index()

    logger.info(f"Total hours processed: {len(final_agg)}")

    # Perform anomaly detection
    final_agg = perform_anomaly_detection(final_agg)

    # Insert results
    insert_predictions(final_agg)

    logger.info("Anomaly detection and insertion completed")

if __name__ == "__main__":
    main()
