import pandas as pd
import numpy as np
import mysql.connector
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random
from config import (
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    NETWORK_TRAFFIC_TABLE, PREDICTION_ANOMALY_TABLE, RANDOM_STATE
)
from logging_utils import logger

# -------------------- MySQL Connection --------------------
def get_db_connection():
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    return connection

# -------------------- Get Last Timestamp --------------------
def get_last_timestamp():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(log_time) FROM {NETWORK_TRAFFIC_TABLE}")
    last_time = cursor.fetchone()[0]
    conn.close()
    return last_time

# -------------------- Generate Synthetic Data --------------------
def generate_synthetic_data(last_time, num_rows=50):
    data = []
    current_time = last_time + timedelta(seconds=1)  # Start from last + 1 second
    for i in range(num_rows):
        log_time = current_time + timedelta(seconds=random.randint(1, 60))  # Increment by 1-60 seconds
        source_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
        request_method = random.choice(['GET', 'POST', 'PUT', 'DELETE'])
        response_code = random.choice([200, 201, 301, 400, 404, 500])
        bytes_sent = random.randint(100, 10000)
        data.append({
            'log_time': log_time,
            'source_ip': source_ip,
            'request_method': request_method,
            'response_code': response_code,
            'bytes_sent': bytes_sent
        })
        current_time = log_time
    return pd.DataFrame(data)

# -------------------- Insert Synthetic Data --------------------
def insert_synthetic_data(data):
    if data.empty:
        logger.info("No synthetic data to insert.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    for _, row in data.iterrows():
        cursor.execute(f"""
            INSERT INTO {NETWORK_TRAFFIC_TABLE}
            (log_time, source_ip, request_method, response_code, bytes_sent)
            VALUES (%s, %s, %s, %s, %s)
        """, (row['log_time'], row['source_ip'], row['request_method'], row['response_code'], row['bytes_sent']))
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(data)} synthetic network traffic rows.")

# -------------------- Fetch New Traffic Data --------------------
def fetch_new_traffic_data(last_time):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = f"""
        SELECT * FROM {NETWORK_TRAFFIC_TABLE}
        WHERE log_time > %s
        ORDER BY log_time ASC
    """
    cursor.execute(query, (last_time,))
    rows = cursor.fetchall()
    conn.close()
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()

# -------------------- Isolation Forest Anomaly Detection --------------------
def detect_anomalies_isolation_forest(data, contamination=0.1):
    if data.empty:
        return data

    numerical_features = ['bytes_sent']
    X = data[numerical_features].fillna(data[numerical_features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=RANDOM_STATE)
    model.fit(X_scaled)

    data = data.copy()
    data['anomaly_score'] = model.decision_function(X_scaled)
    preds = model.predict(X_scaled)
    data['prediction'] = (preds == -1).astype(int)
    data['traffic_type'] = data['prediction'].map({0: 'normal', 1: 'anomaly'})

    return data

# -------------------- Statistical Anomaly Detection --------------------
def detect_anomalies_statistical(data, threshold=3):
    if data.empty:
        return data

    numerical_features = ['bytes_sent']
    data = data.copy()

    for feature in numerical_features:
        if feature not in data.columns:
            continue
        mean = data[feature].mean()
        std = data[feature].std()
        z_scores = np.abs((data[feature] - mean) / std)
        data[f'{feature}_anomaly'] = z_scores > threshold

    anomaly_cols = [f'{f}_anomaly' for f in numerical_features if f'{f}_anomaly' in data.columns]
    if anomaly_cols:
        data['statistical_anomaly'] = data[anomaly_cols].any(axis=1)
        data['prediction'] = np.where(data['statistical_anomaly'], 1, data['prediction'])
        data['traffic_type'] = np.where(data['statistical_anomaly'], 'anomaly', data['traffic_type'])

    return data

# -------------------- Insert Predictions --------------------
def insert_predictions(data):
    if data.empty:
        logger.info("No new predictions to insert.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    for _, row in data.iterrows():
        cursor.execute(f"""
            INSERT INTO {PREDICTION_ANOMALY_TABLE}
            (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s)
        """, (row['log_time'], row['prediction'], row['anomaly_score'], row['traffic_type']))
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(data)} new anomaly predictions.")

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    try:
        # Get last timestamp
        last_time = get_last_timestamp()
        if last_time is None:
            last_time = datetime.now() - timedelta(hours=1)  # If no data, start from 1 hour ago

        logger.info(f"Last timestamp: {last_time}")

        # Generate synthetic data
        synthetic_data = generate_synthetic_data(last_time, num_rows=50)
        insert_synthetic_data(synthetic_data)

        # Fetch new data for anomaly detection
        new_data = fetch_new_traffic_data(last_time)

        if new_data.empty:
            logger.info("No new network traffic data to process.")
        else:
            # Run anomaly detection
            new_data = detect_anomalies_isolation_forest(new_data)
            new_data = detect_anomalies_statistical(new_data)
            insert_predictions(new_data)
            logger.info("Anomaly detection completed for new data.")

    except Exception as e:
        logger.error(f"Error in hourly pipeline: {e}")
