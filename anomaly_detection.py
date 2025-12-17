
import pandas as pd
import numpy as np
import mysql.connector
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random
import joblib
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
    return pd.DataFrame()  # empty dataframe if no new data

# -------------------- Isolation Forest Anomaly Detection --------------------
def detect_anomalies_isolation_forest(data, contamination=0.1):
    if data.empty:
        return data

    numerical_features = ['response_code', 'bytes_sent']
    for col in numerical_features:
        if col not in data.columns:
            data[col] = 0  # fill missing columns if any

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

    numerical_features = ['packet_size']
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
        logger.info("No new data to insert.")
        return

    conn = get_db_connection()
    cursor = conn.cursor()
    for _, row in data.iterrows():
        cursor.execute(f"""
            INSERT INTO {PREDICTION_ANOMALY_TABLE}
            (log_time, source_ip, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s, %s)
        """, (row['log_time'], row['source_ip'], row['prediction'], row['anomaly_score'], row['traffic_type']))
    conn.commit()
    conn.close()
    logger.info(f"Inserted {len(data)} new anomaly predictions.")

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get last timestamp already processed in predictions
    cursor.execute(f"SELECT MAX(log_time) FROM {PREDICTION_ANOMALY_TABLE}")
    last_prediction_time = cursor.fetchone()[0]

    # If no previous predictions, start from 10 Dec 2025
    if last_prediction_time is None:
        last_prediction_time = datetime(2025, 12, 10, 15, 4, 55)

    cursor.close()
    conn.close()

    # Fetch new traffic data
    new_data = fetch_new_traffic_data(last_prediction_time)
    if new_data.empty:
        logger.info("No new network traffic data to process.")
    else:
        # Run anomaly detection
        new_data = detect_anomalies_isolation_forest(new_data)
        new_data = detect_anomalies_statistical(new_data)
        insert_predictions(new_data)
        logger.info("Anomaly detection completed for new data.")
