import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def label_anomalies_rule_based(data):
    """
    Labels anomalies based on rule-based criteria for supervised learning.

    Input features from network_traffic.csv: source_ip, dest_ip, timestamp, packet_size
    Target label: anomaly_flag (0 for normal, 1 for anomaly)

    Anomaly criteria (explainable and threshold-based):
    - packet_size > 4500: Indicates large packet, potential data exfiltration or unusual traffic volume.
    - packet_size < 50: Indicates small packet, potential probe or scanning activity.
    - diff_seconds < 0.1 and > 0: High frequency packets from same source_ip, indicating potential DDoS or rapid scanning.

    These rules are based on deviation from baseline behavior in network traffic.
    """
    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(['source_ip', 'timestamp'])
    data['diff_seconds'] = data.groupby('source_ip')['timestamp'].diff().dt.total_seconds().fillna(0)

    # Rule-based anomaly detection
    anomaly_conditions = (
        (data['packet_size'] > 4500) |  # Large packet size threshold (stricter)
        (data['packet_size'] < 50) |    # Small packet size threshold (stricter)
        ((data['diff_seconds'] < 0.1) & (data['diff_seconds'] > 0))  # High frequency threshold (stricter, exclude 0)
    )
    data['anomaly_flag'] = anomaly_conditions.astype(int)
    data['anomaly_score'] = np.where(anomaly_conditions, -1, 1)  # Score for compatibility
    data['traffic_type'] = data['anomaly_flag'].map({0: 'normal', 1: 'anomaly'})
    return data

# Load network traffic data
try:
    data = pd.read_csv('network_traffic.csv')
    print(f"Loaded {len(data)} rows from network_traffic.csv")
except FileNotFoundError:
    print("network_traffic.csv not found. Generating fallback synthetic data...")
    # Fallback to generate synthetic data if file not found
    rows = 100
    start_time = datetime.now() - timedelta(hours=rows)
    data = pd.DataFrame({
        "source_ip": [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(rows)],
        "dest_ip": [f"10.0.0.{np.random.randint(1,255)}" for _ in range(rows)],
        "timestamp": [start_time + timedelta(minutes=5*i) for i in range(rows)],
        "packet_size": np.random.randint(40, 5000, size=rows)
    })

# Apply rule-based anomaly labeling
data_with_labels = label_anomalies_rule_based(data)

if data_with_labels is not None:
    # Prepare predictions.csv (for supervised learning target)
    predictions_df = data_with_labels[['timestamp', 'anomaly_flag', 'anomaly_score', 'traffic_type']].copy()
    predictions_df = predictions_df.rename(columns={'anomaly_flag': 'prediction'})

    # Prepare prediction_anomaly.csv
    prediction_anomaly_df = data_with_labels[['source_ip', 'timestamp', 'anomaly_flag', 'diff_seconds']].copy()

    prediction_anomaly_df.to_csv("prediction_anomaly.csv", index=False)
    print(f"Prediction anomaly CSV generated successfully with {len(prediction_anomaly_df)} rows")
    print(f"Anomaly flag distribution: {prediction_anomaly_df['anomaly_flag'].value_counts().to_dict()}")

    predictions_df.to_csv("predictions.csv", index=False)
    print(f"Predictions CSV generated successfully with {len(predictions_df)} rows")
    print(f"Prediction distribution: {predictions_df['prediction'].value_counts().to_dict()}")
else:
    print("Failed to generate predictions")
