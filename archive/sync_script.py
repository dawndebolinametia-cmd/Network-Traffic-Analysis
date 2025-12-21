import pandas as pd
import pymysql
import datetime
import os
from config import *

def load_csvs():
    nt_df = pd.read_csv(NETWORK_TRAFFIC_CSV)
    nt_df['timestamp'] = pd.to_datetime(nt_df['timestamp'])
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    pred_df['log_time'] = pd.to_datetime(pred_df['log_time'])
    return nt_df, pred_df

def generate_predictions_from_traffic(nt_df):
    # Aggregate network_traffic by hour
    nt_hourly = nt_df.set_index('timestamp').resample('h').agg({'packet_size': 'mean'}).reset_index()
    nt_hourly.rename(columns={'timestamp': 'log_time'}, inplace=True)

    # Generate predictions based on hourly mean packet_size
    pred_list = []
    for _, row in nt_hourly.iterrows():
        packet_size = row['packet_size']
        anomaly_score = calculate_anomaly_score(packet_size)
        prediction = 1 if is_anomaly(packet_size) else 0
        traffic_type = 'anomaly' if prediction == 1 else 'normal'
        pred_list.append({'log_time': row['log_time'], 'prediction': prediction, 'anomaly_score': anomaly_score, 'traffic_type': traffic_type})

    pred_df = pd.DataFrame(pred_list)
    return pred_df

def save_csvs(nt_df, pred_df):
    nt_df.to_csv(NETWORK_TRAFFIC_CSV, index=False)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

def update_mysql(nt_df, pred_df):
    conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME, port=DB_PORT)
    cursor = conn.cursor()

    # Update network_traffic table
    cursor.execute(f"DELETE FROM {NETWORK_TRAFFIC_TABLE}")
    for _, row in nt_df.iterrows():
        cursor.execute(f"INSERT INTO {NETWORK_TRAFFIC_TABLE} (source_ip, dest_ip, timestamp, packet_size) VALUES (%s, %s, %s, %s)",
                       (row['source_ip'], row['dest_ip'], row['timestamp'], row['packet_size']))

    # Update predictions table
    cursor.execute(f"DELETE FROM {PREDICTIONS_TABLE}")
    for _, row in pred_df.iterrows():
        cursor.execute(f"INSERT INTO {PREDICTIONS_TABLE} (log_time, prediction, anomaly_score, traffic_type) VALUES (%s, %s, %s, %s)",
                       (row['log_time'], row['prediction'], row['anomaly_score'], row['traffic_type']))

    conn.commit()
    cursor.close()
    conn.close()

def sync_metabase():
    # Assuming update_metabase_dashboard is implemented in visualization.py or similar
    from visualization import update_metabase_dashboard
    update_metabase_dashboard(pd.read_csv(PREDICTIONS_CSV), METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD)

def verify_alignment(nt_df, pred_df):
    # Check row counts
    nt_count = len(nt_df)
    pred_count = len(pred_df)
    print(f"network_traffic.csv rows: {nt_count}")
    print(f"predictions.csv rows: {pred_count}")

    # Check timestamps align hourly
    nt_hourly = nt_df.set_index('timestamp').resample('h').agg({'packet_size': 'mean'})
    pred_hourly = pred_df.set_index('log_time')
    aligned_hours = len(nt_hourly.index.intersection(pred_hourly.index))
    print(f"Aligned hourly timestamps: {aligned_hours}")

    # Check anomalies
    anomalies_nt = nt_df[nt_df['packet_size'] > ANOMALY_THRESHOLD]
    anomalies_pred = pred_df[pred_df['prediction'] == 1]
    print(f"Anomalies in network_traffic: {len(anomalies_nt)}")
    print(f"Anomalies in predictions: {len(anomalies_pred)}")

def main():
    nt_df, _ = load_csvs()  # Load network_traffic, ignore existing predictions
    pred_df = generate_predictions_from_traffic(nt_df)
    save_csvs(nt_df, pred_df)
    update_mysql(nt_df, pred_df)
    sync_metabase()
    verify_alignment(nt_df, pred_df)

if __name__ == "__main__":
    main()
