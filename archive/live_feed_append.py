import pandas as pd
import datetime
from config import *

def append_live_data():
    # Read latest timestamp from network_traffic.csv
    df_nt = pd.read_csv(NETWORK_TRAFFIC_CSV)
    df_nt['timestamp'] = pd.to_datetime(df_nt['timestamp'])
    last_ts = df_nt['timestamp'].max()

    # Get current runtime time
    current_time = get_current_time()

    # Determine start hour: last_ts + 1 hour, floored to hour
    start_hour = (last_ts + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    # End hour: up to December 18th for initial fill, then live updates hourly after
    fill_date = datetime.datetime(2025, 12, 18, 0, 0, 0)
    if current_time > fill_date:
        end_hour = current_time.replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=1)
    else:
        end_hour = fill_date

    if start_hour > end_hour:
        print("No new hours to append.")
        return

    new_rows_nt = []
    new_rows_pred = []

    # Generate deterministic synthetic data for each hour
    for hour in pd.date_range(start=start_hour, end=end_hour, freq='H'):
        # Generate network_traffic row
        packet_size = generate_packet_size(hour.hour)
        source_ip, dest_ip = generate_ips(hour.hour)
        new_rows_nt.append({
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'timestamp': hour,
            'packet_size': packet_size
        })

        # Generate predictions row (one per hour, aligned)
        prediction = 1 if is_anomaly(packet_size, hour.hour) else 0
        anomaly_score = calculate_anomaly_score(packet_size, hour.hour)
        traffic_type = 'anomaly' if prediction == 1 else 'normal'
        new_rows_pred.append({
            'log_time': hour,
            'prediction': prediction,
            'anomaly_score': anomaly_score,
            'traffic_type': traffic_type
        })

    # Append to CSVs
    if new_rows_nt:
        df_new_nt = pd.DataFrame(new_rows_nt)
        df_new_nt.to_csv(NETWORK_TRAFFIC_CSV, mode='a', header=False, index=False)
    if new_rows_pred:
        df_new_pred = pd.DataFrame(new_rows_pred)
        df_new_pred.to_csv(PREDICTIONS_CSV, mode='a', header=False, index=False)

    print(f"Appended {len(new_rows_nt)} rows to {NETWORK_TRAFFIC_CSV} and {len(new_rows_pred)} rows to {PREDICTIONS_CSV}.")

if __name__ == "__main__":
    append_live_data()
