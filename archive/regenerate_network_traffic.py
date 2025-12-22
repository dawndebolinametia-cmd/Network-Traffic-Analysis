import pandas as pd
import os
from config import generate_packet_size, generate_ips

# Paths
REPORTS_PATH = os.path.join(os.getcwd(), 'reports')
PREDICTIONS_CSV = os.path.join(REPORTS_PATH, 'predictions.csv')
NETWORK_TRAFFIC_CSV = os.path.join(os.getcwd(), 'network_traffic.csv')

def regenerate_network_traffic():
    # Load predictions.csv
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    pred_df['log_time'] = pd.to_datetime(pred_df['log_time'])

    # List to hold network_traffic rows
    nt_rows = []

    # For each timestamp in predictions.csv
    for _, row in pred_df.iterrows():
        timestamp = row['log_time']
        hour = timestamp.hour

        # Generate deterministic values
        packet_size = generate_packet_size(hour)
        source_ip, dest_ip = generate_ips(hour)

        # Create row
        nt_row = {
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'timestamp': timestamp,
            'packet_size': packet_size
        }
        nt_rows.append(nt_row)

    # Create DataFrame and save
    nt_df = pd.DataFrame(nt_rows)
    nt_df.to_csv(NETWORK_TRAFFIC_CSV, index=False)
    print(f"Regenerated {len(nt_rows)} rows in network_traffic.csv")

if __name__ == "__main__":
    regenerate_network_traffic()
