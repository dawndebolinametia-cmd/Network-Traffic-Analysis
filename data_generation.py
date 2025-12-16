import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from logging_utils import logger

def generate_synthetic_traffic_data(num_rows=50000):
    """
    Generate synthetic network traffic data with realistic patterns.

    Args:
        num_rows (int): Number of rows to generate

    Returns:
        pd.DataFrame: Generated traffic data
    """
    logger.info(f"Generating {num_rows} rows of synthetic network traffic data")

    # Base timestamp
    base_time = datetime(2023, 1, 1, 0, 0, 0)

    data = []

    # IP ranges for realistic generation
    src_ips = [f"192.168.{i}.{j}" for i in range(1, 5) for j in range(1, 255)]
    dst_ips = [f"10.{i}.{j}.{k}" for i in range(0, 5) for j in range(0, 255) for k in range(1, 255)]

    protocols = ['TCP', 'UDP', 'ICMP']
    common_ports = [80, 443, 22, 53, 21, 25, 110, 143, 993, 995]

    for i in range(num_rows):
        # Timestamp with some time progression
        timestamp = base_time + timedelta(seconds=i*random.uniform(0.1, 5))

        # Random IPs
        src_ip = random.choice(src_ips)
        dst_ip = random.choice(dst_ips)

        # Source and destination ports
        source_port = random.randint(1024, 65535)
        destination_port = random.choice(common_ports) if random.random() < 0.7 else random.randint(1024, 65535)

        # Protocol
        protocol = random.choice(protocols)

        # Packet size - normal traffic smaller, anomalies larger
        if random.random() < 0.05:  # 5% anomalies
            label = 'anomaly'
            packet_size = random.randint(1000, 10000)  # Larger packets for anomalies
            duration = random.uniform(2.0, 10.0)  # Longer duration
        else:
            label = 'normal'
            packet_size = random.randint(64, 1500)  # Normal packet sizes
            duration = random.uniform(0.1, 2.0)  # Shorter duration

        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'source_port': source_port,
            'destination_port': destination_port,
            'protocol': protocol,
            'packet_size': packet_size,
            'duration': round(duration, 2),
            'label': label
        })

    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} rows with {len(df[df['label']=='anomaly'])} anomalies")
    return df

def save_synthetic_data(df, filename='synthetic_traffic_data.csv'):
    """
    Save the generated data to CSV file.

    Args:
        df (pd.DataFrame): Data to save
        filename (str): Output filename
    """
    try:
        df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(df)} rows to {filename}")
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")

if __name__ == "__main__":
    # Generate and save the data
    df = generate_synthetic_traffic_data(50000)
    with open('generation_log.txt', 'w') as f:
        f.write(f"Generated {len(df)} rows\n")
        f.write(f"Anomalies: {len(df[df['label']=='anomaly'])}\n")
    save_synthetic_data(df)
    with open('generation_log.txt', 'a') as f:
        f.write("Data generation completed\n")
