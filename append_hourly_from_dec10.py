import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE, PREDICTION_ANOMALY_TABLE
from datetime import datetime, timedelta
import random

try:
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = connection.cursor()

    # Start from 2025-12-10 15:05
    current_time = datetime(2025, 12, 10, 15, 5)
    now = datetime.now()

    while current_time <= now:
        # Network traffic data
        source_ip = f"192.168.1.{random.randint(1,254)}"
        dest_ip = f"10.0.0.{random.randint(1,254)}"
        packet_size = random.randint(40, 1500)

        cursor.execute(f"""
            INSERT INTO {NETWORK_TRAFFIC_TABLE} (log_time, source_ip, dest_ip, packet_size)
            VALUES (%s, %s, %s, %s)
        """, (current_time, source_ip, dest_ip, packet_size))

        # Prediction anomaly data
        prediction = random.choice([0,1])
        anomaly_score = random.uniform(0.0, 1.0)
        traffic_type = random.choice(['HTTP', 'HTTPS', 'FTP', 'SSH'])

        cursor.execute(f"""
            INSERT INTO {PREDICTION_ANOMALY_TABLE} 
            (log_time, source_ip, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s, %s)
        """, (current_time, source_ip, prediction, anomaly_score, traffic_type))

        # Next hour
        current_time += timedelta(hours=1)

    connection.commit()
    print("Hourly data from 2025-12-10 15:05 appended successfully!")

except Exception as e:
    print("Error:", e)

finally:
    cursor.close()
    connection.close()
