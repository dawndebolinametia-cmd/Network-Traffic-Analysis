import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE, PREDICTION_ANOMALY_TABLE
from datetime import datetime, timedelta
import random

try:
    connection = mysql.connector.connect(
        host=DB_HOST, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME, port=DB_PORT
    )
    cursor = connection.cursor()

    # Create network_traffic table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {NETWORK_TRAFFIC_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            source_ip VARCHAR(45),
            dest_ip VARCHAR(45),
            timestamp DATETIME,
            packet_size INT
        )
    """)
    print(f"Table '{NETWORK_TRAFFIC_TABLE}' created successfully!")

    # Create prediction_anomaly table
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_ANOMALY_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            source_ip VARCHAR(45),
            timestamp DATETIME,
            anomaly_flag INT,
            diff_seconds FLOAT
        )
    """)
    print(f"Table '{PREDICTION_ANOMALY_TABLE}' created successfully!")

    # Populate network_traffic table with synthetic data
    for i in range(50):
        source_ip = f"192.168.1.{random.randint(1, 254)}"
        dest_ip = f"10.0.0.{random.randint(1, 254)}"
        timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1000))
        packet_size = random.randint(40, 1500)

        cursor.execute(f"""
            INSERT INTO {NETWORK_TRAFFIC_TABLE} (source_ip, dest_ip, timestamp, packet_size)
            VALUES (%s, %s, %s, %s)
        """, (source_ip, dest_ip, timestamp, packet_size))

    # Populate prediction_anomaly table
    cursor.execute(f"SELECT source_ip, timestamp FROM {NETWORK_TRAFFIC_TABLE}")
    rows = cursor.fetchall()

    for row in rows:
        source_ip, timestamp = row
        anomaly_flag = random.choice([0, 1])
        diff_seconds = random.uniform(0.1, 5.0)

        cursor.execute(f"""
            INSERT INTO {PREDICTION_ANOMALY_TABLE} (source_ip, timestamp, anomaly_flag, diff_seconds)
            VALUES (%s, %s, %s, %s)
        """, (source_ip, timestamp, anomaly_flag, diff_seconds))

    connection.commit()
    print("Tables populated successfully!")

except Exception as e:
    print("Error:", e)

finally:
    cursor.close()
    connection.close()
