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

    # Find last log_time in network_traffic
    cursor.execute(f"SELECT MAX(log_time) FROM {NETWORK_TRAFFIC_TABLE}")
    last_timestamp = cursor.fetchone()[0]

    if last_timestamp is None:
        last_timestamp = datetime(2025, 12, 10, 15, 5)  # fallback if table is empty

    # Generate hourly synthetic data starting AFTER last_timestamp
    current_time = last_timestamp + timedelta(hours=1)
    now = datetime.now()

    while current_time <= now:
        source_ip = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        dest_ip = f"10.0.{random.randint(0,255)}.{random.randint(1,254)}"
        packet_size = random.randint(40, 1500)
        request_method = random.choice(['GET', 'POST', 'PUT', 'DELETE'])
        response_code = random.choice([200, 201, 400, 401, 403, 404, 500])
        bytes_sent = random.randint(100, 5000)

        # Insert new network traffic
        cursor.execute(f"""
            INSERT INTO {NETWORK_TRAFFIC_TABLE} 
            (log_time, request_method, response_code, bytes_sent, source_ip, dest_ip, packet_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (current_time, request_method, response_code, bytes_sent, source_ip, dest_ip, packet_size))

        # Insert corresponding anomaly prediction
        prediction = random.choice([0,1])
        anomaly_score = round(random.uniform(0.0,1.0), 6)
        traffic_type = 'anomaly' if prediction == 1 else 'normal'

        cursor.execute(f"""
            INSERT INTO {PREDICTION_ANOMALY_TABLE} 
            (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s)
        """, (current_time, prediction, anomaly_score, traffic_type))

        current_time += timedelta(hours=1)

    connection.commit()
    print("Hourly data appended successfully! ✅")

except Exception as e:
    print("Error:", e)

finally:
    cursor.close()
    connection.close()
