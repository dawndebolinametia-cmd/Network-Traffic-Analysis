
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

    # Start from 10 Dec 15:05
    start_time = datetime(2025, 12, 10, 15, 5)

    # Find last timestamp in network_traffic
    cursor.execute(f"SELECT MAX(log_time) FROM {NETWORK_TRAFFIC_TABLE}")
    last_timestamp = cursor.fetchone()[0]
    if last_timestamp is not None and last_timestamp > start_time:
        current_time = last_timestamp + timedelta(hours=1)
    else:
        current_time = start_time

    now = datetime.now()

    while current_time <= now:
        # Network traffic
        source_ip = f"192.168.{random.randint(0,254)}.{random.randint(1,254)}"
        dest_ip = f"10.0.{random.randint(0,254)}.{random.randint(1,254)}"
        request_method = random.choice(['GET','POST','PUT','DELETE'])
        response_code = random.choice([200,201,400,401,403,404,500])
        bytes_sent = random.randint(40,1500)
        packet_size = random.randint(40,1500)

        cursor.execute(f"""
            INSERT INTO {NETWORK_TRAFFIC_TABLE}
            (log_time, request_method, response_code, bytes_sent, source_ip, dest_ip, packet_size)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (current_time, request_method, response_code, bytes_sent, source_ip, dest_ip, packet_size))

        # Prediction anomaly
        prediction = random.choice([0,1])
        anomaly_score = round(random.uniform(0,1),3)
        traffic_type = random.choice(['HTTP','HTTPS','FTP','SSH'])

        cursor.execute(f"""
            INSERT INTO {PREDICTION_ANOMALY_TABLE}
            (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s,%s,%s,%s)
        """, (current_time, prediction, anomaly_score, traffic_type))

        current_time += timedelta(hours=1)

    connection.commit()
    print("New hourly data appended correctly!")

except Exception as e:
    print("Error:", e)

finally:
    cursor.close()
    connection.close()
