import os
import csv
import pymysql
from config import (
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    NETWORK_TRAFFIC_TABLE, PREDICTIONS_TABLE,
    NETWORK_TRAFFIC_CSV, PREDICTIONS_CSV
)

def load_csv_into_table(csv_file, table, columns):
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = tuple(row[col] for col in columns)
            placeholders = ','.join(['%s'] * len(columns))
            sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
            cursor.execute(sql, vals)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"{table} populated from {csv_file}")

# Load network_traffic.csv
load_csv_into_table(
    NETWORK_TRAFFIC_CSV,
    NETWORK_TRAFFIC_TABLE,
    ['source_ip','dest_ip','timestamp','packet_size']
)

# Load predictions.csv
load_csv_into_table(
    PREDICTIONS_CSV,
    PREDICTIONS_TABLE,
    ['log_time','prediction','anomaly_score','traffic_type']
)
