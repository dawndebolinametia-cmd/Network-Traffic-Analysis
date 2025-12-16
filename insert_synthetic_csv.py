import pandas as pd
import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

CSV_FILE = "C:/Users/HP/Documents/DataAnalysisProject/synthetic_traffic_data.csv"
TABLE_NAME = "network_traffic"

# Read CSV
df = pd.read_csv(CSV_FILE)

# Connect to MySQL (without specifying database to avoid hanging)
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT
)
cursor = conn.cursor()

# Use the database
cursor.execute(f"USE {DB_NAME}")

# Insert rows
insert_query = f"""
INSERT INTO {TABLE_NAME} (log_time, request_method, response_code, bytes_sent, source_ip)
VALUES (%s, %s, %s, %s, %s)
"""
data_tuples = [tuple(x) for x in df.to_numpy()]
cursor.executemany(insert_query, data_tuples)
conn.commit()

print(f"Inserted {cursor.rowcount} rows into {TABLE_NAME}")

cursor.close()
conn.close()

