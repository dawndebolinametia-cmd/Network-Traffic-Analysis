import pandas as pd
import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

DB_NAME = 'analytics_data_fresh'
TABLE_NAME = 'prediction_anomaly'

# Load the CSV
df = pd.read_csv('reports/predictions.csv')

# Connect to MySQL
conn = mysql.connector.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    database=DB_NAME,
    port=DB_PORT
)
cursor = conn.cursor()

# Insert rows
insert_query = f"""
INSERT INTO {TABLE_NAME} (log_time, request_method, response_code, bytes_sent)
VALUES (%s, %s, %s, %s)
"""

for row in df.itertuples(index=False, name=None):
    cursor.execute(insert_query, row)

conn.commit()
cursor.close()
conn.close()
print(f"Inserted {len(df)} rows into {TABLE_NAME}")
