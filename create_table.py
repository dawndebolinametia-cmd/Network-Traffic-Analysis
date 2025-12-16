import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

PREDICTION_TABLE = "prediction_anomaly"

try:
    connection = mysql.connector.connect(
        host=DB_HOST, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME, port=DB_PORT
    )
    cursor = connection.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
            log_time DATETIME,
            prediction INT,
            anomaly_score FLOAT,
            traffic_type VARCHAR(50)
        )
    """)
    connection.commit()
    print(f"Table '{PREDICTION_TABLE}' created successfully!")
except Exception as e:
    print("Error creating table:", e)
finally:
    cursor.close()
    connection.close()
