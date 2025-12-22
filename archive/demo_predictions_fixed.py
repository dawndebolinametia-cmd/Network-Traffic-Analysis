import pandas as pd
import mysql.connector
from mysql.connector import Error
import os

# Hardcode the correct database for demo
DB_HOST = '127.0.0.1'
DB_USER = 'root'
DB_PASSWORD = 'Anime#210305'
DB_NAME = 'analytics_data_fresh'
DB_PORT = 3306

PREDICTION_TABLE = "prediction_anomaly"
CSV_PATH = 'reports/predictions.csv'

# Demo DataFrame
demo_data = pd.DataFrame({
    "log_time": pd.to_datetime(["2025-12-15 09:00:00", "2025-12-15 10:00:00"]),
    "prediction": [0, 1],
    "anomaly_score": [0.1, 0.95],
    "traffic_type": ["normal", "suspicious"]
})

def create_table():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} (
                log_time DATETIME,
                prediction INT,
                anomaly_score FLOAT,
                traffic_type VARCHAR(50)
            )
        """)
        conn.commit()
        print(f" Table '{PREDICTION_TABLE}' ready in {DB_NAME}!")
    except Error as e:
        print(" Error creating table:", e)
    finally:
        cursor.close()
        conn.close()

def save_csv(df):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f" Predictions saved to CSV at '{CSV_PATH}'")

def insert_into_db(df):
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = conn.cursor()
        insert_query = f"""
            INSERT INTO {PREDICTION_TABLE} (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s)
        """
        data_tuples = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data_tuples)
        conn.commit()
        print(f" Inserted {cursor.rowcount} predictions into '{PREDICTION_TABLE}'")
    except Error as e:
        print(" Error inserting predictions:", e)
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_table()
    save_csv(demo_data)
    insert_into_db(demo_data)
    print(" Demo complete!")
