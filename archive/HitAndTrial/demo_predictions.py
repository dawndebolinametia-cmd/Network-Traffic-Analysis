<<<<<<< HEAD:demo_predictions.py
import pandas as pd
import pymysql
import os

PREDICTION_TABLE = "prediction_anomaly"
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "Anime#210305"  # your password
DB_NAME = "analytics_data_fresh"
DB_PORT = 3306  # default MySQL port

# Demo DataFrame
demo_data = pd.DataFrame({
    "log_time": pd.to_datetime(["2025-12-15 09:00:00", "2025-12-15 10:00:00"]),
    "prediction": [0, 1],
    "anomaly_score": [0.1, 0.95],
    "traffic_type": ["normal", "suspicious"]
})

# CSV path
csv_path = 'reports/predictions.csv'

def create_table():
    try:
        connection = pymysql.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
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
        print(f" Table '{PREDICTION_TABLE}' is ready! ")
    except pymysql.MySQLError as e:
        print(" Error creating table:", e)
    finally:
        cursor.close()
        connection.close()

def save_csv(df):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f" Predictions saved to CSV at '{csv_path}'")

def insert_into_db(df):
    try:
        connection = pymysql.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()
        insert_query = f"""
            INSERT INTO {PREDICTION_TABLE} (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s)
        """
        data_tuples = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data_tuples)
        connection.commit()
        print(f"Inserted {cursor.rowcount} predictions into '{PREDICTION_TABLE}' ")
    except pymysql.MySQLError as e:
        print(" Error inserting predictions:", e)
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_table()
    save_csv(demo_data)
    insert_into_db(demo_data)
    print(" Demo run complete! ")
=======
import pandas as pd
import pymysql
import os

PREDICTION_TABLE = "prediction_anomaly"
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "Anime#210305"  # your password 😈
DB_NAME = "analytics_data_fresh"
DB_PORT = 3306  # default MySQL port

# Demo DataFrame
demo_data = pd.DataFrame({
    "log_time": pd.to_datetime(["2025-12-15 09:00:00", "2025-12-15 10:00:00"]),
    "prediction": [0, 1],
    "anomaly_score": [0.1, 0.95],
    "traffic_type": ["normal", "suspicious"]
})

# CSV path
csv_path = 'reports/predictions.csv'

def create_table():
    try:
        connection = pymysql.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
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
        print(f" Table '{PREDICTION_TABLE}' is ready! 😎")
    except pymysql.MySQLError as e:
        print(" Error creating table:", e)
    finally:
        cursor.close()
        connection.close()

def save_csv(df):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f" Predictions saved to CSV at '{csv_path}' 😈")

def insert_into_db(df):
    try:
        connection = pymysql.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()
        insert_query = f"""
            INSERT INTO {PREDICTION_TABLE} (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s)
        """
        data_tuples = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data_tuples)
        connection.commit()
        print(f"Inserted {cursor.rowcount} predictions into '{PREDICTION_TABLE}' 😏")
    except pymysql.MySQLError as e:
        print(" Error inserting predictions:", e)
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_table()
    save_csv(demo_data)
    insert_into_db(demo_data)
    print(" Demo run complete! 🎉")
>>>>>>> a8d46ec4a665e9c954a07d26c29c9eef5f33215a:archive/HitAndTrial/demo_predictions.py
