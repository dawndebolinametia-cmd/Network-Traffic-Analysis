import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_PORT

DB_NAME = "analytics_data_fresh"
TABLE_NAME = "prediction_anomaly"
CSV_FILE = "reports/predictions.csv"

def create_table_if_not_exists():
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()
        create_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            log_time DATETIME NOT NULL,
            request_method VARCHAR(10),
            response_code SMALLINT,
            bytes_sent INT
        );
        """
        cursor.execute(create_query)
        connection.commit()
        print(f"Table `{TABLE_NAME}` is ready.")
    except Error as e:
        print("Error creating table:", e)
    finally:
        cursor.close()
        connection.close()

def insert_csv_data():
    try:
        df = pd.read_csv(CSV_FILE)
        # Ensure datetime format
        df["log_time"] = pd.to_datetime(df["log_time"], errors="coerce")
        df = df.dropna(subset=["log_time"])

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()
        insert_query = f"""
        INSERT INTO {TABLE_NAME} (log_time, request_method, response_code, bytes_sent)
        VALUES (%s, %s, %s, %s)
        """
        data_tuples = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data_tuples)
        connection.commit()
        print(f"Inserted {cursor.rowcount} rows into `{TABLE_NAME}`.")
    except Error as e:
        print("Error inserting data:", e)
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_table_if_not_exists()
    insert_csv_data()
