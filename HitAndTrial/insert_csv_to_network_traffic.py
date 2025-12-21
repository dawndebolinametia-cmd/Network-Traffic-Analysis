import pandas as pd
import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, TRAFFIC_DATA_TABLE
from logging_utils import logger

TABLE_NAME = "network_traffic"

def insert_csv(csv_file_path):
    try:
        data = pd.read_csv(csv_file_path)
        logger.info(f"Read {len(data)} rows from {csv_file_path}")

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()

        insert_query = f"""
        INSERT INTO {TABLE_NAME} (log_time, request_method, response_code, bytes_sent, source_ip)
        VALUES (%s, %s, %s, %s, %s)
        """
        tuples = [tuple(x) for x in data.to_numpy()]
        cursor.executemany(insert_query, tuples)
        connection.commit()
        logger.info(f"Inserted {cursor.rowcount} rows into {TABLE_NAME}")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("MySQL connection closed")

if __name__ == "__main__":
    csv_file = "C:/Users/HP/Documents/DataAnalysisProject/synthetic_traffic_data.csv"
    insert_csv(csv_file)
