import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, TRAFFIC_DATA_TABLE
from logging_utils import logger

def insert_csv_to_mysql(csv_file_path, table_name=TRAFFIC_DATA_TABLE):
    try:
        # Read the CSV
        data = pd.read_csv(csv_file_path)
        logger.info(f"Read {len(data)} rows from {csv_file_path}")

        # Connect to MySQL
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        cursor = connection.cursor()

        # Create table if not exists
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            log_time DATETIME,
            request_method VARCHAR(10),
            response_code INT,
            bytes_sent INT,
            source_ip VARCHAR(45)
        )
        """
        cursor.execute(create_table_query)

        # Insert rows one by one
        insert_query = f"""
        INSERT INTO {table_name} (log_time, request_method, response_code, bytes_sent, source_ip)
        VALUES (%s, %s, %s, %s, %s)
        """
        data_tuples = list(data.itertuples(index=False, name=None))
        cursor.executemany(insert_query, data_tuples)
        connection.commit()

        logger.info(f"Inserted {cursor.rowcount} rows into {table_name}")

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
    except Error as e:
        logger.error(f"MySQL Error: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("MySQL connection closed")

if __name__ == "__main__":
    csv_file = "C:/Users/HP/Documents/DataAnalysisProject/synthetic_traffic_data.csv"
    insert_csv_to_mysql(csv_file)
