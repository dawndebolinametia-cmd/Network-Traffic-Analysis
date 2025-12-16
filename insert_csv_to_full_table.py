import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
from logging_utils import logger

TABLE_NAME = "network_traffic_full"

def insert_full_csv(csv_file_path, table_name=TABLE_NAME):
    try:
        # Read CSV
        data = pd.read_csv(csv_file_path)
        logger.info(f"Read {len(data)} rows from {csv_file_path}")

        # Map/rename columns to match network_traffic_full
        data_full = pd.DataFrame()
        data_full['timestamp'] = data['log_time']
        data_full['src_ip'] = data['source_ip']
        data_full['dst_ip'] = '0.0.0.0'  # placeholder
        data_full['source_port'] = 0      # ph
        data_full['destination_port'] = 0 # ph
        data_full['protocol'] = 'HTTP'    # ph
        data_full['packet_size'] = data['bytes_sent']
        data_full['duration'] = 0.0       # ph
        data_full['label'] = 'normal'     # ph

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
            timestamp DATETIME,
            src_ip VARCHAR(45),
            dst_ip VARCHAR(45),
            source_port INT,
            destination_port INT,
            protocol VARCHAR(10),
            packet_size INT,
            duration FLOAT,
            label VARCHAR(50)
        )
        """
        cursor.execute(create_table_query)

        # Insert data
        insert_query = f"""
        INSERT INTO {table_name} 
        (timestamp, src_ip, dst_ip, source_port, destination_port, protocol, packet_size, duration, label)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data_tuples = [tuple(x) for x in data_full.to_numpy()]
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
    insert_full_csv(csv_file)
