import pandas as pd
import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE

def insert_network_traffic_csv(csv_file_path='network_traffic.csv'):
    try:
        data = pd.read_csv(csv_file_path)
        print(f"Read {len(data)} rows from {csv_file_path}")

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = connection.cursor()

        insert_query = f"""
        INSERT INTO {NETWORK_TRAFFIC_TABLE} (source_ip, dest_ip, timestamp, packet_size)
        VALUES (%s, %s, %s, %s)
        """
        tuples = [tuple(x) for x in data.to_numpy()]
        cursor.executemany(insert_query, tuples)
        connection.commit()
        print(f"Inserted {cursor.rowcount} rows into {NETWORK_TRAFFIC_TABLE}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed")

if __name__ == "__main__":
    insert_network_traffic_csv()
