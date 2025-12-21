import pandas as pd
import pymysql
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

CSV_FILE = "C:/Users/HP/Documents/DataAnalysisProject/synthetic_traffic_data.csv"
TABLE_NAME = "network_traffic"

try:
    # Read CSV
    df = pd.read_csv(CSV_FILE)
    print(f"Read {len(df)} rows from CSV")

    # Connect to MySQL using pymysql
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    print("Connected to database")

    cursor = conn.cursor()

    # Check current count
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    initial_count = cursor.fetchone()[0]
    print(f"Initial row count: {initial_count}")

    # Insert rows
    insert_query = f"""
    INSERT INTO {TABLE_NAME} (log_time, request_method, response_code, bytes_sent, source_ip)
    VALUES (%s, %s, %s, %s, %s)
    """
    data_tuples = [tuple(x) for x in df.to_numpy()]
    print(f"Prepared {len(data_tuples)} tuples for insertion")

    cursor.executemany(insert_query, data_tuples)
    print(f"Executed insert, rowcount: {cursor.rowcount}")

    conn.commit()
    print("Committed transaction")

    # Check final count
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    final_count = cursor.fetchone()[0]
    print(f"Final row count: {final_count}")

    print(f"Inserted {cursor.rowcount} rows into {TABLE_NAME}")

except Exception as e:
    print(f"Error: {e}")
    if 'conn' in locals():
        conn.rollback()
        print("Rolled back transaction")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()
        print("Connection closed")
