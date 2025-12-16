import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, TRAFFIC_DATA_TABLE

try:
    print("Trying to connect to MySQL...")
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        connection_timeout=5
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TRAFFIC_DATA_TABLE}")
    result = cursor.fetchone()
    count = result[0] if result else 0
    print(f"Rows in {TRAFFIC_DATA_TABLE}: {count}")
    conn.close()
except Exception as e:
    print("Error:", e)


