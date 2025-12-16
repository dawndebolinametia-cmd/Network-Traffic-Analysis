import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

try:
    print("Trying to connect to MySQL...")
    print(f"Host: {DB_HOST}, User: {DB_USER}, Database: {DB_NAME}, Port: {DB_PORT}")
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        connection_timeout=5
    )
    if conn.is_connected():
        print("Connected to MySQL!")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM network_traffic")
        count = cursor.fetchone()[0]
        print(f"Rows in network_traffic table: {count}")
        cursor.close()
    conn.close()
except mysql.connector.Error as e:
    print(f"MySQL Error: {e}")
except Exception as e:
    print(f"General Error: {e}")
