import pymysql
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Try with root user first
try:
    print("Trying with root user...")
    conn = pymysql.connect(
        host=DB_HOST,
        user='root',
        password='12345',
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES;")
    databases = cursor.fetchall()
    print("Available databases with root:")
    for db in databases:
        print(f"  {db[0]}")
    conn.close()
except Exception as e2:
    print(f"Root connection failed: {e2}")

try:
    # First connect without specifying database to list databases
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES;")
    databases = cursor.fetchall()
    print("Available databases:")
    for db in databases:
        print(f"  {db[0]}")
    conn.close()

    # Now try to connect to the specified database
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    print(f"Connection to database '{DB_NAME}' successful!")
    conn.close()
except Exception as e:
    print(f"Connection failed: {e}")
