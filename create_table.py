import pymysql
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE, PREDICTIONS_TABLE

def create_tables():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        cursor = conn.cursor()

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {NETWORK_TRAFFIC_TABLE} (
            source_ip VARCHAR(15),
            dest_ip VARCHAR(15),
            timestamp DATETIME,
            packet_size INT
        )
        """)
        print(f"Created table {NETWORK_TRAFFIC_TABLE}")

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {PREDICTIONS_TABLE} (
            log_time DATETIME,
            prediction INT,
            anomaly_score FLOAT,
            traffic_type VARCHAR(10)
        )
        """)
        print(f"Created table {PREDICTIONS_TABLE}")

        conn.commit()

        # Verify table existence
        cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{DB_NAME}' AND table_name = '{NETWORK_TRAFFIC_TABLE}'")
        if cursor.fetchone()[0] == 0:
            print(f"Error: Table {NETWORK_TRAFFIC_TABLE} does not exist")
        else:
            cursor.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = '{DB_NAME}' AND table_name = '{NETWORK_TRAFFIC_TABLE}'")
            nt_columns = cursor.fetchone()[0]
            print(f"Table {NETWORK_TRAFFIC_TABLE} exists with {nt_columns} columns")

        cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{DB_NAME}' AND table_name = '{PREDICTIONS_TABLE}'")
        if cursor.fetchone()[0] == 0:
            print(f"Error: Table {PREDICTIONS_TABLE} does not exist")
        else:
            cursor.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = '{DB_NAME}' AND table_name = '{PREDICTIONS_TABLE}'")
            pred_columns = cursor.fetchone()[0]
            print(f"Table {PREDICTIONS_TABLE} exists with {pred_columns} columns")

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    create_tables()
