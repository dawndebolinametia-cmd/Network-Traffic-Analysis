import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

TABLE_NAME = "network_traffic"

def clean_and_insert(csv_file):
    try:
        df = pd.read_csv(csv_file)

        # Map CSV columns to your DB schema
        df.columns = [
            "log_time",        # datetime
            "request_method",  # varchar(10)
            "response_code",   # smallint
            "bytes_sent",      # int
            "src_ip",          # extra columns we’ll ignore for DB
            "dst_ip",
            "source_port",
            "destination_port",
            "protocol",
            "packet_size",
            "duration",
            "label"
        ]

        # Keep only the columns DB expects
        df = df[["log_time", "request_method", "response_code", "bytes_sent"]]

        # Convert timestamp
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

        inserted = 0
        for row in df.itertuples(index=False, name=None):
            try:
                cursor.execute(insert_query, row)
                inserted += 1
            except Exception as e:
                print("Insert failed for row:", row)
                print("Error:", e)

        connection.commit()
        print(f"Insertion complete. Successfully inserted {inserted} rows.")

    except Exception as e:
        print("Error:", e)

    finally:
        try:
            cursor.close()
            connection.close()
        except:
            pass


if __name__ == "__main__":
    clean_and_insert("C:/Users/HP/Documents/DataAnalysisProject/synthetic_traffic_data.csv")

