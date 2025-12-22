import pandas as pd
import mysql.connector
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

TABLE_NAME = "prediction_anomaly"

def insert_csv_to_db(csv_file):
    df = pd.read_csv(csv_file)

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

    for row in df.itertuples(index=False, name=None):
        cursor.execute(insert_query, row)

    connection.commit()
    cursor.close()
    connection.close()
    print(f"{len(df)} rows inserted into {TABLE_NAME}.")

if __name__ == "__main__":
    insert_csv_to_db("reports/predictions.csv")
