import pandas as pd
import mysql.connector
from mysql.connector import Error
import os
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
from logging_utils import logger

TABLE_NAME = "network_traffic"

def save_predictions_to_csv(predictions_df: pd.DataFrame, output_path: str = 'reports/predictions.csv'):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to CSV at {output_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {e}")
        raise

def insert_predictions_to_db(predictions_df: pd.DataFrame, table_name: str = TABLE_NAME):
    try:
        # Keep only columns matching your DB
        df = predictions_df[["log_time", "request_method", "response_code", "bytes_sent"]]

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():
            cursor = connection.cursor()
            insert_query = f"""
            INSERT INTO {table_name} (log_time, request_method, response_code, bytes_sent)
            VALUES (%s, %s, %s, %s)
            """
            data_tuples = [tuple(row) for row in df.values]
            cursor.executemany(insert_query, data_tuples)
            connection.commit()
            logger.info(f"Inserted {cursor.rowcount} prediction rows into {table_name}")

    except Error as e:
        logger.error(f"Error inserting predictions: {e}")
        raise

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def save_predictions(predictions_df: pd.DataFrame, csv_path: str = 'reports/predictions.csv'):
    save_predictions_to_csv(predictions_df, csv_path)
    # DB insert optional if table matches
    # insert_predictions_to_db(predictions_df)

if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv("predicted_anomalies.csv")  # Replace with your predictions
    # save_predictions(df)
    pass
