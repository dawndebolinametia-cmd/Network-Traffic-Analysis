import pymysql
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE
from csv_loader import load_and_preprocess_csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def insert_data_in_chunks(df, table_name, chunk_size=1000):
    """Insert dataframe into MySQL table in chunks."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    total_rows = len(df)
    inserted_rows = 0

    logger.info(f"Starting insertion of {total_rows} rows into {table_name}")

    for i in range(0, total_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]

        # Prepare column names and placeholders
        columns = [f"`{col}`" for col in chunk.columns]
        placeholders = ','.join(['%s'] * len(columns))

        # Build INSERT statement
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

        # Convert chunk to list of tuples, handling NaN values
        values = []
        for _, row in chunk.iterrows():
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append(None)
                elif isinstance(val, np.integer):
                    row_values.append(int(val))
                elif isinstance(val, np.floating):
                    # Handle inf, -inf, and very large floats
                    if np.isinf(val) or np.isnan(val) or abs(val) > 1e38:
                        row_values.append(None)
                    else:
                        row_values.append(float(val))
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_values.append(val)
                elif isinstance(val, (bool, np.bool_)):
                    row_values.append(int(val))
                else:
                    row_values.append(str(val))
            values.append(tuple(row_values))

        try:
            cursor.executemany(insert_sql, values)
            conn.commit()
            inserted_rows += len(chunk)
            logger.info(f"Inserted chunk {i//chunk_size + 1}, rows {i+1}-{min(i+chunk_size, total_rows)}, total inserted: {inserted_rows}")
        except Exception as e:
            logger.error(f"Error inserting chunk {i//chunk_size + 1}: {e}")
            conn.rollback()
            raise

    cursor.close()
    conn.close()
    logger.info(f"Successfully inserted {inserted_rows} rows into {table_name}")

def insert_network_traffic_data(df):
    """Main function to insert network traffic data."""
    try:
        logger.info("Starting data insertion process...")

        if df.empty:
            logger.error("No data to insert")
            return

        # Insert data
        insert_data_in_chunks(df, NETWORK_TRAFFIC_TABLE)

        logger.info("Data insertion completed successfully")

    except Exception as e:
        logger.error(f"Error in data insertion: {e}")
        raise

def get_table_row_count(table_name):
    """Get row count from table."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return count

if __name__ == "__main__":
    csv_path = 'data/ASNM-NBPOv2.csv'

    # Load and preprocess data
    df = load_and_preprocess_csv(csv_path)

    # Insert data
    insert_network_traffic_data(df)

    # Verify insertion
    row_count = get_table_row_count(NETWORK_TRAFFIC_TABLE)
    logger.info(f"Verification: {row_count} rows in {NETWORK_TRAFFIC_TABLE}")
