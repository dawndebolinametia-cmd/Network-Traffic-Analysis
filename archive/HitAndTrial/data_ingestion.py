import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, TRAFFIC_DATA_TABLE
from logging_utils import logger

def fetch_network_traffic_data():
    """
    Fetch network traffic data from MySQL database.

    Returns:
        pd.DataFrame: DataFrame containing network traffic data.
    """
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Query to fetch all data from the table
            query = f"SELECT * FROM {TRAFFIC_DATA_TABLE}"
            cursor.execute(query)

            # Fetch all rows
            rows = cursor.fetchall()

            # Get column names
            columns = [desc[0] for desc in cursor.description]

            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)

            logger.info(f"Successfully fetched {len(df)} rows from {TRAFFIC_DATA_TABLE}")

            return df

    except Error as e:
        logger.error(f"MySQL Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching data from database: {e}")
        return pd.DataFrame()
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
