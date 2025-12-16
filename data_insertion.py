import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, TRAFFIC_DATA_TABLE
from logging_utils import logger

def insert_csv_to_mysql(csv_file_path, table_name=TRAFFIC_DATA_TABLE):
    try:
        data = pd.read_csv(csv_file_path)
        logger.info(f"Read {len(data)} rows from {csv_file_path}")

        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Table already exists, we just insert
            insert_query = f"""
            INSERT INTO {table_name} (
                log_time, request_method, response_code, bytes_sent, source_ip
            ) VALUES (%s, %s, %s, %s, %s)
            """

            data_tuples = []
            for _, row in data.iterrows():
                data_tuples.append((
                    row.get('log_time'),
                    row.get('request_method'),
                    row.get('response_code'),
                    row.get('bytes_sent'),
                    row.get('source_ip')
                ))

            cursor.executemany(insert_query, data_tuples)
            connection.commit()
            logger.info(f"Successfully inserted {cursor.rowcount} rows into {table_name}")
            return True

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
        return False
    except Error as e:
        logger.error(f"MySQL Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error inserting data: {e}")
        return False
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            logger.info("MySQL connection closed")

def generate_synthetic_data(num_rows=1000, output_file='synthetic_traffic_data.csv'):
    import random
    from datetime import datetime, timedelta

    data = []
    base_time = datetime.now()
    methods = ['GET', 'POST', 'PUT', 'DELETE']
    for _ in range(num_rows):
        log_time = base_time + timedelta(seconds=random.randint(0, 86400))
        request_method = random.choice(methods)
        response_code = random.choice([200, 201, 400, 401, 403, 404, 500])
        bytes_sent = random.randint(100, 5000)
        source_ip = f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
        data.append({
            'log_time': log_time.strftime('%Y-%m-%d %H:%M:%S'),
            'request_method': request_method,
            'response_code': response_code,
            'bytes_sent': bytes_sent,
            'source_ip': source_ip
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logger.info(f"Generated {num_rows} synthetic rows and saved to {output_file}")
    return output_file

if __name__ == "__main__":
    csv_file = generate_synthetic_data(1000)
    if csv_file:
        success = insert_csv_to_mysql(csv_file)
        if success:
            logger.info("Data insertion completed successfully")
        else:
            logger.error("Data insertion failed")
