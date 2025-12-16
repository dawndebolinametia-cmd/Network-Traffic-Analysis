import pandas as pd
import mysql.connector
from mysql.connector import Error
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

def create_anomaly_predictions_table():
    """
    Create a new table for storing ML anomaly predictions.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Drop table if exists (for demo purposes)
            cursor.execute("DROP TABLE IF EXISTS anomaly_predictions")

            # Create table
            create_table_query = """
            CREATE TABLE anomaly_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                log_time DATETIME NOT NULL,
                prediction INT NOT NULL,
                anomaly_score FLOAT NOT NULL,
                traffic_type VARCHAR(50)
            )
            """
            cursor.execute(create_table_query)
            connection.commit()
            print("Table 'anomaly_predictions' created successfully.")

    except Error as e:
        print(f"Error creating table: {e}")
        raise

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def insert_predictions_from_csv(csv_path='reports/predictions.csv'):
    """
    Read predictions from CSV and insert into the anomaly_predictions table.
    """
    try:
        # Read CSV
        predictions_df = pd.read_csv(csv_path)
        print(f"Read {len(predictions_df)} predictions from {csv_path}")

        # Connect to database
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Insert query
            insert_query = """
            INSERT INTO anomaly_predictions (log_time, prediction, anomaly_score, traffic_type)
            VALUES (%s, %s, %s, %s)
            """

            # Prepare data (assuming columns exist; handle missing traffic_type if needed)
            data_tuples = []
            for _, row in predictions_df.iterrows():
                # If traffic_type is missing, set to None or default
                traffic_type = row.get('traffic_type', 'unknown') if 'traffic_type' in row else 'unknown'
                data_tuples.append((
                    row['log_time'],
                    int(row['prediction']),
                    float(row['anomaly_score']),
                    traffic_type
                ))

            # Execute batch insert
            cursor.executemany(insert_query, data_tuples)
            connection.commit()
            print(f"Inserted {cursor.rowcount} predictions into 'anomaly_predictions' table.")

    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        raise
    except KeyError as e:
        print(f"Missing column in CSV: {e}")
        raise
    except Error as e:
        print(f"Error inserting data: {e}")
        raise

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

def verify_table():
    """
    Verify the table was created and data inserted.
    """
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM anomaly_predictions")
            count = cursor.fetchone()[0]
            print(f"Table 'anomaly_predictions' contains {count} rows.")

            # Show sample data
            cursor.execute("SELECT * FROM anomaly_predictions LIMIT 5")
            rows = cursor.fetchall()
            print("Sample data:")
            for row in rows:
                print(row)

    except Error as e:
        print(f"Error verifying table: {e}")
        raise

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    # Step 1: Create table
    create_anomaly_predictions_table()

    # Step 2: Insert predictions from CSV
    insert_predictions_from_csv()

    # Step 3: Verify
    verify_table()

    print("Demo completed successfully!")
