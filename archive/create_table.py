import pymysql
import pandas as pd
import logging
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE, PREDICTIONS_TABLE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_mysql_type(dtype, column_name):
    """Map pandas dtypes to MySQL types."""
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"  # Use DOUBLE for larger range than FLOAT
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"
    elif pd.api.types.is_bool_dtype(dtype):
        return "TINYINT(1)"
    else:
        # For IP addresses and other strings, use appropriate VARCHAR lengths
        if 'ip' in column_name.lower():
            return "VARCHAR(45)"  # IPv6 compatible
        elif 'mac' in column_name.lower():
            return "VARCHAR(17)"
        else:
            return "VARCHAR(255)"

def sanitize_column_name(col):
    """Sanitize column name to be MySQL compatible."""
    return col.replace('<', '_').replace('>', '_').replace(' ', '_').replace('-', '_')

def create_network_traffic_table(df):
    """Create network traffic table based on dataframe columns."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Drop tables if exists to recreate with proper structure (drop dependent tables first)
    cursor.execute(f"DROP TABLE IF EXISTS {PREDICTIONS_TABLE}")
    cursor.execute(f"DROP TABLE IF EXISTS {NETWORK_TRAFFIC_TABLE}")

    # Build CREATE TABLE statement
    columns_sql = []
    for col in df.columns:
        sanitized_col = sanitize_column_name(col)
        # Skip 'id' column since we add it as primary key
        if sanitized_col.lower() == 'id':
            continue
        mysql_type = get_mysql_type(df[col].dtype, col)
        columns_sql.append(f"`{sanitized_col}` {mysql_type}")

    # Add primary key
    columns_sql.insert(0, "id BIGINT AUTO_INCREMENT PRIMARY KEY")

    create_sql = f"""
    CREATE TABLE {NETWORK_TRAFFIC_TABLE} (
        {', '.join(columns_sql)}
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    logger.info(f"Creating table {NETWORK_TRAFFIC_TABLE} with SQL: {create_sql}")
    cursor.execute(create_sql)
    logger.info(f"Created table {NETWORK_TRAFFIC_TABLE}")

    conn.commit()
    cursor.close()
    conn.close()

def create_predictions_table():
    """Create predictions/anomalies table."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Drop table if exists
    cursor.execute(f"DROP TABLE IF EXISTS {PREDICTIONS_TABLE}")

    create_sql = f"""
    CREATE TABLE {PREDICTIONS_TABLE} (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        log_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        prediction TINYINT(1),
        anomaly_score FLOAT,
        traffic_type VARCHAR(50),
        is_anomaly TINYINT(1) GENERATED ALWAYS AS (CASE WHEN prediction = 1 THEN 1 ELSE 0 END) STORED,
        session_id BIGINT,
        model_type VARCHAR(50),
        confidence_score FLOAT,
        FOREIGN KEY (session_id) REFERENCES {NETWORK_TRAFFIC_TABLE}(id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    logger.info(f"Creating table {PREDICTIONS_TABLE}")
    cursor.execute(create_sql)
    logger.info(f"Created table {PREDICTIONS_TABLE}")

    conn.commit()
    cursor.close()
    conn.close()

def verify_tables():
    """Verify table creation."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Verify network traffic table
    cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{DB_NAME}' AND table_name = '{NETWORK_TRAFFIC_TABLE}'")
    if cursor.fetchone()[0] == 0:
        logger.error(f"Error: Table {NETWORK_TRAFFIC_TABLE} does not exist")
    else:
        cursor.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = '{DB_NAME}' AND table_name = '{NETWORK_TRAFFIC_TABLE}'")
        nt_columns = cursor.fetchone()[0]
        logger.info(f"Table {NETWORK_TRAFFIC_TABLE} exists with {nt_columns} columns")

    # Verify predictions table
    cursor.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = '{DB_NAME}' AND table_name = '{PREDICTIONS_TABLE}'")
    if cursor.fetchone()[0] == 0:
        logger.error(f"Error: Table {PREDICTIONS_TABLE} does not exist")
    else:
        cursor.execute(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_schema = '{DB_NAME}' AND table_name = '{PREDICTIONS_TABLE}'")
        pred_columns = cursor.fetchone()[0]
        logger.info(f"Table {PREDICTIONS_TABLE} exists with {pred_columns} columns")

    cursor.close()
    conn.close()

def setup_database(df):
    """Main function to set up database tables."""
    try:
        logger.info("Starting database setup...")
        create_network_traffic_table(df)
        create_predictions_table()
        verify_tables()
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Error in database setup: {e}")
        raise

if __name__ == "__main__":
    # Import here to avoid circular import
    from csv_loader import load_and_preprocess_csv

    # Load sample data to determine table structure
    csv_path = 'data/ASNM-NBPOv2.csv'
    logger.info("Loading sample data to determine table structure...")
    df = load_and_preprocess_csv(csv_path, chunksize=1000)  # Small chunk for structure determination

    if df.empty:
        logger.error("No data loaded, cannot create tables")
        exit(1)

    setup_database(df)
