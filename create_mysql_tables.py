#!/usr/bin/env python3
"""
MySQL Table Creation Script for Network Traffic Analysis Project

This script creates all required MySQL tables with proper schemas:
- network_traffic: Matches all 904 columns from ASNM-NBPOv2.csv
- anomaly_results: Stores anomaly detection results
- supervised_predictions: Stores supervised learning predictions
- summary_stats: Stores aggregated statistics

Features:
- Proper data types mapping from pandas dtypes to MySQL types
- Safe table creation with DROP IF EXISTS
- Transaction handling with rollback on errors
- Comprehensive error handling and logging
- Index creation for performance optimization
"""

import pymysql
import pandas as pd
import logging
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_mysql_column_type(pandas_dtype, column_name):
    """
    Convert pandas dtype to appropriate MySQL column type.

    Args:
        pandas_dtype: pandas dtype object
        column_name: column name for special handling

    Returns:
        str: MySQL column type definition
    """
    dtype_str = str(pandas_dtype)

    if dtype_str == 'int64':
        # Handle special cases for large integers
        if 'id' in column_name.lower():
            return 'BIGINT'
        elif any(keyword in column_name.lower() for keyword in ['cnt', 'sum', 'total', 'count']):
            return 'BIGINT'
        else:
            return 'INT'
    elif dtype_str == 'float64':
        return 'DOUBLE'
    elif dtype_str == 'bool':
        return 'TINYINT(1)'
    elif dtype_str == 'object':
        # Check for IP addresses and MAC addresses
        if 'ip' in column_name.lower():
            return 'VARCHAR(45)'  # IPv6 addresses can be up to 45 chars
        elif 'mac' in column_name.lower():
            return 'VARCHAR(17)'  # MAC addresses are 17 chars
        else:
            return 'VARCHAR(255)'  # Default for other strings
    else:
        # Fallback for unknown types
        logger.warning(f"Unknown dtype {dtype_str} for column {column_name}, using VARCHAR(255)")
        return 'VARCHAR(255)'

def get_csv_column_types(csv_path):
    """
    Read CSV and return column type mapping.

    Args:
        csv_path: Path to the CSV file

    Returns:
        dict: Column name -> MySQL type mapping
    """
    try:
        # Read just the first row to get dtypes
        df = pd.read_csv(csv_path, sep=';', nrows=1)
        column_types = {}

        for col, dtype in df.dtypes.items():
            mysql_type = get_mysql_column_type(dtype, col)
            column_types[col] = mysql_type

        logger.info(f"Successfully mapped {len(column_types)} columns from CSV")
        return column_types

    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

def create_network_traffic_table(cursor, column_types):
    """
    Create the network_traffic table with all 904 columns.

    Args:
        cursor: MySQL cursor
        column_types: dict of column name -> MySQL type
    """
    logger.info("Creating network_traffic table...")

    # Temporarily disable foreign key checks to allow dropping tables with dependencies
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")

    # Drop dependent tables first
    dependent_tables = ['hourly_predictions', 'predictions', 'anomaly_results', 'supervised_predictions', 'summary_stats']
    for table in dependent_tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    # Drop main table
    cursor.execute("DROP TABLE IF EXISTS network_traffic")

    # Re-enable foreign key checks
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")

    # Build CREATE TABLE statement
    columns_sql = []
    indexes = []

    for col_name, col_type in column_types.items():
        # Skip 'id' column since we'll define it as primary key separately
        if col_name == 'id':
            continue

        # Escape column names with special characters
        safe_col_name = f'`{col_name}`'

        # Add NOT NULL for most columns (except potentially nullable ones)
        nullable = "NULL" if col_name in ['srcIP', 'dstIP', 'srcMAC', 'dstMAC'] else "NOT NULL"

        columns_sql.append(f"{safe_col_name} {col_type} {nullable}")

        # Create indexes for commonly queried columns
        if col_name in ['srcIP', 'dstIP', 'srcPort', 'dstPort', 'label']:
            indexes.append(f"INDEX idx_{col_name.replace('.', '_')} ({safe_col_name})")

    # Add primary key (using the 'id' column from CSV)
    columns_sql.insert(0, "`id` BIGINT NOT NULL PRIMARY KEY")

    # Combine all parts
    create_sql = f"""
    CREATE TABLE network_traffic (
        {','.join(columns_sql)},
        {','.join(indexes)}
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    cursor.execute(create_sql)
    logger.info("network_traffic table created successfully")

def create_anomaly_results_table(cursor):
    """Create the anomaly_results table."""
    logger.info("Creating anomaly_results table...")

    cursor.execute("DROP TABLE IF EXISTS anomaly_results")

    create_sql = """
    CREATE TABLE anomaly_results (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        original_id BIGINT NOT NULL,
        anomaly_score DOUBLE NOT NULL,
        is_anomaly TINYINT(1) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_original_id (original_id),
        INDEX idx_is_anomaly (is_anomaly),
        INDEX idx_anomaly_score (anomaly_score),
        FOREIGN KEY (original_id) REFERENCES network_traffic(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    cursor.execute(create_sql)
    logger.info("anomaly_results table created successfully")

def create_supervised_predictions_table(cursor):
    """Create the supervised_predictions table."""
    logger.info("Creating supervised_predictions table...")

    cursor.execute("DROP TABLE IF EXISTS supervised_predictions")

    create_sql = """
    CREATE TABLE supervised_predictions (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        original_id BIGINT NOT NULL,
        predicted_label INT NOT NULL,
        confidence_score DOUBLE NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_original_id (original_id),
        INDEX idx_predicted_label (predicted_label),
        FOREIGN KEY (original_id) REFERENCES network_traffic(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    cursor.execute(create_sql)
    logger.info("supervised_predictions table created successfully")

def create_summary_stats_table(cursor):
    """Create the summary_stats table for aggregated statistics."""
    logger.info("Creating summary_stats table...")

    cursor.execute("DROP TABLE IF EXISTS summary_stats")

    create_sql = """
    CREATE TABLE summary_stats (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        total_records BIGINT NOT NULL,
        anomaly_count BIGINT NOT NULL DEFAULT 0,
        normal_count BIGINT NOT NULL DEFAULT 0,
        supervised_predictions BIGINT NOT NULL DEFAULT 0,
        anomaly_percentage DECIMAL(5,2) NOT NULL DEFAULT 0.00,
        avg_anomaly_score DOUBLE NULL,
        min_anomaly_score DOUBLE NULL,
        max_anomaly_score DOUBLE NULL,
        model_accuracy DECIMAL(5,2) NULL,
        processing_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_processing_date (processing_date),
        UNIQUE KEY unique_processing_date (processing_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    cursor.execute(create_sql)
    logger.info("summary_stats table created successfully")

def create_predictions_table(cursor):
    """Create the predictions table for storing all model predictions."""
    logger.info("Creating predictions table...")

    cursor.execute("DROP TABLE IF EXISTS predictions")

    create_sql = """
    CREATE TABLE predictions (
        id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        session_id BIGINT NOT NULL,
        model_type VARCHAR(50) NOT NULL,
        prediction TINYINT(1) NOT NULL,
        anomaly_score DOUBLE NULL,
        confidence_score DOUBLE NULL,
        traffic_type VARCHAR(50) NULL,
        is_anomaly TINYINT(1) NULL,
        risk_level VARCHAR(20) NULL,
        log_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_session_id (session_id),
        INDEX idx_model_type (model_type),
        INDEX idx_prediction (prediction),
        INDEX idx_log_time (log_time),
        FOREIGN KEY (session_id) REFERENCES network_traffic(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """

    cursor.execute(create_sql)
    logger.info("predictions table created successfully")

def verify_table_creation(cursor):
    """Verify that all tables were created successfully."""
    logger.info("Verifying table creation...")

    tables_to_check = [
        'network_traffic',
        'anomaly_results',
        'supervised_predictions',
        'summary_stats',
        'predictions'
    ]

    for table in tables_to_check:
        try:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            result = cursor.fetchone()
            if result:
                logger.info(f"✓ Table '{table}' exists")

                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"  - Current row count: {count}")
            else:
                logger.error(f"✗ Table '{table}' does not exist")
                return False
        except Exception as e:
            logger.error(f"Error checking table '{table}': {e}")
            return False

    return True

def main():
    """Main function to create all MySQL tables."""
    logger.info("Starting MySQL table creation process...")

    # CSV file path
    csv_path = "data/ASNM-NBPOv2.csv"

    # Establish database connection
    conn = None
    try:
        logger.info("Connecting to MySQL database...")
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            autocommit=False  # Use transactions
        )
        cursor = conn.cursor()

        # Get column types from CSV
        logger.info(f"Reading column types from {csv_path}...")
        column_types = get_csv_column_types(csv_path)

        # Create all tables within a transaction
        try:
            # Create tables
            create_network_traffic_table(cursor, column_types)
            create_anomaly_results_table(cursor)
            create_supervised_predictions_table(cursor)
            create_summary_stats_table(cursor)
            create_predictions_table(cursor)

            # Verify creation
            if verify_table_creation(cursor):
                # Commit transaction
                conn.commit()
                logger.info("✅ All tables created successfully!")
                logger.info("Database schema is ready for data ingestion.")
            else:
                raise Exception("Table verification failed")

        except Exception as e:
            # Rollback on error
            conn.rollback()
            logger.error(f"Error during table creation: {e}")
            raise

    except pymysql.Error as e:
        logger.error(f"MySQL error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
