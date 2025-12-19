import os
import sys
import argparse
import logging
import pymysql
from datetime import datetime
from config import LOG_FILE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_data_ingestion(csv_path):
    """Step 1: Data Ingestion & Preprocessing."""
    logger.info("=== STEP 1: Data Ingestion & Preprocessing ===")
    try:
        from csv_loader import load_and_preprocess_csv
        df = load_and_preprocess_csv(csv_path)
        logger.info(f"Successfully loaded and preprocessed {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

def run_database_setup(df):
    """Step 2: Database Setup."""
    logger.info("=== STEP 2: Database Setup ===")
    try:
        from create_table import setup_database
        setup_database(df)
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

def run_data_insertion(df):
    """Step 3: Insert Preprocessed Data."""
    logger.info("=== STEP 3: Insert Preprocessed Data ===")
    try:
        from insert_network_traffic import insert_network_traffic_data
        insert_network_traffic_data(df)
        logger.info("Data insertion completed successfully")
    except Exception as e:
        logger.error(f"Data insertion failed: {e}")
        raise

def run_anomaly_detection():
    """Step 4: Unsupervised Anomaly Detection."""
    logger.info("=== STEP 4: Unsupervised Anomaly Detection ===")
    try:
        from anomaly_detection import detect_anomalies_isolation_forest, insert_predictions
        import pandas as pd
        from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE

        # Load data from database
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT
        )
        df = pd.read_sql(f"SELECT * FROM {NETWORK_TRAFFIC_TABLE}", conn)
        conn.close()

        if df.empty:
            logger.warning("No data available for anomaly detection")
            return

        # Run anomaly detection
        df_with_predictions = detect_anomalies_isolation_forest(df)

        if df_with_predictions is not None:
            # Count anomalies
            anomaly_count = df_with_predictions['prediction'].sum()
            logger.info(f"Detected {anomaly_count} anomalies out of {len(df_with_predictions)} samples")

            # Insert predictions
            insert_predictions(df_with_predictions, df['id'])
            logger.info("Anomaly detection completed successfully")
        else:
            logger.error("Anomaly detection failed")

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise

def run_supervised_learning():
    """Step 5: Supervised Learning Preparation."""
    logger.info("=== STEP 5: Supervised Learning Preparation ===")
    try:
        from supervised_learning import perform_supervised_learning
        perform_supervised_learning()
        logger.info("Supervised learning completed successfully")
    except Exception as e:
        logger.error(f"Supervised learning failed: {e}")
        raise

def run_metabase_integration():
    """Step 6: Integration with Metabase."""
    logger.info("=== STEP 6: Integration with Metabase ===")
    try:
        from metabase_integration import create_traffic_summary_view, create_anomaly_analysis_view, create_session_stats_view
        create_traffic_summary_view()
        create_anomaly_analysis_view()
        create_session_stats_view()
        logger.info("Metabase integration completed successfully")
    except Exception as e:
        logger.error(f"Metabase integration failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Network Traffic Analysis Pipeline')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    args = parser.parse_args()

    csv_path = args.csv

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    try:
        # Step 1: Data Ingestion & Preprocessing
        df = run_data_ingestion(csv_path)

        # Step 2: Database Setup
        run_database_setup(df)

        # Step 3: Insert Preprocessed Data
        run_data_insertion(df)

        # Step 4: Unsupervised Anomaly Detection
        run_anomaly_detection()

        # Step 5: Supervised Learning Preparation
        run_supervised_learning()

        # Step 6: Integration with Metabase
        run_metabase_integration()

        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
