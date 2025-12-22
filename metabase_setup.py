#!/usr/bin/env python3
"""
Metabase Setup and Testing Script
This script helps set up and test Metabase integration with the network traffic analysis project.
"""

import requests
import json
import logging
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Metabase configuration - Update these with your Metabase instance details
METABASE_URL = "http://localhost:3000"  # Default Metabase URL
METABASE_USERNAME = "admin@metabase.com"  # Default admin email
METABASE_PASSWORD = "admin123"  # Default admin password

def test_metabase_connection():
    """Test connection to Metabase instance."""
    try:
        # Test if Metabase is running
        response = requests.get(f"{METABASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Metabase is running and accessible")
            return True
        else:
            logger.error(f"✗ Metabase health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Cannot connect to Metabase: {e}")
        logger.info("Make sure Metabase is running on the specified URL")
        return False

def authenticate_metabase():
    """Authenticate with Metabase and get session token."""
    try:
        auth_data = {
            "username": METABASE_USERNAME,
            "password": METABASE_PASSWORD
        }
        response = requests.post(f"{METABASE_URL}/api/session", json=auth_data, timeout=10)

        if response.status_code == 200:
            token = response.json().get('id')
            logger.info("✓ Successfully authenticated with Metabase")
            return token
        else:
            logger.error(f"✗ Metabase authentication failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Authentication request failed: {e}")
        return None

def test_mysql_connection_from_metabase(token):
    """Test if Metabase can connect to the MySQL database."""
    try:
        # Get list of databases
        headers = {'X-Metabase-Session': token}
        response = requests.get(f"{METABASE_URL}/api/database", headers=headers, timeout=10)

        if response.status_code == 200:
            databases = response.json()
            mysql_db = None
            for db in databases:
                if db.get('engine') == 'mysql' and DB_NAME in db.get('name', ''):
                    mysql_db = db
                    break

            if mysql_db:
                logger.info(f"✓ MySQL database '{mysql_db['name']}' found in Metabase")
                return mysql_db['id']
            else:
                logger.warning("⚠ MySQL database not found in Metabase")
                logger.info("You need to add the MySQL database in Metabase admin panel:")
                logger.info(f"  - Host: {DB_HOST}")
                logger.info(f"  - Port: {DB_PORT}")
                logger.info(f"  - Database: {DB_NAME}")
                logger.info(f"  - Username: {DB_USER}")
                logger.info(f"  - Password: {DB_PASSWORD}")
                return None
        else:
            logger.error(f"✗ Failed to get database list: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Database check request failed: {e}")
        return None

def sync_database_schema(token, database_id):
    """Trigger database schema sync in Metabase."""
    try:
        headers = {'X-Metabase-Session': token}
        response = requests.post(f"{METABASE_URL}/api/database/{database_id}/sync_schema", headers=headers, timeout=30)

        if response.status_code == 200:
            logger.info("✓ Database schema sync triggered successfully")
            return True
        else:
            logger.error(f"✗ Schema sync failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Schema sync request failed: {e}")
        return False

def check_tables_visibility(token, database_id):
    """Check if the required tables are visible in Metabase."""
    try:
        headers = {'X-Metabase-Session': token}
        response = requests.get(f"{METABASE_URL}/api/database/{database_id}/metadata", headers=headers, timeout=10)

        if response.status_code == 200:
            metadata = response.json()
            tables = metadata.get('tables', [])

            required_tables = ['network_traffic', 'predictions', 'traffic_summary', 'anomaly_analysis', 'session_stats']
            found_tables = []

            for table in tables:
                table_name = table.get('name', '').lower()
                if any(req_table in table_name for req_table in required_tables):
                    found_tables.append(table_name)

            logger.info(f"✓ Found tables in Metabase: {found_tables}")

            missing_tables = [t for t in required_tables if not any(t in ft for ft in found_tables)]
            if missing_tables:
                logger.warning(f"⚠ Missing tables: {missing_tables}")
                logger.info("Run the metabase_integration.py script to create the required views")
            else:
                logger.info("✓ All required tables/views are available")

            return len(found_tables) > 0
        else:
            logger.error(f"✗ Failed to get database metadata: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Metadata request failed: {e}")
        return False

def main():
    """Main setup and testing function."""
    print("METABASE SETUP AND TESTING")
    print("=" * 40)

    # Test Metabase connection
    if not test_metabase_connection():
        print("\n❌ Metabase is not accessible. Please:")
        print("1. Install and start Metabase")
        print("2. Update METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD in this script")
        return

    # Authenticate
    token = authenticate_metabase()
    if not token:
        print("\n❌ Cannot authenticate with Metabase. Please check credentials.")
        return

    # Test MySQL connection
    database_id = test_mysql_connection_from_metabase(token)
    if not database_id:
        print("\n❌ MySQL database not configured in Metabase. Please add it manually.")
        return

    # Sync schema
    print("\n🔄 Syncing database schema...")
    if sync_database_schema(token, database_id):
        print("⏳ Waiting a moment for sync to complete...")
        import time
        time.sleep(5)

    # Check tables
    if check_tables_visibility(token, database_id):
        print("\n✅ Metabase integration is ready!")
        print("\nNext steps:")
        print("1. Open Metabase in your browser")
        print("2. Create questions and dashboards using the available tables")
        print("3. The 'traffic_summary', 'anomaly_analysis', and 'session_stats' views are ready for visualization")
    else:
        print("\n❌ Tables not found. Run the main pipeline to create them.")

if __name__ == "__main__":
    main()
