import requests
import json
from config import METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD, METABASE_DATABASE_ID

def get_metabase_session():
    """Authenticate with Metabase and get session."""
    login_url = f"{METABASE_URL}/api/session"
    payload = {
        "username": METABASE_USERNAME,
        "password": METABASE_PASSWORD
    }
    response = requests.post(login_url, json=payload)
    if response.status_code == 200:
        return response.json()['id']
    else:
        raise Exception(f"Failed to authenticate with Metabase: {response.text}")

def sync_metabase_database(session_id, database_id):
    """Trigger database sync in Metabase."""
    sync_url = f"{METABASE_URL}/api/database/{database_id}/sync"
    headers = {
        "X-Metabase-Session": session_id
    }
    response = requests.post(sync_url, headers=headers)
    if response.status_code == 200:
        print("Metabase database sync triggered successfully.")
    else:
        print(f"Failed to sync Metabase database: {response.text}")

def main():
    try:
        session_id = get_metabase_session()
        sync_metabase_database(session_id, METABASE_DATABASE_ID)
    except Exception as e:
        print(f"Error syncing Metabase: {e}")

if __name__ == "__main__":
    main()
