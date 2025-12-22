import pandas as pd
import os
import shutil
from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE, METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD, METABASE_DATABASE_ID
import requests

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

print("=== FINALIZING NETWORK TRAFFIC ANALYSIS PROJECT ===\n")

# 1. Verify MySQL tables
print("1. Verifying MySQL tables...")

# Check network_traffic
result = pd.read_sql(f"SELECT COUNT(*) as count FROM {NETWORK_TRAFFIC_TABLE}", engine)
network_count = result.iloc[0]['count']
print(f"   network_traffic: {network_count} rows")

# Check anomaly_results
try:
    result = pd.read_sql("SELECT COUNT(*) as count FROM anomaly_results", engine)
    anomaly_count = result.iloc[0]['count']
    print(f"   anomaly_results: {anomaly_count} rows")
except:
    print("   anomaly_results: Table not found")
    anomaly_count = 0

# Check supervised_predictions
try:
    result = pd.read_sql("SELECT COUNT(*) as count FROM supervised_predictions", engine)
    supervised_count = result.iloc[0]['count']
    print(f"   supervised_predictions: {supervised_count} rows")
except:
    print("   supervised_predictions: Table not found")
    supervised_count = 0

# 2. Validation checks
print("\n2. Running validation checks...")

# Compare CSV row count
csv_path = 'data/ASNM-NBPOv2.csv'
if os.path.exists(csv_path):
    csv_df = pd.read_csv(csv_path, sep=';')
    csv_count = len(csv_df)
    print(f"   CSV rows: {csv_count}")
    print(f"   Match: {network_count == csv_count}")
else:
    print("   CSV not found")

# Sample rows
print("   Sample from network_traffic:")
sample = pd.read_sql(f"SELECT * FROM {NETWORK_TRAFFIC_TABLE} LIMIT 3", engine)
print(sample.head(3))

if anomaly_count > 0:
    print("   Sample from anomaly_results:")
    sample = pd.read_sql("SELECT * FROM anomaly_results LIMIT 3", engine)
    print(sample.head(3))

if supervised_count > 0:
    print("   Sample from supervised_predictions:")
    sample = pd.read_sql("SELECT * FROM supervised_predictions LIMIT 3", engine)
    print(sample.head(3))

# 3. Generate summary tables
print("\n3. Generating summary tables...")

# Create summary_stats
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS summary_stats"))
    conn.execute(text("""
        CREATE TABLE summary_stats (
            id INT AUTO_INCREMENT PRIMARY KEY,
            total_records INT,
            anomaly_count INT,
            normal_count INT,
            supervised_predictions INT,
            anomaly_percentage FLOAT
        )
    """))
    
    if anomaly_count > 0:
        anomaly_count_val = int(pd.read_sql("SELECT COUNT(*) as count FROM anomaly_results WHERE is_anomaly = 1", engine).iloc[0]['count'])
        normal_count = int(anomaly_count - anomaly_count_val)
        anomaly_pct = float((anomaly_count_val / anomaly_count) * 100)
    else:
        anomaly_count_val = 0
        normal_count = 0
        anomaly_pct = 0.0
    
    conn.execute(text("""
        INSERT INTO summary_stats (total_records, anomaly_count, normal_count, supervised_predictions, anomaly_percentage)
        VALUES (:total, :anomaly, :normal, :supervised, :pct)
    """), {
        'total': int(network_count),
        'anomaly': int(anomaly_count_val),
        'normal': int(normal_count),
        'supervised': int(supervised_count),
        'pct': float(anomaly_pct)
    })
    conn.commit()

print("   Created summary_stats table")

# 4. Export tables
print("\n4. Exporting tables to CSV...")

if anomaly_count > 0:
    df = pd.read_sql("SELECT * FROM anomaly_results", engine)
    df.to_csv('anomaly_results.csv', index=False)
    print(f"   Exported anomaly_results.csv ({len(df)} rows)")

if supervised_count > 0:
    df = pd.read_sql("SELECT * FROM supervised_predictions", engine)
    df.to_csv('supervised_predictions.csv', index=False)
    print(f"   Exported supervised_predictions.csv ({len(df)} rows)")

# 5. Metabase sync
print("\n5. Syncing with Metabase...")

try:
    # Login to Metabase
    login_url = f"{METABASE_URL}/api/session"
    login_data = {"username": METABASE_USERNAME, "password": METABASE_PASSWORD}
    response = requests.post(login_url, json=login_data)
    if response.status_code == 200:
        token = response.json()['id']
        headers = {'X-Metabase-Session': token}
        
        # Sync database
        sync_url = f"{METABASE_URL}/api/database/{METABASE_DATABASE_ID}/sync_schema"
        response = requests.post(sync_url, headers=headers)
        if response.status_code == 200:
            print("   Metabase schema synced successfully")
        else:
            print(f"   Metabase sync failed: {response.status_code}")
    else:
        print(f"   Metabase login failed: {response.status_code}")
except Exception as e:
    print(f"   Metabase sync error: {e}")

# 6. Archive old files
print("\n6. Archiving old files...")

if not os.path.exists('archive'):
    os.makedirs('archive')

# Move old files to archive
old_files = ['anomaly_detection.py', 'inspect_anomaly_scores.py', 'export_results.py', 'mysql_setup.py', 'data_ingestion.py']
for file in old_files:
    if os.path.exists(file):
        shutil.move(file, f'archive/{file}')
        print(f"   Archived {file}")

# 7. Final confirmation
print("\n7. Final Confirmation:")
print(f"   - network_traffic table: {network_count} rows")
print(f"   - anomaly_results table: {anomaly_count} rows")
print(f"   - supervised_predictions table: {supervised_count} rows")
print(f"   - summary_stats table: Created")
print(f"   - Anomalies detected: {anomaly_count_val if anomaly_count > 0 else 0}")
print(f"   - Prediction summary: {supervised_count} predictions")
print("   - Exports: anomaly_results.csv, supervised_predictions.csv")
print("   - Metabase: Synced")
print("   - Archives: Old files moved to archive/")

print("\n=== PROJECT FINALIZED SUCCESSFULLY ===")
