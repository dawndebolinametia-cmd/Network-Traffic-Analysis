import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

print("Generating summary statistics...")

# Get counts
network_count = int(pd.read_sql("SELECT COUNT(*) as count FROM network_traffic", engine).iloc[0]['count'])
anomaly_count = int(pd.read_sql("SELECT COUNT(*) as count FROM anomaly_results", engine).iloc[0]['count'])
supervised_count = int(pd.read_sql("SELECT COUNT(*) as count FROM supervised_predictions", engine).iloc[0]['count'])

# Calculate anomaly stats
if anomaly_count > 0:
    anomaly_count_val = int(pd.read_sql("SELECT COUNT(*) as count FROM anomaly_results WHERE is_anomaly = 1", engine).iloc[0]['count'])
    normal_count = int(anomaly_count - anomaly_count_val)
    anomaly_pct = float((anomaly_count_val / anomaly_count) * 100)
else:
    anomaly_count_val = 0
    normal_count = 0
    anomaly_pct = 0.0

# Create summary_stats table
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
    
    conn.execute(text("""
        INSERT INTO summary_stats (total_records, anomaly_count, normal_count, supervised_predictions, anomaly_percentage)
        VALUES (:total, :anomaly, :normal, :supervised, :pct)
    """), {
        'total': network_count,
        'anomaly': anomaly_count_val,
        'normal': normal_count,
        'supervised': supervised_count,
        'pct': anomaly_pct
    })
    conn.commit()

print("Summary statistics table created")
print(f"Total records: {network_count}")
print(f"Anomalies: {anomaly_count_val}")
print(f"Normal: {normal_count}")
print(f"Anomaly percentage: {anomaly_pct:.2f}%")
