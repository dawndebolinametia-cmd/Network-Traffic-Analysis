from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Inspect anomaly_results table
with engine.connect() as conn:
    # Check if table exists
    result = conn.execute(text("SHOW TABLES LIKE 'anomaly_results'"))
    if not result.fetchone():
        print("anomaly_results table does not exist")
        exit(1)
    
    # Get count
    result = conn.execute(text("SELECT COUNT(*) FROM anomaly_results"))
    count = result.fetchone()[0]
    print(f"Anomaly results table has {count} rows")
    
    # Get min max scores
    result = conn.execute(text("SELECT MIN(anomaly_score), MAX(anomaly_score) FROM anomaly_results"))
    min_score, max_score = result.fetchone()
    print(f"Anomaly scores range: {min_score} to {max_score}")
    
    # Check for missing entries
    result = conn.execute(text("SELECT COUNT(*) FROM anomaly_results WHERE anomaly_score IS NULL"))
    null_count = result.fetchone()[0]
    print(f"Missing anomaly scores: {null_count}")
    
    # Sample rows
    result = conn.execute(text("SELECT * FROM anomaly_results LIMIT 5"))
    rows = result.fetchall()
    print("Sample anomaly results:")
    for row in rows:
        print(row)

# Check for labeled data
with engine.connect() as conn:
    # Check label column
    result = conn.execute(text(f"SELECT COUNT(*) FROM {NETWORK_TRAFFIC_TABLE} WHERE label IS NOT NULL"))
    labeled_count = result.fetchone()[0]
    print(f"Labeled samples: {labeled_count}")
    
    if labeled_count > 0:
        print("Labeled data exists, proceeding to supervised learning")
    else:
        print("No labeled data, skipping supervised learning")

print("Inspection completed")
