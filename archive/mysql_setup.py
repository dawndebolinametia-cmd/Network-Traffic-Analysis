import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE

# Load CSV
df = pd.read_csv('data/ASNM-NBPOv2.csv', sep=';')
print(f"Loaded {len(df)} rows from CSV")

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Drop table if exists
with engine.connect() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {NETWORK_TRAFFIC_TABLE}"))
    conn.commit()
    print(f"Dropped table {NETWORK_TRAFFIC_TABLE} if existed")

# Create table and insert data
df.to_sql(NETWORK_TRAFFIC_TABLE, engine, if_exists='replace', index=False)
print(f"Created table {NETWORK_TRAFFIC_TABLE} and inserted data")

# Verify row count
with engine.connect() as conn:
    result = conn.execute(text(f"SELECT COUNT(*) FROM {NETWORK_TRAFFIC_TABLE}"))
    count = result.fetchone()[0]
    print(f"Table has {count} rows")
    if count == len(df):
        print("Row count matches CSV")
    else:
        print("Row count mismatch!")
        exit(1)

print("MySQL setup completed successfully")
