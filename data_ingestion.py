import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE
import sys

# Create engine
print("Establishing MySQL connection...")
try:
    engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("MySQL connection successful")
except Exception as e:
    print(f"ERROR: Failed to connect to MySQL: {e}")
    sys.exit(1)

# Load CSV
print("Loading CSV data...")
csv_path = 'data/ASNM-NBPOv2.csv'
try:
    df = pd.read_csv(csv_path, sep=';')
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print("Column dtypes:")
    print(df.dtypes)
    print(f"First 5 rows preview:")
    print(df.head())
except Exception as e:
    print(f"ERROR: Failed to load CSV: {e}")
    sys.exit(1)

# Create table
print("Creating MySQL table...")
with engine.connect() as conn:
    # Disable foreign key checks to allow dropping table with dependencies
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
    conn.execute(text(f"DROP TABLE IF EXISTS {NETWORK_TRAFFIC_TABLE}"))
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    columns = []
    has_id = 'id' in df.columns
    for col in df.columns:
        if col == 'id' and has_id:
            columns.append(f"`{col}` BIGINT PRIMARY KEY")
        elif df[col].dtype == 'object':
            columns.append(f"`{col}` TEXT")
        elif df[col].dtype == 'int64':
            columns.append(f"`{col}` BIGINT")
        elif df[col].dtype == 'float64':
            columns.append(f"`{col}` DOUBLE")
        else:
            columns.append(f"`{col}` TEXT")
    columns_str = ", ".join(columns)
    if not has_id:
        columns_str = f"id BIGINT AUTO_INCREMENT PRIMARY KEY, {columns_str}"
    create_query = f"CREATE TABLE {NETWORK_TRAFFIC_TABLE} ({columns_str})"
    conn.execute(text(create_query))
    conn.commit()

# Insert data in chunks
print("Inserting data into MySQL in chunks...")
chunksize = 1000  # Adjust as needed
total_inserted = 0
chunk_num = 0

for start in range(0, len(df), chunksize):
    chunk_num += 1
    end = min(start + chunksize, len(df))
    chunk_df = df.iloc[start:end]
    print(f"Inserting chunk {chunk_num}: rows {start+1}-{end} ({len(chunk_df)} rows)")

    conn = engine.connect()
    try:
        # Start transaction
        trans = conn.begin()
        chunk_df.to_sql(NETWORK_TRAFFIC_TABLE, conn, if_exists='append', index=False)
        trans.commit()
        total_inserted += len(chunk_df)
        print(f"Successfully inserted chunk {chunk_num}")
    except Exception as e:
        print(f"ERROR: Failed to insert chunk {chunk_num} (rows {start+1}-{end}): {e}")
        try:
            trans.rollback()
        except:
            pass
        conn.close()
        print("Transaction rolled back and connection closed")
        sys.exit(1)
    finally:
        conn.close()

print(f"Inserted {total_inserted} rows into {NETWORK_TRAFFIC_TABLE}")

# Verify row count
print("Verifying row count in MySQL...")
try:
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {NETWORK_TRAFFIC_TABLE}"))
        mysql_row_count = result.fetchone()[0]
    if mysql_row_count == len(df):
        print(f"SUCCESS: Row count verification passed - {mysql_row_count} rows in MySQL")
    else:
        print(f"ERROR: Row count mismatch - CSV has {len(df)} rows, MySQL has {mysql_row_count} rows")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: Failed to verify row count: {e}")
    sys.exit(1)

print("Data ingestion completed successfully")
