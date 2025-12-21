import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Load CSV
print("Loading CSV data...")
csv_path = 'data/ASNM-NBPOv2.csv'
df = pd.read_csv(csv_path, sep=';')
print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Create table
print("Creating MySQL table...")
with engine.connect() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {NETWORK_TRAFFIC_TABLE}"))
    columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            columns.append(f"`{col}` TEXT")
        elif df[col].dtype == 'int64':
            columns.append(f"`{col}` BIGINT")
        elif df[col].dtype == 'float64':
            columns.append(f"`{col}` FLOAT")
        else:
            columns.append(f"`{col}` TEXT")
    columns_str = ", ".join(columns)
    create_query = f"CREATE TABLE {NETWORK_TRAFFIC_TABLE} (id BIGINT AUTO_INCREMENT PRIMARY KEY, {columns_str})"
    conn.execute(text(create_query))
    conn.commit()

# Insert data
print("Inserting data into MySQL...")
df.to_sql(NETWORK_TRAFFIC_TABLE, engine, if_exists='append', index=False)
print(f"Inserted {len(df)} rows into {NETWORK_TRAFFIC_TABLE}")

print("Data ingestion completed")
