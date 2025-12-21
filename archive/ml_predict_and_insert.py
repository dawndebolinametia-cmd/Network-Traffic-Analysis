import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import mysql.connector
import os
from datetime import datetime, timedelta

# --------------------------
# 1. Generate synthetic data
# --------------------------
NUM_ROWS = 100

# timestamps starting from now, one per hour
timestamps = [datetime.now() - timedelta(hours=i) for i in range(NUM_ROWS)][::-1]

df = pd.DataFrame({
    "log_time": timestamps,
    "anomaly_score": np.random.rand(NUM_ROWS),  # 0 to 1
    "traffic_type": np.random.choice(["normal", "suspicious"], size=NUM_ROWS, p=[0.8, 0.2])
})

# --------------------------
# 2. ML Model: Isolation Forest
# --------------------------
model = IsolationForest(contamination=0.2, random_state=42)
df['prediction'] = model.fit_predict(df[['anomaly_score']])
df['prediction'] = df['prediction'].apply(lambda x: 1 if x == -1 else 0)  # -1 -> anomaly

# --------------------------
# 3. Save CSV (optional)
# --------------------------
csv_path = 'reports/predictions.csv'
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df.to_csv(csv_path, index=False)
print(f"CSV saved at {csv_path}")

# --------------------------
# 4. Insert into MySQL
# --------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Anime#210305",
    database="analytics_data_fresh",
    port=3306
)
cursor = conn.cursor()

# clear previous synthetic predictions (optional)
cursor.execute("DELETE FROM prediction_anomaly")
conn.commit()

# insert new predictions
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO prediction_anomaly (log_time, prediction, anomaly_score, traffic_type)
        VALUES (%s, %s, %s, %s)
    """, (row['log_time'], row['prediction'], row['anomaly_score'], row['traffic_type']))

conn.commit()
cursor.close()
conn.close()

print(f"{len(df)} ML predictions inserted into MySQL successfully ")

input("Press Enter to exit…")
