import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sqlalchemy import create_engine, text
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE
import joblib
import os

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Load data
print("Loading data from MySQL...")
df = pd.read_sql(f"SELECT * FROM {NETWORK_TRAFFIC_TABLE}", engine)
print(f"Loaded {len(df)} rows")

# Select numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['id']  # exclude id
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
print(f"Using {len(numeric_cols)} numeric features: {numeric_cols[:10]}...")

if not numeric_cols:
    print("No numeric features found")
    exit(1)

# Prepare data
X = df[numeric_cols].fillna(df[numeric_cols].mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Unsupervised Anomaly Detection
print("Training Isolation Forest...")
model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
model.fit(X_scaled)
anomaly_scores = model.decision_function(X_scaled)
predictions = model.predict(X_scaled)
predictions = [1 if p == -1 else 0 for p in predictions]  # 1 for anomaly, 0 for normal

# Create anomaly_results table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS anomaly_results"))
    conn.execute(text("""
        CREATE TABLE anomaly_results (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            original_id BIGINT,
            anomaly_score FLOAT,
            is_anomaly TINYINT(1)
        )
    """))
    conn.commit()

# Insert results
anomaly_df = pd.DataFrame({
    'original_id': df['id'],
    'anomaly_score': anomaly_scores,
    'is_anomaly': predictions
})
anomaly_df.to_sql('anomaly_results', engine, if_exists='append', index=False)
print(f"Inserted {len(anomaly_df)} anomaly results")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/isolation_forest_model.pkl')
joblib.dump(scaler, 'models/isolation_forest_scaler.pkl')
print("Saved Isolation Forest model")

# Check for labeled data
label_col = 'label'
if label_col in df.columns and df[label_col].notna().sum() > 0:
    print("Labeled data found, training supervised model...")
    y = df[label_col].fillna(0).astype(int)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)
    
    # Predict
    supervised_predictions = clf.predict(X_scaled)
    
    # Create supervised_predictions table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS supervised_predictions"))
        conn.execute(text("""
            CREATE TABLE supervised_predictions (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                original_id BIGINT,
                predicted_label INT
            )
        """))
        conn.commit()
    
    # Insert results
    supervised_df = pd.DataFrame({
        'original_id': df['id'],
        'predicted_label': supervised_predictions
    })
    supervised_df.to_sql('supervised_predictions', engine, if_exists='append', index=False)
    print(f"Inserted {len(supervised_df)} supervised predictions")
    
    # Save model
    joblib.dump(clf, 'models/random_forest_model.pkl')
    print("Saved Random Forest model")
else:
    print("No labeled data, skipping supervised learning")

print("Anomaly detection completed")
