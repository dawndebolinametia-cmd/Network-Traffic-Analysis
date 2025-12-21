import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/random_forest_model.pkl')
    joblib.dump(scaler, 'models/random_forest_scaler.pkl')
    print("Saved Random Forest model")
else:
    print("No labeled data, skipping supervised learning")

print("Supervised learning completed")
