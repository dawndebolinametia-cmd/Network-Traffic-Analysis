import pandas as pd
from sqlalchemy import create_engine
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT
import joblib
import os

# Create engine
engine = create_engine(f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Export anomaly_results
print("Exporting anomaly_results...")
df_anomaly = pd.read_sql("SELECT * FROM anomaly_results", engine)
df_anomaly.to_csv('anomaly_results.csv', index=False)
print(f"Exported {len(df_anomaly)} rows to anomaly_results.csv")
print("First 10 rows of anomaly_results:")
print(df_anomaly.head(10))

# Save Isolation Forest model (assuming it's already trained and saved)
if os.path.exists('models/isolation_forest_model.pkl'):
    print("Isolation Forest model already saved as models/isolation_forest_model.pkl")
else:
    print("Warning: Isolation Forest model not found")

# Export supervised_predictions
print("\nExporting supervised_predictions...")
df_supervised = pd.read_sql("SELECT * FROM supervised_predictions", engine)
df_supervised.to_csv('supervised_predictions.csv', index=False)
print(f"Exported {len(df_supervised)} rows to supervised_predictions.csv")
print("First 10 rows of supervised_predictions:")
print(df_supervised.head(10))

# Save Random Forest model (assuming it's already trained and saved)
if os.path.exists('models/random_forest_model.pkl'):
    print("Random Forest model already saved as models/random_forest_model.pkl")
else:
    print("Warning: Random Forest model not found")

print("\nExport completed. Check VS Code for files.")
