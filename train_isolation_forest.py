import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from config import NETWORK_TRAFFIC_CSV, RANDOM_STATE
from logging_utils import logger

def train_isolation_forest():
    """
    Train an Isolation Forest model for unsupervised anomaly detection.
    Uses features: packet_size and hour extracted from timestamp.
    """
    try:
        # Load data
        df = pd.read_csv(NETWORK_TRAFFIC_CSV)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour

        # Features for training
        features = ['packet_size', 'hour']
        X = df[features]

        # Train Isolation Forest (unsupervised)
        model = IsolationForest(n_estimators=100, contamination='auto', random_state=RANDOM_STATE)
        model.fit(X)

        # Save model
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'isolation_forest_model.pkl')
        joblib.dump(model, model_path)

        logger.info(f"Isolation Forest model trained and saved to {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error training Isolation Forest: {e}")
        return None

if __name__ == "__main__":
    train_isolation_forest()
