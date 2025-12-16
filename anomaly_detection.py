import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from config import MODEL_PATH, RANDOM_STATE
from logging_utils import logger

def detect_anomalies_isolation_forest(data, contamination=0.1):
    """
    Detects anomalies using Isolation Forest algorithm.

    Args:
        data (pd.DataFrame): Input data for anomaly detection.
        contamination (float): Expected proportion of outliers.

    Returns:
        pd.DataFrame: Data with anomaly scores and labels.
    """
    try:
        # Select numerical features for anomaly detection
        # For web traffic data, use available numerical columns
        numerical_features = ['response_code', 'bytes_sent']
        X = data[numerical_features].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Isolation Forest
        model = IsolationForest(
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_estimators=100
        )

        # Fit the model
        model.fit(X_scaled)

        # Fit and predict
        anomaly_scores = model.decision_function(X_scaled)
        predictions = model.predict(X_scaled)

        # Convert predictions to labels (0 for normal, 1 for anomaly)
        data = data.copy()
        data['anomaly_score'] = anomaly_scores
        data['prediction'] = (predictions == -1).astype(int)  # 1 for anomaly, 0 for normal
        data['anomaly_label'] = data['prediction'].map({0: 'normal', 1: 'anomaly'})
        data['traffic_type'] = data['anomaly_label']

        logger.info(f"Anomaly detection completed. Detected {sum(predictions == -1)} anomalies out of {len(data)} samples")
        return data

    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        return None

def detect_anomalies_statistical(data, threshold=3):
    """
    Detects anomalies using statistical methods (Z-score).

    Args:
        data (pd.DataFrame): Input data.
        threshold (float): Z-score threshold for anomaly detection.

    Returns:
        pd.DataFrame: Data with anomaly labels.
    """
    try:
        numerical_features = ['packet_size', 'duration']
        data = data.copy()

        for feature in numerical_features:
            if feature in data.columns:
                mean = data[feature].mean()
                std = data[feature].std()
                z_scores = np.abs((data[feature] - mean) / std)
                data[f'{feature}_z_score'] = z_scores
                data[f'{feature}_anomaly'] = z_scores > threshold

        # Overall anomaly if any feature is anomalous
        data['statistical_anomaly'] = data[[f'{feature}_anomaly' for feature in numerical_features if f'{feature}_anomaly' in data.columns]].any(axis=1)

        anomaly_count = sum(data['statistical_anomaly'])
        logger.info(f"Statistical anomaly detection completed. Detected {anomaly_count} anomalies out of {len(data)} samples")
        return data

    except Exception as e:
        logger.error(f"Error in statistical anomaly detection: {e}")
        return None

def evaluate_anomaly_detection(true_labels, predicted_labels):
    """
    Evaluates anomaly detection performance.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Predicted labels.

    Returns:
        dict: Evaluation metrics.
    """
    try:
        # Convert to binary (1 for anomaly, 0 for normal)
        true_binary = [1 if label == 'anomaly' else 0 for label in true_labels]
        pred_binary = [1 if label == 'anomaly' else 0 for label in predicted_labels]

        report = classification_report(true_binary, pred_binary, output_dict=True)
        cm = confusion_matrix(true_binary, pred_binary)

        logger.info("Anomaly detection evaluation completed")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info(f"Classification Report:\n{classification_report(true_binary, pred_binary)}")

        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }

    except Exception as e:
        logger.error(f"Error in anomaly detection evaluation: {e}")
        return None

def save_anomaly_model(model, filename='anomaly_model.pkl'):
    """
    Saves the anomaly detection model.

    Args:
        model: Trained model.
        filename (str): Filename to save the model.
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Anomaly model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving anomaly model: {e}")

def load_anomaly_model(filename='anomaly_model.pkl'):
    """
    Loads the anomaly detection model.

    Args:
        filename (str): Filename to load the model from.

    Returns:
        model: Loaded model.
    """
    try:
        model = joblib.load(filename)
        logger.info(f"Anomaly model loaded from {filename}")
        return model
    except Exception as e:
        logger.error(f"Error loading anomaly model: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from data_ingestion import fetch_network_traffic_data

    data = fetch_network_traffic_data()
    if data is not None:
        # Isolation Forest detection
        data_with_anomalies = detect_anomalies_isolation_forest(data)

        # Statistical detection
        data_with_anomalies = detect_anomalies_statistical(data_with_anomalies)

        if 'label' in data_with_anomalies.columns:
            evaluation = evaluate_anomaly_detection(data_with_anomalies['label'], data_with_anomalies['anomaly_label'])
            print("Evaluation results:", evaluation)
    else:
        logger.error("No data available for anomaly detection")
