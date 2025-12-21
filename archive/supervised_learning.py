import pandas as pd
import numpy as np
import joblib
import os
import pymysql
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from config import (
    DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT,
    NETWORK_TRAFFIC_TABLE, PREDICTIONS_TABLE, RANDOM_STATE, TEST_SIZE,
    MODEL_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_labeled_data():
    """Load network traffic data with anomaly labels."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )

    # Join network traffic with predictions to get labeled data
    query = f"""
    SELECT nt.*, p.is_anomaly, p.anomaly_score
    FROM {NETWORK_TRAFFIC_TABLE} nt
    LEFT JOIN {PREDICTIONS_TABLE} p ON nt.id = p.session_id
    WHERE p.model_type = 'isolation_forest'
    """

    df = pd.read_sql(query, conn)
    conn.close()

    logger.info(f"Loaded {len(df)} labeled samples for supervised learning")
    return df

def prepare_features_and_labels(df):
    """Prepare features and labels for supervised learning."""
    # Select features (same as anomaly detection)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['id', 'is_anomaly', 'anomaly_score']  # Exclude labels and IDs

    # Include timestamp-derived features if available
    if 'tsesstart' in df.columns:
        df['hour_of_day'] = pd.to_datetime(df['tsesstart']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['tsesstart']).dt.dayofweek
        numeric_cols = numeric_cols.union(['hour_of_day', 'day_of_week'])

    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df['is_anomaly'].fillna(0).astype(int)

    logger.info(f"Prepared {len(feature_cols)} features and {len(y)} labels")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y, feature_cols

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    logger.info("Training Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced classes
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance."""
    logger.info(f"Evaluating {model_name}...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUC might not be available if only one class in test set
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = None
        logger.warning("AUC could not be calculated (possibly only one class in test set)")

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }

    logger.info(f"{model_name} Metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    if auc is not None:
        logger.info(f"  AUC: {auc:.4f}")

    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    return metrics

def save_model_and_scaler(model, scaler, feature_cols, metrics):
    """Save trained model, scaler, and metadata."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'metrics': metrics,
        'model_type': 'random_forest'
    }

    joblib.dump(model_data, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

def save_supervised_predictions_to_db(session_ids, y_pred, y_pred_proba, model_type):
    """Save supervised model predictions to database."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Insert predictions (don't delete existing ones, keep both unsupervised and supervised)
    insert_sql = f"""
    INSERT INTO {PREDICTIONS_TABLE}
    (session_id, is_anomaly, prediction_label, model_type, confidence_score)
    VALUES (%s, %s, %s, %s, %s)
    """

    values = []
    for session_id, pred, proba in zip(session_ids, y_pred, y_pred_proba):
        prediction_label = 'anomaly' if pred == 1 else 'normal'
        confidence_score = float(proba) if pred == 1 else float(1 - proba)
        values.append((int(session_id), int(pred), prediction_label, model_type, confidence_score))

    cursor.executemany(insert_sql, values)
    conn.commit()

    logger.info(f"Inserted {len(values)} supervised predictions into {PREDICTIONS_TABLE}")

    cursor.close()
    conn.close()

def perform_supervised_learning():
    """Main function for supervised learning."""
    try:
        logger.info("Starting supervised learning...")

        # Load labeled data
        df = load_labeled_data()

        if df.empty or 'is_anomaly' not in df.columns:
            logger.warning("No labeled data available for supervised learning. Skipping supervised learning step.")
            return

        # Prepare features and labels
        X, y, feature_cols = prepare_features_and_labels(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = train_random_forest(X_train_scaled, y_train)

        # Evaluate model
        metrics = evaluate_model(model, X_test_scaled, y_test, "Random Forest")

        # Save model
        save_model_and_scaler(model, scaler, feature_cols, metrics)

        # Generate predictions for all data
        X_all_scaled = scaler.transform(X)
        y_pred_all = model.predict(X_all_scaled)
        y_pred_proba_all = model.predict_proba(X_all_scaled)[:, 1]

        # Save predictions to database
        save_supervised_predictions_to_db(df['id'], y_pred_all, y_pred_proba_all, 'random_forest')

        logger.info("Supervised learning completed successfully")

    except Exception as e:
        logger.error(f"Error in supervised learning: {e}")
        raise

if __name__ == "__main__":
    perform_supervised_learning()
