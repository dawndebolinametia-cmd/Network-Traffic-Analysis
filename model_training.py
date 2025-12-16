import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from config import MODEL_PATH, CACHE_PATH, RANDOM_STATE, TEST_SIZE
from logging_utils import logger

def load_existing_model():
    """
    Loads an existing model if available.
    """
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Existing model loaded.")
            return model
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}")
    return None

def combine_data(new_data, existing_model=None):
    """
    Combines new data with existing data for incremental learning.
    For simplicity, returns new_data (incremental learning placeholder).
    """
    # Placeholder: In a real scenario, load cached old data and combine
    logger.info("Combining data for incremental learning (placeholder).")
    return new_data

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model.
    """
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    logger.info("Random Forest model trained.")
    return model

def train_xgboost(X_train, y_train):
    """
    Trains an XGBoost model.
    """
    model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    logger.info("XGBoost model trained.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    logger.info(f"Model evaluation - Accuracy: {accuracy:.4f}")
    logger.info(f"Classification Report:\n{report}")
    return accuracy, report

def save_model(model, path=MODEL_PATH):
    """
    Saves the trained model to disk.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

def train_model(data, target_column='label', model_type='random_forest'):
    """
    Orchestrates the model training process.
    """
    logger.info("Starting model training.")
    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found.")
        return None

    # Prepare features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Load existing model for incremental learning (placeholder)
    existing_model = load_existing_model()
    combined_data = combine_data(data, existing_model)

    # Train model
    if model_type == 'random_forest':
        model = train_random_forest(X_train, y_train)
    elif model_type == 'xgboost':
        model = train_xgboost(X_train, y_train)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return None

    # Evaluate
    accuracy, report = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    logger.info("Model training completed.")
    return model

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_data
    from data_ingestion import fetch_network_traffic_data
    data = fetch_network_traffic_data()
    if data is not None:
        processed_data, _, _ = preprocess_data(data)
        model = train_model(processed_data)
        if model:
            print("Model training completed.")
        else:
            print("Model training failed.")
    else:
        print("No data available for training.")
