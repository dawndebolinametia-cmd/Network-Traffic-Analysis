import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
from data_ingestion import fetch_network_traffic_data
from logging_utils import logger

def preprocess_data(df):
    """
    Preprocess the data for training.

    Args:
        df (pd.DataFrame): Raw data

    Returns:
        tuple: (X, y) features and labels
    """
    logger.info("Preprocessing data for training")

    # Encode categorical variables
    le_protocol = LabelEncoder()
    df['protocol_encoded'] = le_protocol.fit_transform(df['protocol'])

    # Select features (exclude timestamp, IPs, and label)
    features = ['source_port', 'destination_port', 'protocol_encoded', 'packet_size', 'duration']
    X = df[features]

    # Encode labels: normal=0, anomaly=1
    y = df['label'].map({'normal': 0, 'anomaly': 1})

    logger.info(f"Selected features: {features}")
    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    return X, y

def train_model(X_train, y_train):
    """
    Train the RandomForestClassifier.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        RandomForestClassifier: Trained model
    """
    logger.info("Training RandomForestClassifier")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    logger.info("Model training completed")

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    logger.info("Evaluating model performance")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['normal', 'anomaly']))

    return accuracy, f1

def save_model(model, filename='network_model.pkl'):
    """
    Save the trained model to disk.

    Args:
        model: Trained model
        filename (str): Output filename
    """
    try:
        joblib.dump(model, filename)
        logger.info(f"Model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")

def main():
    """
    Main training pipeline.
    """
    logger.info("Starting ML training pipeline")

    # Load data
    df = fetch_network_traffic_data()
    if df.empty:
        logger.error("No data available for training")
        return

    # Preprocess data
    X, y = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    # Train model (check again)
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy, f1 = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    logger.info("ML training pipeline completed successfully")

if __name__ == "__main__":
    main()
