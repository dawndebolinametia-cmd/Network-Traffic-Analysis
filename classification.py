import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from logging_utils import logger

def classify_traffic_types(data, target_column='traffic_type'):
    """
    Classifies network traffic types using Random Forest.

    Args:
        data (pd.DataFrame): Input data with features and target.
        target_column (str): Name of the target column.

    Returns:
        dict: Classification results including model and metrics.
    """
    try:
        # Prepare features
        feature_columns = ['source_port', 'destination_port', 'packet_size', 'duration', 'protocol_encoded']
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0])

        # Encode categorical variables
        if 'protocol' in data.columns:
            le = LabelEncoder()
            data['protocol_encoded'] = le.fit_transform(data['protocol'])
            X['protocol_encoded'] = data['protocol_encoded']

        # Encode target if it's categorical
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
        else:
            y_encoded = y
            target_encoder = None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            max_depth=10
        )
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Traffic classification completed. Accuracy: {accuracy:.4f}")

        return {
            'model': model,
            'scaler': scaler,
            'target_encoder': target_encoder,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }

    except Exception as e:
        logger.error(f"Error in traffic classification: {e}")
        return None

def predict_traffic_type(model_data, new_data):
    """
    Predicts traffic type for new data using trained model.

    Args:
        model_data (dict): Trained model data from classify_traffic_types.
        new_data (pd.DataFrame): New data to classify.

    Returns:
        pd.DataFrame: Data with predictions.
    """
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        target_encoder = model_data['target_encoder']

        feature_columns = ['source_port', 'destination_port', 'packet_size', 'duration', 'protocol_encoded']
        X_new = new_data[feature_columns].copy()

        # Handle missing values
        X_new = X_new.fillna(X_new.mean())

        # Encode protocol if needed
        if 'protocol' in new_data.columns and 'protocol_encoded' not in new_data.columns:
            # Assuming same encoding as training
            le = LabelEncoder()
            X_new['protocol_encoded'] = le.fit_transform(new_data['protocol'])

        # Scale features
        X_new_scaled = scaler.transform(X_new)

        # Make predictions
        predictions = model.predict(X_new_scaled)
        probabilities = model.predict_proba(X_new_scaled)

        # Decode predictions if needed
        if target_encoder:
            predictions_decoded = target_encoder.inverse_transform(predictions)
        else:
            predictions_decoded = predictions

        new_data = new_data.copy()
        new_data['predicted_traffic_type'] = predictions_decoded
        new_data['prediction_confidence'] = np.max(probabilities, axis=1)

        logger.info(f"Traffic type prediction completed for {len(new_data)} samples")
        return new_data

    except Exception as e:
        logger.error(f"Error in traffic type prediction: {e}")
        return None

def save_classification_model(model_data, filename='traffic_classifier.pkl'):
    """
    Saves the classification model and related objects.

    Args:
        model_data (dict): Model data to save.
        filename (str): Filename to save the model.
    """
    try:
        joblib.dump(model_data, filename)
        logger.info(f"Classification model saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving classification model: {e}")

def load_classification_model(filename='traffic_classifier.pkl'):
    """
    Loads the classification model and related objects.

    Args:
        filename (str): Filename to load the model from.

    Returns:
        dict: Loaded model data.
    """
    try:
        model_data = joblib.load(filename)
        logger.info(f"Classification model loaded from {filename}")
        return model_data
    except Exception as e:
        logger.error(f"Error loading classification model: {e}")
        return None

def get_traffic_type_distribution(data, traffic_type_column='traffic_type'):
    """
    Analyzes the distribution of traffic types in the data.

    Args:
        data (pd.DataFrame): Input data.
        traffic_type_column (str): Column containing traffic types.

    Returns:
        pd.Series: Distribution of traffic types.
    """
    try:
        distribution = data[traffic_type_column].value_counts()
        logger.info(f"Traffic type distribution:\n{distribution}")
        return distribution
    except Exception as e:
        logger.error(f"Error analyzing traffic type distribution: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    from data_ingestion import fetch_network_traffic_data

    data = fetch_network_traffic_data()
    if data is not None and not data.empty:
        # Add synthetic traffic_type column for demonstration
        import random
        traffic_types = ['HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP', 'SMTP']
        data['traffic_type'] = [random.choice(traffic_types) for _ in range(len(data))]

        # Classify traffic types
        results = classify_traffic_types(data)
        if results:
            print(f"Classification Accuracy: {results['accuracy']:.4f}")
            print("Feature Importance:", results['feature_importance'])

            # Save model
            save_classification_model(results)
    else:
        logger.error("No data available for traffic classification")
