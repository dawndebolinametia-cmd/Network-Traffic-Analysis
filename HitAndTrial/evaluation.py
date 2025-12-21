import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from config import MODEL_PATH
from logging_utils import logger

def load_model():
    """
    Loads the trained model.
    """
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    else:
        logger.error("Model file not found.")
        return None

def make_predictions(model, data, feature_columns=None):
    """
    Makes predictions using the loaded model.
    """
    try:
        if feature_columns:
            X = data[feature_columns]
        else:
            X = data.select_dtypes(include=['int64', 'float64'])  # Assume numerical features
        predictions = model.predict(X)
        probabilities = getattr(model, 'predict_proba', lambda x: None)(X)
        logger.info("Predictions made successfully.")
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        return None, None

def evaluate_performance(y_true, y_pred):
    """
    Evaluates model performance.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

def save_evaluation_results(metrics, file_path='reports/evaluation_results.txt'):
    """
    Saves evaluation results to a file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 30 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(metrics['confusion_matrix']))
            f.write("\n\nClassification Report:\n")
            f.write(metrics['classification_report'])
        logger.info(f"Evaluation results saved to {file_path}.")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")

def predict_and_evaluate(data, target_column='label', feature_columns=None):
    """
    Loads model, makes predictions, evaluates, and saves results.
    """
    logger.info("Starting prediction and evaluation pipeline.")
    model = load_model()
    if model is None:
        return None

    if target_column not in data.columns:
        logger.error(f"Target column '{target_column}' not found in data.")
        return None

    X = data.drop(columns=[target_column]) if feature_columns is None else data[feature_columns]
    y_true = data[target_column]

    predictions, probabilities = make_predictions(model, X, feature_columns)
    if predictions is None:
        return None

    metrics = evaluate_performance(y_true, predictions)
    save_evaluation_results(metrics)

    # Add predictions to the data
    data_with_predictions = data.copy()
    data_with_predictions['prediction'] = predictions
    if probabilities is not None:
        data_with_predictions['prediction_proba'] = probabilities.max(axis=1) if probabilities.ndim > 1 else probabilities

    logger.info("Prediction and evaluation pipeline completed.")
    return {
        'metrics': metrics,
        'predictions': predictions,
        'probabilities': probabilities,
        'data_with_predictions': data_with_predictions
    }

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_data
    from data_ingestion import fetch_network_traffic_data
    data = fetch_network_traffic_data()
    if data is not None:
        processed_data, _, _ = preprocess_data(data)
        results = predict_and_evaluate(processed_data)
        if results:
            print("Evaluation completed. Check reports/evaluation_results.txt")
        else:
            print("Evaluation failed.")
    else:
        print("No data available for evaluation.")
