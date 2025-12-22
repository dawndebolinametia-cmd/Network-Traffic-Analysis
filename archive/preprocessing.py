import pandas as pd
import hashlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from logging_utils import logger

def clean_data(df):
    """
    Cleans the data by removing duplicates and handling missing values.
    """
    logger.info("Starting data cleaning.")
    initial_shape = df.shape
    df = df.drop_duplicates()
    df = df.dropna()  # Simple drop for missing values
    logger.info(f"Data cleaned: {initial_shape} -> {df.shape}")
    return df

def anonymize_data(df, ip_columns=['source_ip']):
    """
    Anonymizes IP addresses by hashing them.
    """
    logger.info("Starting data anonymization.")
    for col in ip_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
            logger.info(f"Anonymized column: {col}")
    return df

def create_label(df):
    """
    Creates a label column for supervised learning based on response_code.
    Assumes response_code >= 400 is anomaly (1), else normal (0).
    """
    if 'response_code' in df.columns:
        df['label'] = df['response_code'].apply(lambda x: 1 if x >= 400 else 0)
        logger.info("Label column created based on response_code.")
    else:
        logger.warning("response_code column not found, cannot create label.")
    return df

def feature_engineering(df):
    """
    Performs feature engineering: converts timestamps, extracts features, encodes categoricals, scales numericals.
    """
    logger.info("Starting feature engineering.")
    encoders = {}
    scaler = None

    # Convert log_time to datetime 
    if 'log_time' in df.columns:
        df['log_time'] = pd.to_datetime(df['log_time'])
        df['hour'] = df['log_time'].dt.hour
        df['day_of_week'] = df['log_time'].dt.dayofweek
        df['month'] = df['log_time'].dt.month
        df['day'] = df['log_time'].dt.day
        # Drop original log_time if not needed
        df = df.drop(columns=['log_time'])

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    # Scale numerical variables
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'label']
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    logger.info("Feature engineering completed.")
    return df, encoders, scaler

def preprocess_data(df, save_preprocessors=True):
    """
    Full preprocessing pipeline: clean, anonymize, create label, feature engineering.
    """
    logger.info("Starting full preprocessing pipeline.")
    df = clean_data(df)
    df = anonymize_data(df)
    df = create_label(df)
    df, encoders, scaler = feature_engineering(df)
    if save_preprocessors:
        save_encoders_scaler(encoders, scaler)
    logger.info("Preprocessing pipeline completed.")
    return df, encoders, scaler

def save_encoders_scaler(encoders, scaler, path='models/'):
    """
    Saves encoders and scaler to disk.
    """
    import joblib
    import os
    os.makedirs(path, exist_ok=True)
    joblib.dump(encoders, os.path.join(path, 'encoders.pkl'))
    joblib.dump(scaler, os.path.join(path, 'scaler.pkl'))
    logger.info("Encoders and scaler saved.")

def load_encoders_scaler(path='models/'):
    """
    Loads encoders and scaler from disk.
    """
    import joblib
    import os
    encoders = joblib.load(os.path.join(path, 'encoders.pkl')) if os.path.exists(os.path.join(path, 'encoders.pkl')) else {}
    scaler = joblib.load(os.path.join(path, 'scaler.pkl')) if os.path.exists(os.path.join(path, 'scaler.pkl')) else None
    logger.info("Encoders and scaler loaded.")
    return encoders, scaler

if __name__ == "__main__":
    # Example
    from data_ingestion import fetch_network_traffic_data
    data = fetch_network_traffic_data()
    if not data.empty:
        processed_data, _, _ = preprocess_data(data)
        print(processed_data.head())
    else:
        print("No data available for preprocessing.")
