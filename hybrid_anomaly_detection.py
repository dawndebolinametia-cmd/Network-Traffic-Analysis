"""
Unsupervised Anomaly Detection System for Hourly Network Traffic

This module implements a single unsupervised anomaly detection approach for network traffic,
using Isolation Forest on hourly aggregated data from network_traffic.csv.

Structure:
1. Data Aggregation: Aggregate raw network traffic data into hourly summaries
2. Unsupervised Detection: Apply Isolation Forest to identify anomalous hours
3. Output: Anomaly scores and labels for each hour

Features:
- Hourly aggregation of packet counts, average packet sizes, unique IPs, etc.
- Isolation Forest for density-based anomaly detection
- No labels required - purely unsupervised
"""

import pandas as pd
import numpy as np
from anomaly_detection import detect_anomalies_isolation_forest
from logging_utils import logger

def aggregate_hourly_traffic(network_traffic_path='network_traffic.csv'):
    """
    Aggregates raw network traffic data into hourly summaries.

    Args:
        network_traffic_path (str): Path to network_traffic.csv

    Returns:
        pd.DataFrame: Hourly aggregated traffic data
    """
    try:
        # Load data
        data = pd.read_csv(network_traffic_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Aggregate to hourly
        hourly_data = data.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
            'packet_size': ['count', 'mean', 'std', 'min', 'max'],
            'source_ip': 'nunique',
            'dest_ip': 'nunique'
        }).reset_index()

        # Flatten column names
        hourly_data.columns = ['timestamp', 'packet_count', 'avg_packet_size', 'std_packet_size',
                              'min_packet_size', 'max_packet_size', 'unique_source_ips', 'unique_dest_ips']

        # Fill NaN values
        hourly_data = hourly_data.fillna(0)

        logger.info(f"Aggregated data into {len(hourly_data)} hourly records")
        return hourly_data

    except Exception as e:
        logger.error(f"Error aggregating hourly traffic: {e}")
        return None

def unsupervised_anomaly_detection_hourly(network_traffic_path='network_traffic.csv', contamination=0.1):
    """
    Unsupervised Anomaly Detection on Hourly Network Traffic Data

    Purpose: Identify anomalous hours in network traffic using Isolation Forest
    without requiring any labeled data.

    Method: Isolation Forest (unsupervised, density-based anomaly detection)
    - Identifies anomalies as data points that are isolated in the feature space
    - Applied to hourly aggregated features: packet counts, sizes, unique IPs
    - No assumption of data distribution
    - Effective for high-dimensional data

    Args:
        network_traffic_path (str): Path to network_traffic.csv
        contamination (float): Expected proportion of anomalous hours

    Returns:
        pd.DataFrame: Hourly data with anomaly scores and labels
    """
    logger.info("Starting Unsupervised Anomaly Detection on Hourly Data...")

    try:
        # Aggregate data to hourly
        hourly_data = aggregate_hourly_traffic(network_traffic_path)
        if hourly_data is None:
            return None

        # Apply Isolation Forest for anomaly detection
        data_with_anomalies = detect_anomalies_isolation_forest(hourly_data, contamination=contamination)

        if data_with_anomalies is not None:
            anomalies_detected = sum(data_with_anomalies['prediction'])
            logger.info(f"Unsupervised anomaly detection completed. Detected {anomalies_detected} anomalous hours out of {len(hourly_data)} hours")

            return data_with_anomalies
        else:
            logger.error("Unsupervised anomaly detection failed")
            return None

    except Exception as e:
        logger.error(f"Error in unsupervised anomaly detection: {e}")
        return None

def supervised_anomaly_detection_hourly(network_traffic_path='network_traffic.csv', predictions_path='predictions.csv', contamination=0.1):
    """
    Supervised Anomaly Detection on Hourly Network Traffic Data

    Purpose: Train a supervised model using labeled data and apply it to hourly aggregated traffic
    for more accurate anomaly predictions when ground truth labels are available.

    Method: Random Forest Classifier trained on labeled hourly data
    - Uses ground truth labels from predictions.csv
    - Applied to hourly aggregated features for consistency
    - Provides explainable predictions with feature importance

    Args:
        network_traffic_path (str): Path to network_traffic.csv
        predictions_path (str): Path to predictions.csv (ground truth labels)
        contamination (float): Expected proportion of anomalous hours (for reference)

    Returns:
        pd.DataFrame: Hourly data with supervised anomaly predictions
    """
    logger.info("Starting Supervised Anomaly Detection on Hourly Data...")

    try:
        from anomaly_detection import train_supervised_anomaly_model, predict_anomalies_supervised

        # Aggregate data to hourly
        hourly_data = aggregate_hourly_traffic(network_traffic_path)
        if hourly_data is None:
            return None

        # Train supervised model using ground truth labels
        model, scaler, feature_columns = train_supervised_anomaly_model(network_traffic_path, predictions_path)

        if model is not None:
            logger.info("Supervised model trained successfully, applying to hourly data...")

            # Apply supervised model to hourly aggregated data
            # Note: We need to use the original raw data features for supervised prediction
            # since the model was trained on raw features, not aggregated ones
            raw_data = pd.read_csv(network_traffic_path)
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])

            # Get supervised predictions on raw data
            supervised_results = predict_anomalies_supervised(raw_data, model, scaler, feature_columns)

            if supervised_results is not None:
                # Aggregate supervised predictions to hourly level
                supervised_results['timestamp'] = pd.to_datetime(supervised_results['timestamp'])
                hourly_supervised = supervised_results.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
                    'prediction': 'mean',  # Average prediction (0-1 range)
                    'anomaly_score': 'mean'  # Average anomaly score
                }).reset_index()

                # Convert to binary predictions based on threshold
                hourly_supervised['prediction'] = (hourly_supervised['prediction'] > 0.5).astype(int)
                hourly_supervised['anomaly_label'] = hourly_supervised['prediction'].map({0: 'normal', 1: 'anomaly'})

                # Merge with aggregated features
                final_results = hourly_data.copy()
                final_results = final_results.merge(hourly_supervised[['timestamp', 'prediction', 'anomaly_score', 'anomaly_label']],
                                                  on='timestamp', how='left')

                supervised_anomalies = sum(final_results['prediction'])
                logger.info(f"Supervised anomaly detection completed. Detected {supervised_anomalies} anomalous hours out of {len(final_results)} hours")

                return final_results
            else:
                logger.error("Supervised prediction failed")
                return None
        else:
            logger.error("Supervised model training failed")
            return None

    except Exception as e:
        logger.error(f"Error in supervised anomaly detection: {e}")
        return None

def hybrid_anomaly_detection_hourly(network_traffic_path='network_traffic.csv', predictions_path='predictions.csv', contamination=0.1):
    """
    Hybrid Anomaly Detection: Combines unsupervised and supervised approaches

    Purpose: Leverage both unsupervised discovery and supervised accuracy when labels are available.
    Falls back to unsupervised-only approach if supervised training fails.

    Method:
    1. Apply unsupervised Isolation Forest to establish baseline
    2. If supervised labels available, train and apply supervised model
    3. Combine results: use supervised as primary, unsupervised as secondary signal

    Args:
        network_traffic_path (str): Path to network_traffic.csv
        predictions_path (str): Path to predictions.csv (optional)
        contamination (float): Expected proportion of anomalous hours

    Returns:
        pd.DataFrame: Hourly data with hybrid anomaly predictions
    """
    logger.info("Starting Hybrid Anomaly Detection on Hourly Data...")

    try:
        # Always run unsupervised as baseline
        unsupervised_results = unsupervised_anomaly_detection_hourly(network_traffic_path, contamination)
        if unsupervised_results is None:
            return None

        # Try supervised approach if predictions file exists
        try:
            supervised_results = supervised_anomaly_detection_hourly(network_traffic_path, predictions_path, contamination)

            if supervised_results is not None:
                # Combine results
                hybrid_results = unsupervised_results.copy()
                hybrid_results['supervised_prediction'] = supervised_results['prediction']
                hybrid_results['supervised_score'] = supervised_results['anomaly_score']
                hybrid_results['supervised_label'] = supervised_results['anomaly_label']

                # Use supervised as final prediction, unsupervised as additional signal
                hybrid_results['final_prediction'] = hybrid_results['supervised_prediction']
                hybrid_results['final_label'] = hybrid_results['supervised_label']
                hybrid_results['final_score'] = hybrid_results['supervised_score']

                # Add ensemble confidence
                hybrid_results['ensemble_confidence'] = (hybrid_results['prediction'] + hybrid_results['supervised_prediction']) / 2

                final_anomalies = sum(hybrid_results['final_prediction'])
                logger.info(f"Hybrid anomaly detection completed. Final predictions: {final_anomalies} anomalies out of {len(hybrid_results)} hours")
                return hybrid_results
            else:
                logger.warning("Supervised approach failed, using unsupervised results only")
                return unsupervised_results

        except FileNotFoundError:
            logger.info("No predictions.csv found, using unsupervised results only")
            return unsupervised_results
        except Exception as e:
            logger.warning(f"Supervised approach failed ({e}), using unsupervised results only")
            return unsupervised_results

    except Exception as e:
        logger.error(f"Error in hybrid anomaly detection: {e}")
        return None

def explain_detection_approaches():
    """
    Explains the available anomaly detection approaches for hourly network traffic.
    """
    explanation = """
    ANOMALY DETECTION APPROACHES FOR HOURLY NETWORK TRAFFIC

    1. UNSUPERVISED APPROACH (Default):
       - Uses Isolation Forest on hourly aggregated features
       - No labels required - purely data-driven
       - Identifies anomalous patterns based on feature isolation
       - Best for: New environments, no labeled data available

    2. SUPERVISED APPROACH (When labels available):
       - Trains Random Forest on labeled traffic data
       - Uses ground truth from predictions.csv
       - Provides explainable predictions with feature importance
       - Best for: Improving accuracy when labeled data exists

    3. HYBRID APPROACH (Recommended):
       - Combines unsupervised baseline with supervised accuracy
       - Falls back to unsupervised if supervised fails
       - Uses supervised as primary prediction when available
       - Best for: Robust detection with maximum accuracy

    4. HOURLY AGGREGATION FEATURES:
       - packet_count: Number of packets per hour
       - avg_packet_size: Average packet size
       - std_packet_size: Standard deviation of packet sizes
       - min/max_packet_size: Range of packet sizes
       - unique_source_ips: Number of unique source IPs
       - unique_dest_ips: Number of unique destination IPs
    """
    print(explanation)

if __name__ == "__main__":
    # Explain the approach
    explain_unsupervised_approach()

    # Run unsupervised anomaly detection on hourly data
    results = unsupervised_anomaly_detection_hourly()
    if results is not None:
        print(f"Detected {sum(results['prediction'])} anomalous hours out of {len(results)} total hours")
        print("Sample results:")
        print(results[['timestamp', 'packet_count', 'avg_packet_size', 'prediction', 'anomaly_label']].head())
