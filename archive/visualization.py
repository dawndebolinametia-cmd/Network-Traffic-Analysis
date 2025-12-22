import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import json
from datetime import datetime
from config import REPORTS_PATH, METABASE_URL, METABASE_USERNAME, METABASE_PASSWORD
from logging_utils import logger

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_time_series_traffic(data, time_column='timestamp', value_column='packet_size', title='Network Traffic Time Series'):
    """
    Creates a time series plot of network traffic.

    Args:
        data (pd.DataFrame): Input data.
        time_column (str): Column containing timestamps.
        value_column (str): Column to plot over time.
        title (str): Plot title.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Convert timestamp if needed
        if time_column in data.columns:
            data[time_column] = pd.to_datetime(data[time_column])
            data = data.sort_values(time_column)

            ax.plot(data[time_column], data[value_column], linewidth=1, alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel(value_column.replace('_', ' ').title())
            ax.set_title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            filename = f"{REPORTS_PATH}/time_series_{value_column}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Time series plot saved to {filename}")
            plt.show()
        else:
            logger.error(f"Time column '{time_column}' not found in data")

    except Exception as e:
        logger.error(f"Error creating time series plot: {e}")

def plot_anomaly_distribution(data, anomaly_column='is_anomaly', title='Anomaly Distribution'):
    """
    Creates a pie chart showing anomaly distribution.

    Args:
        data (pd.DataFrame): Input data.
        anomaly_column (str): Column containing anomaly labels.
        title (str): Plot title.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8))

        if anomaly_column in data.columns:
            anomaly_counts = data[anomaly_column].value_counts()

            # Convert boolean to readable labels if needed
            if data[anomaly_column].dtype == bool:
                labels = ['Normal', 'Anomaly']
                sizes = [anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)]
            else:
                labels = anomaly_counts.index
                sizes = anomaly_counts.values

            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.set_title(title)
            ax.axis('equal')

            # Save plot
            filename = f"{REPORTS_PATH}/anomaly_distribution.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly distribution plot saved to {filename}")
            plt.show()
        else:
            logger.error(f"Anomaly column '{anomaly_column}' not found in data")

    except Exception as e:
        logger.error(f"Error creating anomaly distribution plot: {e}")

def plot_traffic_type_distribution(data, traffic_type_column='traffic_type', title='Traffic Type Distribution'):
    """
    Creates a bar chart showing traffic type distribution.

    Args:
        data (pd.DataFrame): Input data.
        traffic_type_column (str): Column containing traffic types.
        title (str): Plot title.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if traffic_type_column in data.columns:
            type_counts = data[traffic_type_column].value_counts()

            bars = ax.bar(range(len(type_counts)), type_counts.values)
            ax.set_xlabel('Traffic Type')
            ax.set_ylabel('Count')
            ax.set_title(title)
            ax.set_xticks(range(len(type_counts)))
            ax.set_xticklabels(type_counts.index, rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()

            # Save plot
            filename = f"{REPORTS_PATH}/traffic_type_distribution.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Traffic type distribution plot saved to {filename}")
            plt.show()
        else:
            logger.error(f"Traffic type column '{traffic_type_column}' not found in data")

    except Exception as e:
        logger.error(f"Error creating traffic type distribution plot: {e}")

def plot_packet_size_distribution(data, packet_size_column='packet_size', title='Packet Size Distribution'):
    """
    Creates a histogram of packet sizes.

    Args:
        data (pd.DataFrame): Input data.
        packet_size_column (str): Column containing packet sizes.
        title (str): Plot title.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if packet_size_column in data.columns:
            ax.hist(data[packet_size_column], bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Packet Size (bytes)')
            ax.set_ylabel('Frequency')
            ax.set_title(title)

            # Add statistics
            mean_size = data[packet_size_column].mean()
            median_size = data[packet_size_column].median()
            ax.axvline(mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.0f}')
            ax.axvline(median_size, color='green', linestyle='--', label=f'Median: {median_size:.0f}')
            ax.legend()

            plt.tight_layout()

            # Save plot
            filename = f"{REPORTS_PATH}/packet_size_distribution.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Packet size distribution plot saved to {filename}")
            plt.show()
        else:
            logger.error(f"Packet size column '{packet_size_column}' not found in data")

    except Exception as e:
        logger.error(f"Error creating packet size distribution plot: {e}")

def plot_anomaly_scores_over_time(data, time_column='timestamp', score_column='anomaly_score', title='Anomaly Scores Over Time'):
    """
    Creates a scatter plot of anomaly scores over time.

    Args:
        data (pd.DataFrame): Input data.
        time_column (str): Column containing timestamps.
        score_column (str): Column containing anomaly scores.
        title (str): Plot title.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        if time_column in data.columns and score_column in data.columns:
            data[time_column] = pd.to_datetime(data[time_column])
            data = data.sort_values(time_column)

            # Color points based on anomaly threshold
            colors = ['red' if score > 0 else 'blue' for score in data[score_column]]

            ax.scatter(data[time_column], data[score_column], c=colors, alpha=0.6)
            ax.set_xlabel('Time')
            ax.set_ylabel('Anomaly Score')
            ax.set_title(title)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Threshold')

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='blue', label='Normal'),
                             Patch(facecolor='red', label='Anomaly')]
            ax.legend(handles=legend_elements)

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            filename = f"{REPORTS_PATH}/anomaly_scores_time.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly scores over time plot saved to {filename}")
            plt.show()
        else:
            logger.error(f"Required columns '{time_column}' or '{score_column}' not found in data")

    except Exception as e:
        logger.error(f"Error creating anomaly scores over time plot: {e}")

def plot_correlation_heatmap(data, title='Feature Correlation Heatmap'):
    """
    Creates a correlation heatmap of numerical features.

    Args:
        data (pd.DataFrame): Input data.
        title (str): Plot title.
    """
    try:
        # Select numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))

            correlation_matrix = data[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, ax=ax)
            ax.set_title(title)

            plt.tight_layout()

            # Save plot
            filename = f"{REPORTS_PATH}/correlation_heatmap.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {filename}")
            plt.show()
        else:
            logger.error("Not enough numerical columns for correlation analysis")

    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")

def create_comprehensive_dashboard(data):
    """
    Creates a comprehensive dashboard with multiple visualizations.

    Args:
        data (pd.DataFrame): Input data.
    """
    try:
        logger.info("Creating comprehensive visualization dashboard...")

        # Create multiple plots
        if 'timestamp' in data.columns and 'packet_size' in data.columns:
            plot_time_series_traffic(data)

        if 'is_anomaly' in data.columns:
            plot_anomaly_distribution(data)

        if 'traffic_type' in data.columns:
            plot_traffic_type_distribution(data)

        if 'packet_size' in data.columns:
            plot_packet_size_distribution(data)

        if 'timestamp' in data.columns and 'anomaly_score' in data.columns:
            plot_anomaly_scores_over_time(data)

        plot_correlation_heatmap(data)

        logger.info("Comprehensive dashboard created successfully")

    except Exception as e:
        logger.error(f"Error creating comprehensive dashboard: {e}")

def authenticate_metabase(url, username, password):
    """
    Authenticate with Metabase API and return session token.

    Args:
        url (str): Metabase URL
        username (str): Metabase username
        password (str): Metabase password

    Returns:
        str: Session token or None if authentication fails
    """
    try:
        auth_url = f"{url}/api/session"
        auth_data = {
            "username": username,
            "password": password
        }

        response = requests.post(auth_url, json=auth_data)
        response.raise_for_status()

        token = response.json().get("id")
        logger.info("Successfully authenticated with Metabase")
        return token

    except Exception as e:
        logger.error(f"Failed to authenticate with Metabase: {e}")
        return None

def create_metabase_dataset(data, token, url, dataset_name="Network Traffic Predictions"):
    """
    Create a dataset in Metabase from the prediction data.

    Args:
        data (pd.DataFrame): Data to upload
        token (str): Metabase session token
        url (str): Metabase URL
        dataset_name (str): Name for the dataset

    Returns:
        int: Dataset ID or None if creation fails
    """
    try:
        # Save data to CSV temporarily for upload
        temp_csv = "temp_predictions.csv"
        data.to_csv(temp_csv, index=False)

        # Create dataset via Metabase API
        headers = {
            "X-Metabase-Session": token,
            "Content-Type": "multipart/form-data"
        }

        # for now, we'll save the data in a format Metabase can access
        metabase_csv = f"{REPORTS_PATH}/metabase_predictions.csv"
        data.to_csv(metabase_csv, index=False)

        logger.info(f"Prepared data for Metabase at {metabase_csv}")
        return metabase_csv

    except Exception as e:
        logger.error(f"Failed to create Metabase dataset: {e}")
        return None

def create_metabase_question(token, url, database_id, table_name, question_name):
    """
    Create a Metabase question for a table.

    Args:
        token (str): Metabase session token
        url (str): Metabase URL
        database_id (int): Database ID
        table_name (str): Table name
        question_name (str): Name for the question

    Returns:
        int: Question ID or None
    """
    try:
        headers = {"X-Metabase-Session": token, "Content-Type": "application/json"}
        question_url = f"{url}/api/card"

        query = {
            "database": database_id,
            "query": {"source-table": f"card__{table_name}"},  # Assuming table is already synced
            "type": "query",
            "name": question_name
        }

        response = requests.post(question_url, headers=headers, json=query)
        response.raise_for_status()
        question_id = response.json().get("id")
        logger.info(f"Created Metabase question '{question_name}' with ID {question_id}")
        return question_id
    except Exception as e:
        logger.error(f"Failed to create Metabase question: {e}")
        return None

def create_metabase_dashboard(token, url, dashboard_name, question_ids):
    """
    Create a Metabase dashboard with questions.

    Args:
        token (str): Metabase session token
        url (str): Metabase URL
        dashboard_name (str): Name for the dashboard
        question_ids (list): List of question IDs to add

    Returns:
        int: Dashboard ID or None
    """
    try:
        headers = {"X-Metabase-Session": token, "Content-Type": "application/json"}
        dashboard_url = f"{url}/api/dashboard"

        dashboard_data = {
            "name": dashboard_name,
            "description": "Network Traffic Analysis Dashboard"
        }

        response = requests.post(dashboard_url, headers=headers, json=dashboard_data)
        response.raise_for_status()
        dashboard_id = response.json().get("id")
        logger.info(f"Created Metabase dashboard '{dashboard_name}' with ID {dashboard_id}")

        # Add questions to dashboard
        for i, qid in enumerate(question_ids):
            card_data = {
                "cardId": qid,
                "row": i // 2,
                "col": (i % 2) * 6,
                "sizeX": 6,
                "sizeY": 4
            }
            add_card_url = f"{url}/api/dashboard/{dashboard_id}/cards"
            requests.post(add_card_url, headers=headers, json=card_data)

        return dashboard_id
    except Exception as e:
        logger.error(f"Failed to create Metabase dashboard: {e}")
        return None

def update_metabase_dashboard(data, metabase_url, username, password, database_id=None, dashboard_id=None):
    """
    Update Metabase dashboard with prediction data.

    Args:
        data (pd.DataFrame): Prediction data
        metabase_url (str): Metabase URL
        username (str): Metabase username
        password (str): Metabase password
        database_id (int): Metabase database ID (optional)
        dashboard_id (int): Metabase dashboard ID (optional)
    """
    try:
        logger.info("Starting Metabase dashboard update...")

        # Authenticate
        token = authenticate_metabase(metabase_url, username, password)
        if not token:
            logger.error("Cannot update Metabase dashboard without authentication")
            return False

        from config import METABASE_DATABASE_ID, NETWORK_TRAFFIC_TABLE, PREDICTIONS_TABLE
        database_id = database_id or METABASE_DATABASE_ID

        # Create questions for tables
        traffic_question = create_metabase_question(token, metabase_url, database_id, NETWORK_TRAFFIC_TABLE, "Network Traffic Data")
        pred_question = create_metabase_question(token, metabase_url, database_id, PREDICTIONS_TABLE, "Traffic Predictions")

        if traffic_question and pred_question:
            # Create dashboard if not exists
            dashboard_name = "Network Traffic Analysis"
            dashboard_id = create_metabase_dashboard(token, metabase_url, dashboard_name, [traffic_question, pred_question])
            if dashboard_id:
                logger.info(f"Dashboard '{dashboard_name}' created/updated with ID {dashboard_id}")
            else:
                logger.error("Failed to create dashboard")
                return False
        else:
            logger.error("Failed to create questions")
            return False

        # Save summary statistics for dashboard
        summary = {
            "total_records": len(data),
            "anomalies": len(data[data.get('prediction', 0) == 1]) if 'prediction' in data.columns else 0,
            "normal_traffic": len(data[data.get('prediction', 0) == 0]) if 'prediction' in data.columns else len(data),
            "anomaly_percentage": (len(data[data.get('prediction', 0) == 1]) / len(data) * 100) if 'prediction' in data.columns and len(data) > 0 else 0
        }

        summary_file = f"{REPORTS_PATH}/metabase_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dashboard summary saved to {summary_file}")
        logger.info("Metabase dashboard update completed successfully")

        return True

    except Exception as e:
        logger.error(f"Error updating Metabase dashboard: {e}")
        return False

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    synthetic_data = pd.DataFrame({
        'timestamp': dates,
        'packet_size': np.random.normal(500, 200, 1000),
        'duration': np.random.exponential(1, 1000),
        'is_anomaly': np.random.choice([True, False], 1000, p=[0.1, 0.9]),
        'anomaly_score': np.random.normal(0, 1, 1000),
        'traffic_type': np.random.choice(['HTTP', 'HTTPS', 'DNS', 'SSH'], 1000)
    })

    # Create visualizations
    create_comprehensive_dashboard(synthetic_data)


def create_visualization(data):
    """
    Wrapper function to create all visualizations.
    Returns True if successful.
    """
    try:
        from config import REPORTS_PATH
        from logging_utils import logger
        create_comprehensive_dashboard(data)  # the dashboard function we wrote earlier
        return True
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return False
