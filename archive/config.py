import os
import datetime
import pandas as pd

# Database Configuration
DB_HOST = 'localhost'
DB_USER = 'debbie'
DB_PASSWORD = '12345'
DB_NAME = 'our_mysql'
DB_PORT = 3306

# Table Names
NETWORK_TRAFFIC_TABLE = 'network_traffic'
PREDICTIONS_TABLE = 'hourly_predictions'

TRAFFIC_DATA_TABLE = NETWORK_TRAFFIC_TABLE

# File Paths (dynamic, runtime-driven)
BASE_DIR = os.getcwd()
NETWORK_TRAFFIC_CSV = os.path.join(BASE_DIR, 'network_traffic.csv')
REPORTS_PATH = os.path.join(BASE_DIR, 'reports')
PREDICTIONS_CSV = os.path.join(REPORTS_PATH, 'predictions.csv')

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
CACHE_PATH = os.path.join(BASE_DIR, 'cache')

# Logging Configuration
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'app.log')
LOG_LEVEL = 'INFO'

# Other Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Monitoring Configuration
MONITORING_FILE = os.path.join(BASE_DIR, 'monitoring', 'performance_history.json')
RETRAIN_THRESHOLD = 0.05  # Retrain if accuracy drops by more than 5%

# Metabase Configuration
METABASE_URL = 'http://localhost:3000'
METABASE_USERNAME = 'Debbie'
METABASE_PASSWORD = 'Anime#210305'
METABASE_DATABASE_ID = 1  

# Time-handling logic (dynamic, runtime-driven)
def get_current_time():
    return datetime.datetime.now()

CURRENT_TIME = get_current_time()

# Load Isolation Forest model for anomaly detection
ISOLATION_FOREST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'isolation_forest_model.pkl')

def load_isolation_forest_model():
    """Load the trained Isolation Forest model."""
    try:
        import joblib
        return joblib.load(ISOLATION_FOREST_MODEL_PATH)
    except Exception as e:
        print(f"Error loading Isolation Forest model: {e}")
        return None

_ISO_MODEL = None

def get_isolation_forest_model():
    global _ISO_MODEL
    if _ISO_MODEL is None:
        _ISO_MODEL = load_isolation_forest_model()
    return _ISO_MODEL


# Deterministic generation functions for synthetic data
def generate_packet_size(hour):
    # Deterministic formula based on hour
    base = 2000
    variation = (hour * 17) % 2000
    return base + variation

def generate_ips(hour):
    ip_base = hour % 256
    source_ip = f"192.168.{ip_base}.1"
    dest_ip = f"192.168.{(ip_base + 1) % 256}.2"
    return source_ip, dest_ip

def is_anomaly(packet_size, hour):
    """
    Determine if a packet is an anomaly using Isolation Forest.
    Returns True if anomaly (outlier), False otherwise.
    """
    if ISOLATION_FOREST_MODEL is None:
        raise ValueError("Isolation Forest model not loaded")
    # Prepare features
    features = [[packet_size, hour]]
    prediction = ISOLATION_FOREST_MODEL.predict(features)
    return prediction[0] == -1  # -1 indicates anomaly

def calculate_anomaly_score(packet_size, hour):
    """
    Calculate anomaly score using Isolation Forest decision function.
    Higher scores indicate more anomalous behavior.
    """
    if ISOLATION_FOREST_MODEL is None:
        raise ValueError("Isolation Forest model not loaded")
    # Prepare features
    features = [[packet_size, hour]]
    score = ISOLATION_FOREST_MODEL.decision_function(features)
    return score[0]
