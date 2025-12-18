import os
import datetime
import pandas as pd

# Database Configuration
DB_HOST = '127.0.0.1'
DB_USER = 'debbie'
DB_PASSWORD = 'Anime#210305'
DB_NAME = 'analytics_data'
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
METABASE_DATABASE_ID = 1  # Assuming the MySQL database ID in Metabase is 1; adjust if different

# Time-handling logic (dynamic, runtime-driven)
def get_current_time():
    return datetime.datetime.now()

CURRENT_TIME = get_current_time()

# Live-feed anomaly rules derived from network_traffic.csv
def derive_anomaly_rules():
    df = pd.read_csv(NETWORK_TRAFFIC_CSV)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    mean_packet = df['packet_size'].mean()
    std_packet = df['packet_size'].std()
    threshold = mean_packet + 2 * std_packet
    return mean_packet, std_packet, threshold

MEAN_PACKET, STD_PACKET, ANOMALY_THRESHOLD = derive_anomaly_rules()

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

def is_anomaly(packet_size):
    return packet_size > ANOMALY_THRESHOLD

def calculate_anomaly_score(packet_size):
    return (packet_size - MEAN_PACKET) / STD_PACKET
