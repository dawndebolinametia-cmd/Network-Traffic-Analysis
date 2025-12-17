# Database Configuration
DB_HOST = '127.0.0.1'          
DB_USER = 'debbie'             
DB_PASSWORD = 'Anime#210305'   
DB_NAME = 'analytics_data'
DB_PORT = 3306

NETWORK_TRAFFIC_TABLE = 'network_traffic'
PREDICTION_ANOMALY_TABLE = 'prediction_anomaly'
TRAFFIC_DATA_TABLE = NETWORK_TRAFFIC_TABLE



# Table Names
NETWORK_TRAFFIC_TABLE = 'network_traffic'
PREDICTION_ANOMALY_TABLE = 'prediction_anomaly'


# Model Paths
MODEL_PATH = 'models/random_forest_model.pkl'
CACHE_PATH = 'cache/'

# Logging Configuration
LOG_FILE = 'logs/app.log'
LOG_LEVEL = 'INFO'

# Other Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
REPORTS_PATH = 'reports/'

# Monitoring Configuration
MONITORING_FILE = 'monitoring/performance_history.json'
RETRAIN_THRESHOLD = 0.05  # Retrain if accuracy drops by more than 5%

# Metabase Configuration
METABASE_URL = 'http://localhost:3000'
METABASE_USERNAME = 'Debbie'
METABASE_PASSWORD = 'Anime#210305'
