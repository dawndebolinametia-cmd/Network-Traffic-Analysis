import pymysql
import logging
from config import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, NETWORK_TRAFFIC_TABLE, PREDICTIONS_TABLE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_traffic_summary_view():
    """Create traffic summary view for Metabase dashboards."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Drop view if exists
    cursor.execute("DROP VIEW IF EXISTS traffic_summary")

    # Create traffic summary view
    create_sql = f"""
    CREATE VIEW traffic_summary AS
    SELECT
        DATE(nt.TSesStart) as date,
        HOUR(nt.TSesStart) as hour,
        COUNT(*) as session_count,
        SUM(nt.PktPerSesIn + nt.PktPerSesOut) as total_packets,
        SUM(nt.BPerSesIn + nt.BPerSesOut) as total_bytes,
        AVG(nt.PktPerSesIn + nt.PktPerSesOut) as avg_packets_per_session,
        AVG(nt.BPerSesIn + nt.BPerSesOut) as avg_bytes_per_session,
        COUNT(CASE WHEN p.prediction = 1 THEN 1 END) as anomaly_count,
        ROUND((COUNT(CASE WHEN p.prediction = 1 THEN 1 END) / COUNT(*)) * 100, 2) as anomaly_percentage
    FROM {NETWORK_TRAFFIC_TABLE} nt
    LEFT JOIN {PREDICTIONS_TABLE} p ON nt.id = p.session_id AND p.model_type = 'isolation_forest'
    GROUP BY DATE(nt.TSesStart), HOUR(nt.TSesStart)
    ORDER BY date DESC, hour DESC
    """

    cursor.execute(create_sql)
    logger.info("Created traffic_summary view")

    conn.commit()
    cursor.close()
    conn.close()

def create_anomaly_analysis_view():
    """Create anomaly analysis view for Metabase dashboards."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Drop view if exists
    cursor.execute("DROP VIEW IF EXISTS anomaly_analysis")

    # Create anomaly analysis view
    create_sql = f"""
    CREATE VIEW anomaly_analysis AS
    SELECT
        nt.*,
        p.prediction,
        p.anomaly_score,
        p.traffic_type,
        p.is_anomaly,
        CASE
            WHEN p.anomaly_score < -0.5 THEN 'High Risk'
            WHEN p.anomaly_score < 0 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END as risk_level,
        p.model_type,
        p.log_time as prediction_time
    FROM {NETWORK_TRAFFIC_TABLE} nt
    LEFT JOIN {PREDICTIONS_TABLE} p ON nt.id = p.session_id
    WHERE p.prediction = 1
    ORDER BY p.anomaly_score ASC, nt.TSesStart DESC
    """

    cursor.execute(create_sql)
    logger.info("Created anomaly_analysis view")

    conn.commit()
    cursor.close()
    conn.close()

def create_session_stats_view():
    """Create session statistics view for Metabase dashboards."""
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Drop view if exists
    cursor.execute("DROP VIEW IF EXISTS session_stats")

    # Create session stats view
    create_sql = f"""
    CREATE VIEW session_stats AS
    SELECT
        nt.srcIP as source_ip,
        nt.dstIP as dest_ip,
        nt.srcPort as source_port,
        nt.dstPort as dest_port,
        COUNT(*) as session_count,
        SUM(nt.PktPerSesIn + nt.PktPerSesOut) as total_packets,
        SUM(nt.BPerSesIn + nt.BPerSesOut) as total_bytes,
        AVG(nt.PktPerSesIn + nt.PktPerSesOut) as avg_packets,
        AVG(nt.BPerSesIn + nt.BPerSesOut) as avg_bytes,
        AVG(nt.SessDuration) as avg_duration_seconds,
        COUNT(CASE WHEN p.prediction = 1 THEN 1 END) as anomaly_sessions,
        ROUND((COUNT(CASE WHEN p.prediction = 1 THEN 1 END) / COUNT(*)) * 100, 2) as anomaly_rate,
        MAX(nt.TSesStart) as last_session_time
    FROM {NETWORK_TRAFFIC_TABLE} nt
    LEFT JOIN {PREDICTIONS_TABLE} p ON nt.id = p.session_id AND p.model_type = 'isolation_forest'
    GROUP BY nt.srcIP, nt.dstIP, nt.srcPort, nt.dstPort
    HAVING session_count > 1
    ORDER BY session_count DESC
    """

    cursor.execute(create_sql)
    logger.info("Created session_stats view")

    conn.commit()
    cursor.close()
    conn.close()

def setup_metabase_views():
    """Set up all Metabase views."""
    try:
        logger.info("Setting up Metabase views...")
        create_traffic_summary_view()
        create_anomaly_analysis_view()
        create_session_stats_view()
        logger.info("All Metabase views created successfully")
    except Exception as e:
        logger.error(f"Error setting up Metabase views: {e}")
        raise

if __name__ == "__main__":
    setup_metabase_views()
