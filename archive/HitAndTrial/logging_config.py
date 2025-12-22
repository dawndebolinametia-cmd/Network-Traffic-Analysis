import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file='logs/ml_pipeline.log', max_bytes=10*1024*1024, backup_count=5):
    """
    Set up logging configuration with rotation.

    Args:
        log_level: Logging level (e.g., logging.INFO)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_pipeline_logger():
    """
    Get a specific logger for pipeline runs.
    """
    return logging.getLogger('ml_pipeline')

def log_pipeline_run(run_id, mode, start_time, end_time, success, steps_completed, error_message=None, metrics=None, data_stats=None):
    """
    Log a pipeline run with comprehensive details.
    """
    logger = get_pipeline_logger()

    # Calculate duration
    duration = (end_time - start_time).total_seconds()

    # Prepare log message
    message = f"Pipeline Run: {run_id} | Mode: {mode} | Success: {success} | Duration: {duration:.2f}s | Steps: {', '.join(steps_completed)}"

    if error_message:
        message += f" | Error: {error_message}"

    if metrics:
        message += f" | Metrics: {metrics}"

    if data_stats:
        message += f" | Data Stats: {data_stats}"

    if success:
        logger.info(message)
    else:
        logger.error(message)

# Initialize logging when module is imported
setup_logging()
