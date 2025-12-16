import logging
import os
import json
from datetime import datetime
from config import LOG_FILE, LOG_LEVEL

def setup_logger(name='ml_pipeline'):
    """
    Sets up a singleton logger with file and console handlers.
    Ensures duplicate handlers are not added.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Prevent duplicate handlers every time script runs
    if logger.handlers:
        return logger

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

class PipelineLogger:
    """
    Comprehensive logger for ML pipeline runs.
    Logs detailed information about each pipeline execution.
    """

    def __init__(self, log_file='logs/pipeline_runs.log'):
        """
        Initialize the pipeline logger.

        Args:
            log_file (str): Path to the pipeline runs log file
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_pipeline_run(self, run_id, mode, start_time, end_time, success, steps_completed=None,
                        error_message=None, metrics=None, data_stats=None):
        """
        Log a complete pipeline run.

        Args:
            run_id (str): Unique identifier for the run
            mode (str): Pipeline mode ('supervised' or 'unsupervised')
            start_time (datetime): Start time of the run
            end_time (datetime): End time of the run
            success (bool): Whether the run was successful
            steps_completed (list): List of completed steps
            error_message (str): Error message if failed
            metrics (dict): Performance metrics
            data_stats (dict): Data statistics
        """
        execution_time = (end_time - start_time).total_seconds()

        log_entry = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'execution_time_seconds': execution_time,
            'success': success,
            'steps_completed': steps_completed or [],
            'error_message': error_message,
            'metrics': metrics or {},
            'data_stats': data_stats or {}
        }

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # Also log to main logger
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Pipeline run {run_id} completed: {status} "
                       f"(Execution time: {execution_time:.2f}s)")

            if error_message:
                logger.error(f"Pipeline run {run_id} error: {error_message}")

        except Exception as e:
            logger.error(f"Failed to log pipeline run: {e}")

    def get_recent_runs(self, limit=10):
        """
        Get recent pipeline runs.

        Args:
            limit (int): Number of recent runs to retrieve

        Returns:
            list: List of recent pipeline run logs
        """
        if not os.path.exists(self.log_file):
            return []

        runs = []
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        runs.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"Failed to read pipeline logs: {e}")
            return []

        # Sort by timestamp descending and return most recent
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        return runs[:limit]

    def get_run_summary(self):
        """
        Get summary statistics of pipeline runs.

        Returns:
            dict: Summary statistics
        """
        runs = self.get_recent_runs(limit=1000)  # Get all runs

        if not runs:
            return {"message": "No pipeline runs found"}

        total_runs = len(runs)
        successful_runs = len([r for r in runs if r['success']])
        failed_runs = total_runs - successful_runs

        execution_times = [r['execution_time_seconds'] for r in runs if 'execution_time_seconds' in r]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        supervised_runs = len([r for r in runs if r['mode'] == 'supervised'])
        unsupervised_runs = len([r for r in runs if r['mode'] == 'unsupervised'])

        return {
            'total_runs': total_runs,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
            'average_execution_time': avg_execution_time,
            'supervised_runs': supervised_runs,
            'unsupervised_runs': unsupervised_runs,
            'last_run': runs[0] if runs else None
        }

# Global instances
logger = setup_logger()
pipeline_logger = PipelineLogger()
