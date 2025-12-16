import pandas as pd
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from logging_utils import logger, pipeline_logger
from data_ingestion import fetch_network_traffic_data
from preprocessing import preprocess_data
from model_training import train_model, load_existing_model
from anomaly_detection import detect_anomalies_isolation_forest
from evaluation import predict_and_evaluate
from prediction_output import save_predictions
from visualization import create_visualization, update_metabase_dashboard
from model_monitoring import monitor
from config import *

class AutomatedMLPipeline:
    """
    Automated ML pipeline for network traffic analysis.
    Supports both supervised classification and unsupervised anomaly detection.
    """

    def __init__(self, mode='supervised', schedule_interval_hours=24):
        """
        Initialize the automated pipeline.

        Args:
            mode (str): 'supervised' for classification or 'unsupervised' for anomaly detection
            schedule_interval_hours (int): Hours between automated runs
        """
        self.mode = mode
        self.schedule_interval = schedule_interval_hours
        self.scheduler = BackgroundScheduler()
        self.is_running = False

        # Register cleanup
        atexit.register(self.cleanup)

    def run_pipeline(self):
        """
        Execute the complete ML pipeline.
        """
        # Generate unique run ID
        run_id = f"{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_datetime = datetime.now()
        logger.info(f"Starting automated ML pipeline in {self.mode} mode.")

        steps_completed = []
        metrics = {}
        data_stats = {}
        error_message = None

        try:
            start_time = time.time()

            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            data = fetch_network_traffic_data()
            if data is None or data.empty:
                error_message = "Data ingestion failed. Pipeline terminated."
                logger.error(error_message)
                return False

            steps_completed.append("data_ingestion")
            data_stats = {
                'raw_data_rows': len(data),
                'raw_data_columns': len(data.columns),
                'raw_data_columns_list': list(data.columns)
            }

            # Step 2: Preprocessing
            logger.info("Step 2: Preprocessing")
            processed_data, encoders, scaler = preprocess_data(data, save_preprocessors=True)
            steps_completed.append("preprocessing")
            data_stats.update({
                'processed_data_rows': len(processed_data),
                'processed_data_columns': len(processed_data.columns),
                'processed_data_columns_list': list(processed_data.columns)
            })

            if self.mode == 'supervised':
                # Supervised Classification Pipeline
                success = self._run_supervised_pipeline(processed_data, metrics)
                if success:
                    steps_completed.extend(["model_training", "model_evaluation", "prediction_saving"])
            elif self.mode == 'unsupervised':
                # Unsupervised Anomaly Detection Pipeline
                success = self._run_unsupervised_pipeline(processed_data)
                if success:
                    steps_completed.extend(["anomaly_detection", "prediction_saving"])
            else:
                error_message = f"Invalid mode: {self.mode}"
                logger.error(error_message)
                return False

            if success:
                # Step 5: Visualization
                logger.info("Step 5: Visualization")
                viz_success = create_visualization(processed_data)
                if viz_success:
                    steps_completed.append("visualization")
                    logger.info("Visualization completed successfully")
                else:
                    logger.warning("Visualization failed, but pipeline continues.")

                # Step 6: Update Metabase dashboard
                metabase_success = update_metabase_dashboard(
                    processed_data,
                    METABASE_URL,
                    METABASE_USERNAME,
                    METABASE_PASSWORD
                )
                if metabase_success:
                    steps_completed.append("metabase_update")
                    logger.info("Metabase dashboard updated successfully")
                else:
                    logger.warning("Failed to update Metabase dashboard")

            execution_time = time.time() - start_time
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds.")

            # Log comprehensive pipeline run
            end_datetime = datetime.now()
            pipeline_logger.log_pipeline_run(
                run_id=run_id,
                mode=self.mode,
                start_time=start_datetime,
                end_time=end_datetime,
                success=success,
                steps_completed=steps_completed,
                error_message=error_message,
                metrics=metrics,
                data_stats=data_stats
            )

            return success

        except Exception as e:
            error_message = str(e)
            logger.error(f"Pipeline execution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Log failed pipeline run
            end_datetime = datetime.now()
            pipeline_logger.log_pipeline_run(
                run_id=run_id,
                mode=self.mode,
                start_time=start_datetime,
                end_time=end_datetime,
                success=False,
                steps_completed=steps_completed,
                error_message=error_message,
                metrics=metrics,
                data_stats=data_stats
            )

            return False

    def _run_supervised_pipeline(self, processed_data, metrics=None):
        """
        Run supervised classification pipeline.
        """
        if metrics is None:
            metrics = {}

        try:
            # Check if we have labels for training
            if 'label' not in processed_data.columns:
                logger.warning("No labels found. Switching to prediction mode with existing model.")
                # Load existing model for prediction
                model = load_existing_model()
                if model is None:
                    logger.error("No existing model found for prediction.")
                    return False

                # Make predictions
                predictions = predict_and_evaluate(model, processed_data)
                processed_data['prediction'] = predictions

            else:
                # Training mode
                logger.info("Step 3: Model Training")
                model = train_model(processed_data)
                if model is None:
                    logger.error("Model training failed.")
                    return False

                # Step 4: Evaluation
                logger.info("Step 4: Model Evaluation")
                eval_results = predict_and_evaluate(processed_data)
                if eval_results:
                    accuracy = eval_results['metrics']['accuracy']
                    logger.info(f"Model Accuracy: {accuracy:.4f}")

                    # Store metrics for logging
                    metrics.update(eval_results['metrics'])

                    # Add predictions to processed_data
                    processed_data['prediction'] = eval_results['predictions']
                    if eval_results['probabilities'] is not None:
                        processed_data['prediction_proba'] = eval_results['probabilities'].max(axis=1) if eval_results['probabilities'].ndim > 1 else eval_results['probabilities']

                    # Record performance for monitoring
                    if 'label' in processed_data.columns and 'prediction' in processed_data.columns:
                        monitor.record_performance(processed_data['label'], processed_data['prediction'])
                else:
                    logger.error("Evaluation failed.")
                    return False

            # Save predictions
            logger.info("Step 5: Saving Predictions")
            save_predictions(processed_data)

            return True

        except Exception as e:
            logger.error(f"Supervised pipeline failed: {e}")
            return False

    def _run_unsupervised_pipeline(self, processed_data):
        """
        Run unsupervised anomaly detection pipeline.
        """
        try:
            # Step 3: Anomaly Detection
            logger.info("Step 3: Unsupervised Anomaly Detection")
            processed_data = detect_anomalies_isolation_forest(processed_data)

            if processed_data is None:
                logger.error("Anomaly detection failed.")
                return False

            # Step 4: Save Predictions
            logger.info("Step 4: Saving Predictions")
            save_predictions(processed_data)

            return True

        except Exception as e:
            logger.error(f"Unsupervised pipeline failed: {e}")
            return False

    def start_automated_runs(self):
        """
        Start automated pipeline runs at specified intervals.
        """
        if self.is_running:
            logger.warning("Automated pipeline is already running.")
            return

        logger.info(f"Starting automated pipeline runs every {self.schedule_interval} hours.")

        # Add job to scheduler
        self.scheduler.add_job(
            func=self.run_pipeline,
            trigger="interval",
            hours=self.schedule_interval,
            id='ml_pipeline_job',
            name='Automated ML Pipeline'
        )

        # Start scheduler
        self.scheduler.start()
        self.is_running = True
        logger.info("Automated pipeline scheduler started.")

    def stop_automated_runs(self):
        """
        Stop automated pipeline runs.
        """
        if not self.is_running:
            logger.warning("Automated pipeline is not running.")
            return

        logger.info("Stopping automated pipeline runs.")
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        logger.info("Automated pipeline scheduler stopped.")

    def run_once(self):
        """
        Run the pipeline once manually.
        """
        return self.run_pipeline()

    def cleanup(self):
        """
        Cleanup resources on exit.
        """
        if self.is_running:
            self.stop_automated_runs()

def main():
    """
    Main function to run the automated pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Automated ML Pipeline for Network Traffic Analysis')
    parser.add_argument('--mode', choices=['supervised', 'unsupervised'],
                       default='supervised', help='Pipeline mode')
    parser.add_argument('--schedule', type=int, default=24,
                       help='Schedule interval in hours (0 for manual run)')
    parser.add_argument('--run-once', action='store_true',
                       help='Run pipeline once and exit')

    args = parser.parse_args()

    pipeline = AutomatedMLPipeline(mode=args.mode, schedule_interval_hours=args.schedule)

    if args.run_once:
        # Run once and exit
        success = pipeline.run_once()
        exit(0 if success else 1)
    else:
        # Start automated runs
        pipeline.start_automated_runs()

        try:
            # Keep the script running
            while True:
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Stopping automated pipeline.")
            pipeline.stop_automated_runs()

if __name__ == "__main__":
    main()
