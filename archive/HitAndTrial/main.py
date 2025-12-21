import sys
import argparse
from logging_utils import logger
from automated_pipeline import AutomatedMLPipeline

def main():
    """
    Main entry point for the automated ML pipeline.
    """
    parser = argparse.ArgumentParser(description='Network Traffic Analysis ML Pipeline')
    parser.add_argument('--mode', choices=['supervised', 'unsupervised'],
                       default='supervised', help='Pipeline mode (default: supervised)')
    parser.add_argument('--schedule', type=int, default=0,
                       help='Schedule interval in hours (0 for manual run, default: 0)')
    parser.add_argument('--run-once', action='store_true',
                       help='Run pipeline once and exit (default: False)')

    args = parser.parse_args()

    logger.info("Starting Network Traffic Analysis ML Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Schedule: {args.schedule} hours")
    logger.info(f"Run once: {args.run_once}")

    # Initialize automated pipeline
    pipeline = AutomatedMLPipeline(mode=args.mode, schedule_interval_hours=args.schedule)

    if args.run_once or args.schedule == 0:
        # Run pipeline once
        logger.info("Running pipeline once...")
        success = pipeline.run_once()
        if success:
            logger.info("Pipeline completed successfully.")
            sys.exit(0)
        else:
            logger.error("Pipeline failed.")
            sys.exit(1)
    else:
        # Start automated runs
        logger.info(f"Starting automated pipeline with {args.schedule} hour intervals...")
        pipeline.start_automated_runs()

        try:
            # Keep running until interrupted
            import time
            while True:
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Received interrupt signal. Shutting down...")
            pipeline.stop_automated_runs()
            logger.info("Pipeline stopped.")
            sys.exit(0)

if __name__ == "__main__":
    main()
