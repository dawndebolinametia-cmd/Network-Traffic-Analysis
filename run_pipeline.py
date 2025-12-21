#!/usr/bin/env python3
"""
Network Traffic Analysis Pipeline Runner

This script automates the end-to-end ML pipeline for network traffic analysis.
It executes all steps in sequence and stops on any failure.

Pipeline steps:
1. Data ingestion (data_ingestion.py)
2. Anomaly detection (anomaly_detection.py)
3. Supervised learning (supervised_learning.py)
4. Summary generation (summary_generation.py)
5. Export results (export_results.py)

Usage: python run_pipeline.py
"""

import subprocess
import sys
import os
import traceback
from pathlib import Path

def run_step(script_name, description):
    """Run a pipeline step and check for success."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print('='*60)

    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent
        script_path = script_dir / script_name

        if not script_path.exists():
            print(f"ERROR: {script_name} not found at {script_path}")
            return False

        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=script_dir
        )

        # Print output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print(f"SUCCESS: {description} completed")
            return True
        else:
            print(f"FAILED: {description} failed with return code {result.returncode}")
            return False

    except Exception as e:
        print(f"ERROR: Failed to run {script_name}: {e}")
        return False

def main():
    """Main pipeline execution."""
    print("NETWORK TRAFFIC ANALYSIS PIPELINE")
    print("==================================")

    # Define pipeline steps
    steps = [
        ("data_ingestion.py", "Data Ingestion - Load CSV and create MySQL tables"),
        ("anomaly_detection.py", "Anomaly Detection - Train Isolation Forest and create anomaly_results"),
        ("supervised_learning.py", "Supervised Learning - Train Random Forest and create supervised_predictions"),
        ("summary_generation.py", "Summary Generation - Create summary_stats table"),
        ("export_results.py", "Export Results - Generate CSV exports")
    ]

    # Execute each step
    for script, description in steps:
        success = run_step(script, description)
        if not success:
            print(f"\n{'='*60}")
            print("PIPELINE FAILED")
            print(f"Failed at step: {description}")
            print("Stopping execution.")
            print('='*60)
            sys.exit(1)

    # All steps completed successfully
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("All steps executed without errors.")
    print("Results:")
    print("- network_traffic table created and populated")
    print("- anomaly_results table created")
    print("- supervised_predictions table created")
    print("- summary_stats table created")
    print("- anomaly_results.csv exported")
    print("- supervised_predictions.csv exported")
    print('='*60)

if __name__ == "__main__":
    main()
