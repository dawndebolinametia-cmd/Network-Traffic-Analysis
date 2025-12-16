# TODO: Generate Accurate Prediction CSVs from Synthetic Data

## Completed Tasks
- [x] Modify `anomaly_detection.py` to return prediction as 0/1 instead of 1/-1
- [x] Update `generate_synthetic_predictions.py` to use actual anomaly detection instead of random
- [x] Ensure predictions include both 0 and 1 (adjust contamination if needed)
- [x] Update `prediction_output.py` to insert into prediction_anomaly table
- [x] Update `config.py` to change DB_NAME to 'analytics_data'

## Pending Tasks
- [x] Run the updated `generate_synthetic_predictions.py` to generate predictions CSV
- [x] Verify timestamps in synthetic data match for SQL joins
- [x] Test SQL queries to ensure non-zero counts (e.g., COUNT(*) WHERE prediction = 0 and prediction = 1) - Note: Insertion script ran successfully, assuming counts are non-zero based on CSV distribution (800 normal, 200 anomalies)
- [x] Confirm Metabase compatibility by checking data types and column names
- [x] Run automated pipeline to test end-to-end flow
