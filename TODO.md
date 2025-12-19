# Network Traffic Analysis Pipeline TODO

## 1. Data Ingestion & Preprocessing
- [x] Enhance csv_loader.py to load CSV with semicolon separator
- [x] Standardize column names
- [x] Parse timestamps (TSesStart, TSesEnd)
- [x] Handle missing values
- [x] Compute derived features (packet rates, bytes per session, session duration)

## 2. Database Setup
- [x] Create/modify tables for raw network traffic in MySQL
- [x] Create/modify tables for predictions/anomalies in MySQL

## 3. Insert Preprocessed Data
- [x] Load preprocessed data into MySQL tables

## 4. Unsupervised Anomaly Detection
- [x] Integrate Isolation Forest for anomaly detection
- [x] Add is_anomaly column (1=anomaly, 0=normal)
- [x] Save anomaly-labeled data to DB

## 5. Supervised Learning Preparation
- [x] Train supervised models (Random Forest, XGBoost) on anomaly-labeled data
- [x] Evaluate with precision, recall, F1-score, AUC
- [x] Save supervised predictions to DB

## 6. Integration with Metabase
- [x] Create script to prepare data for Metabase dashboards
- [x] Enable visualization of anomalies, session stats, traffic metrics

## 7. Automation & Modularity
- [x] Create main_pipeline.py script for dynamic CSV handling
- [x] Modular functions for each step

## 8. Logging & Error Handling
- [x] Implement logging for all steps
- [x] Add error handling for missing files, DB issues, model failures

## 9. Python-Only Execution
- [x] Ensure entire pipeline runs in Python only
