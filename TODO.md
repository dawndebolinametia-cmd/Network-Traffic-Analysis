# ML Automation Project TODO

## Completed Tasks
- [x] Analyze existing code and logs
- [x] Identify MySQL bug: Schema mismatch in prediction_output.py
- [x] Create plan for finalizing ML project

## Pending Tasks
- [ ] Update data_ingestion.py: Connect to MySQL, fetch data, return DataFrame with error handling
- [ ] Update preprocessing.py: Automate missing values, encoding, normalization
- [ ] Update model_training.py: Train baseline (LogisticRegression) and XGBoost, compare, save best model
- [ ] Rename prediction_evaluation.py to evaluation.py and update for metrics, save to reports/
- [ ] Update automated_pipeline.py: Chain all steps with logging and error handling
- [ ] Create scheduler.py: Use APScheduler for scheduling pipeline runs
- [ ] Create logging_config.py: Standard logging configuration with rotation
- [ ] Create report_template.md: Auto-generate daily reports
- [ ] Update requirements.txt: Add missing packages
- [ ] Test end-to-end pipeline
- [ ] Ensure all scripts run independently
