# ML Pipeline Daily Report

**Report Date:** {date}  
**Pipeline Run ID:** {run_id}  
**Mode:** {mode}  
**Execution Time:** {execution_time:.2f} seconds  
**Success:** {success}

## Summary
- **Steps Completed:** {steps_completed}
- **Data Processed:** {data_rows} rows, {data_columns} columns
- **Error Message:** {error_message}

## Model Performance Metrics
- **Accuracy:** {accuracy:.4f}
- **Precision:** {precision:.4f}
- **Recall:** {recall:.4f}
- **F1 Score:** {f1_score:.4f}

## Data Statistics
- **Raw Data:** {raw_data_rows} rows, {raw_data_columns} columns
- **Processed Data:** {processed_data_rows} rows, {processed_data_columns} columns
- **Columns:** {data_columns_list}

## Anomalies Detected
- **Total Anomalies:** {anomaly_count}
- **Anomaly Percentage:** {anomaly_percentage:.2f}%

## Visualizations
- [View Dashboard]({dashboard_url})
- [Download Predictions CSV](reports/predictions.csv)
- [View Evaluation Report](reports/evaluation_results.txt)

## Logs
- [View Full Logs](logs/ml_pipeline.log)

---
*Generated automatically by ML Pipeline System*
