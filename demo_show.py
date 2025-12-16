import pandas as pd
from prediction_output import save_predictions

def main():
    # Number of dummy rows
    num_rows = 1000

    # Create dummy predictions DataFrame
    predictions_df = pd.DataFrame({
        "log_time": pd.date_range(start="2025-12-15 08:00", periods=num_rows, freq="h"),
        "request_method": ["GET"] * num_rows,
        "response_code": [200] * num_rows,
        "bytes_sent": [1024] * num_rows
    })

    # Save predictions to CSV (DB saving disabled by default)
    save_predictions(predictions_df)

    print(f"Demo predictions saved successfully. Total rows: {num_rows}")

if __name__ == "__main__":
    main()
