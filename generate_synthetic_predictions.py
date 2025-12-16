import pandas as pd
import numpy as np
from datetime import datetime, timedelta

rows = 100

start_time = datetime.now() - timedelta(hours=rows)

data = {
    "log_time": [start_time + timedelta(minutes=5*i) for i in range(rows)],
    "prediction": np.random.choice([0, 1], size=rows),
    "anomaly_score": np.round(np.random.uniform(0, 1, size=rows), 3),
    "traffic_type": np.random.choice(
        ["normal", "suspicious"], size=rows, p=[0.8, 0.2]
    )
}

df = pd.DataFrame(data)

df.to_csv("reports/predictions.csv", index=False)

print("Synthetic predictions CSV generated successfully.")
