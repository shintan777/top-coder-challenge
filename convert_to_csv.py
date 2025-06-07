import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Convert JSON to CSV
with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Flatten and convert to DataFrame
flattened_data = []
for entry in data:
    row = entry["input"]
    row["expected_output"] = entry["expected_output"]
    flattened_data.append(row)

df = pd.DataFrame(flattened_data)

# Save to CSV
df.to_csv("reimbursements.csv", index=False)
