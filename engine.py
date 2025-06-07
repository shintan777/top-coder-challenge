# engine.py

import pickle
import pandas as pd
import numpy as np
import sys
import joblib

def compute(trip_days, miles, receipts):
    with open('linear_model.pkl', 'rb') as f:
        model = pickle.load(f)

    row = pd.DataFrame([{
        'trip_duration_days': trip_days,
        'miles_traveled': miles,
        'total_receipts_amount': receipts
    }])

    row['trip_category'] = 'short_≤7' if trip_days <= 7 else 'long_>7'

    def receipt_category(r):
        if r <= 500:
            return 'short_≤500'
        elif r <= 1250:
            return 'medium_500-1250'
        else:
            return 'long_≥1250'

    row['receipt_category'] = receipt_category(receipts)
    row['miles_per_day'] = miles / trip_days
    row['receipt_per_day'] = receipts / trip_days
    row['triplen_receipt_per_mile'] = trip_days * (receipts / (miles if miles != 0 else 1))
    row['spend_per_day_times_miles_per_day'] = row['receipt_per_day'] * row['miles_per_day']
    row['rounding_flag'] = int(round(receipts % 1, 2) in [0.49, 0.99])

    row = row.drop(columns=['trip_len_cat'], errors='ignore')

    # One-hot encode with same columns as training
    row_encoded = pd.get_dummies(row, drop_first=True)

    # Align columns with training
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[model_features]

    result = round(model.predict(row_encoded)[0], 2)
    return result



    
xgb_model = joblib.load('xgb_base.pkl')
residual_model = joblib.load('residual_tree_model.pkl')

def compute_tree(trip_days, miles, receipts):
    X = np.array([[trip_days, miles, receipts]])
    base_pred = xgb_model.predict(X) 
    resid_corr = residual_model.predict(X)
    final =  float(base_pred[0] + resid_corr[0])
    return round(final, 2)

if __name__ == "__main__":
    trip_days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    result = compute(trip_days, miles, receipts)
    print(result)
