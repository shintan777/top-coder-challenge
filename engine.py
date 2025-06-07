# engine.py

import pickle
import pandas as pd
import numpy as np
import sys
import joblib
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

class LegacyReimbursementTreeRegressorWithNewComponents:
    def fit(self, X, y=None):
        return self

    def get_components(self, X):
        X = np.asarray(X)
        per_diem_list = []
        mileage_list = []
        receipt_adj_list = []
        bonus_list = []

        for row in X:
            days, miles, receipts = row
            per_diem = 100 * days
            bonus = 0

            # Calculate derived metrics
            miles_per_day = miles / days if days > 0 else 0
            spend_per_day = receipts / days if days > 0 else 0

            # Core per diem adjustment
            if days == 5:
                bonus += 50
            elif 4 <= days <= 6:
                bonus += 20  # sweet spot range
            elif days >= 8 and spend_per_day > 90:
                bonus -= 60  # vacation penalty

            # Mileage rules with sweet spot reward
            if miles_per_day < 50:
                mileage = 0.40 * miles
                bonus -= 15
            elif 180 <= miles_per_day <= 220:
                mileage = 0.58 * miles
                bonus += 60  # efficiency sweet spot
            elif miles_per_day > 300:
                mileage = 0.35 * miles
                bonus -= 30  # over-efficiency penalty
            else:
                mileage = 0.45 * miles

            # Spend-per-day rules
            if days <= 3 and spend_per_day > 75:
                bonus -= 30
            elif 4 <= days <= 6 and spend_per_day > 120:
                bonus -= 30
            elif days > 6 and spend_per_day > 90:
                bonus -= 40

            # Sweet spot combo
            if (days == 5) and (180 <= miles_per_day) and (spend_per_day < 100):
                bonus += 80

            # Receipt adjustment logic
            if receipts < 50:
                bonus -= 25
                receipt_adj = receipts * 0.5
            elif receipts <= 800:
                receipt_adj = receipts * 0.9
            elif receipts <= 1200:
                receipt_adj = 800 * 0.9 + (receipts - 800) * 0.5
            else:
                receipt_adj = 800 * 0.9 + 400 * 0.5 + (receipts - 1200) * 0.2

            # Rounding bonus
            cents = round(receipts, 2) % 1
            if np.isclose(cents, 0.49, atol=0.01) or np.isclose(cents, 0.99, atol=0.01):
                bonus += 10

            per_diem_list.append(per_diem)
            mileage_list.append(mileage)
            receipt_adj_list.append(receipt_adj)
            bonus_list.append(bonus)

        return (
            np.array(per_diem_list),
            np.array(mileage_list),
            np.array(receipt_adj_list),
            np.array(bonus_list)
        )

class ResidualTreeModel(LegacyReimbursementTreeRegressorWithNewComponents):
    def fit(self, X, residuals):
        self.X_train = np.asarray(X)
        self.residuals = np.asarray(residuals)

        per_diem, mileage, receipt_adj, bonus = self.get_components(self.X_train)

        self.component_features = np.vstack([per_diem, mileage, receipt_adj, bonus]).T

        X = sm.add_constant(self.component_features)
        y = self.residuals

        self.model = QuantReg(y, X).fit(q=0.5)

        return self

    def predict(self, X):
        per_diem, mileage, receipt_adj, bonus = self.get_components(X)
        features = np.vstack([per_diem, mileage, receipt_adj, bonus]).T

        X_new = sm.add_constant(features)
        y_pred = self.model.predict(X_new)

        return y_pred

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
