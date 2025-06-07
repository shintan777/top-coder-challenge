# engine.py

import sys

def compute(trip_days, miles, receipts):
    # Base per diem: $100/day, bonus for 5-day trips
    per_diem = 100 * trip_days
    if trip_days == 5:
        per_diem += 25

    # Tiered mileage reimbursement
    if miles <= 100:
        mileage_reimb = miles * 0.58
    else:
        mileage_reimb = 100 * 0.58 + (miles - 100) * 0.45

    # Receipts handling
    if receipts < 50:
        receipt_bonus = 0
    elif receipts < 800:
        receipt_bonus = receipts * 0.75
    elif receipts <= 1000:
        receipt_bonus = 600 + (receipts - 800) * 0.5
    else:
        receipt_bonus = 700 + (receipts - 1000) * 0.1

    # Rounding bug bonus
    if str(receipts).endswith("0.49") or str(receipts).endswith("0.99"):
        receipt_bonus += 0.02

    total = per_diem + mileage_reimb + receipt_bonus
    return round(total, 2)

if __name__ == "__main__":
    trip_days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])
    result = compute(trip_days, miles, receipts)
    print(result)
