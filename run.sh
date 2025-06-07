#!/bin/bash

# Validate argument count
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    exit 1
fi

# Extract arguments
trip_days=$1
miles=$2
receipts=$3

# Call the Python logic
python3 engine.py "$trip_days" "$miles" "$receipts"
