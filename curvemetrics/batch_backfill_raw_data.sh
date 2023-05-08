#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 start_date end_date"
  exit 1
fi

start_date="$1"
end_date="$2"

# Convert dates to seconds since the epoch
start_sec=$(date -d "$start_date" +%s)
end_sec=$(date -d "$end_date" +%s)

# Calculate the number of seconds per day and 5 minutes
day_sec=$((24 * 60 * 60))
overlap_sec=$((5 * 60))

# Loop through daily batches
current_sec=$start_sec
next_sec=$((current_sec + day_sec))
while [ $current_sec -lt $end_sec ]; do

  # Convert seconds to date string format
  current_date=$(date -d "@$current_sec" "+%Y-%m-%d %H:%M:%S")
  next_date=$(date -d "@$next_sec" "+%Y-%m-%d %H:%M:%S")

  # Execute the Python script for the current batch
  echo "[`date "+%Y-%m-%d %H:%M:%S"`] Running task for $current_date to $next_date"
  python3 -u -m curvemetrics.backfill_raw_data "$current_date" "$next_date" > "out_$(date -d "@$current_sec" "+%Y-%m-%d").log" 2>&1
  echo "[`date "+%Y-%m-%d %H:%M:%S"`] Task completed for $current_date to $next_date"

  # Move to the next batch, subtracting the 5-minute overlap
  current_sec=$((next_sec - overlap_sec))
  current_sec=$(date -d "$(date -d "@$current_sec" "+%Y-%m-%d") 23:55:00" +%s)
  next_sec=$((current_sec + day_sec + overlap_sec))
done