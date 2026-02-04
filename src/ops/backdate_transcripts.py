#!/usr/bin/env python3
import os
import json
import glob
import random
from datetime import datetime, timedelta

# Configuration
HISTORY_DIR = 'data/synthetic_transcripts/velofit/sorted/history_pre_change'
CURRENT_DIR = 'data/synthetic_transcripts/velofit/sorted/current_post_change'

# Date Ranges
# History: Dec 1, 2025 - Dec 31, 2025
HISTORY_START = datetime(2025, 12, 1)
HISTORY_END = datetime(2025, 12, 31)

# Current: Jan 1, 2026 - Jan 15, 2026
CURRENT_START = datetime(2026, 1, 1)
CURRENT_END = datetime(2026, 1, 15)

def random_timestamp_in_range(start_date, end_date):
    """Generates a random timestamp (microseconds) within the given date range."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + timedelta(days=random_number_of_days)
    
    # Add random time of day (0-23 hours, 0-59 minutes, 0-59 seconds)
    random_date = random_date.replace(
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )
    
    return int(random_date.timestamp() * 1000000)

def backdate_files(directory, start_date, end_date):
    print(f"Processing {directory}...")
    files = glob.glob(os.path.join(directory, '*.json'))
    count = 0
    errors = 0
    
    if not files:
        print(f"No files found in {directory}")
        return

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'entries' not in data or not data['entries']:
                continue

            # 1. Generate random base start time
            base_start_time = random_timestamp_in_range(start_date, end_date)
            
            # 2. Identify Zero Point (first entry's timestamp)
            first_entry = data['entries'][0]
            original_zero_point = first_entry.get('start_timestamp_usec', 0)
            
            # 3. Shift every entry
            for entry in data['entries']:
                original_time = entry.get('start_timestamp_usec', 0)
                offset = original_time - original_zero_point
                entry['start_timestamp_usec'] = base_start_time + offset
            
            # 4. Overwrite file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            errors += 1
            
    print(f"Updated {count} files in {directory} (Errors: {errors})")

def main():
    backdate_files(HISTORY_DIR, HISTORY_START, HISTORY_END)
    backdate_files(CURRENT_DIR, CURRENT_START, CURRENT_END)
    print("\nBackdating complete.")

if __name__ == "__main__":
    main()
