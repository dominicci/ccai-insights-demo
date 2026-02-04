
import os
import json
import glob
from collections import Counter
from datetime import datetime
import sys

def visualize():
    data_dir = "data/synthetic_transcripts/velofit/storyboard_validation_jan28"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Scanning {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    print(f"Found {len(files)} files.")

    daily_counts = Counter()
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                # Try to find timestamp in root or entries
                # Based on generate script, it might be in 'call_metadata' or root.
                # Logic: assemble_call_data puts it in... I'll check the file content or infer.
                # Assuming 'start_timestamp' (RFC3339) or 'entries[0].start_timestamp'
                
                # Check known common patterns
                timestamp_str = None
                if "start_timestamp" in data:
                    timestamp_str = data["start_timestamp"] # e.g. "2025-12-01T10:00:00Z"
                elif "entries" in data and len(data["entries"]) > 0:
                     # Check first entry
                     ent = data["entries"][0]
                     if "start_timestamp" in ent:
                         timestamp_str = ent["start_timestamp"]
                
                if timestamp_str:
                    # Parse ISO format
                    # 2025-12-01T10:00:00.000000 or similar
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    day_str = dt.strftime("%Y-%m-%d")
                    daily_counts[day_str] += 1
                else:
                    # Fallback to usec
                    timestamp_usec = None
                    if "entries" in data and len(data["entries"]) > 0:
                         ent = data["entries"][0]
                         if "start_timestamp_usec" in ent:
                             timestamp_usec = ent["start_timestamp_usec"]
                    
                    if timestamp_usec:
                        dt = datetime.fromtimestamp(timestamp_usec / 1_000_000)
                        day_str = dt.strftime("%Y-%m-%d")
                        daily_counts[day_str] += 1
                        
        except Exception as e:
            # print(f"Error reading {fpath}: {e}")
            pass

    # Sort by date
    sorted_days = sorted(daily_counts.keys())
    
    if not sorted_days:
        print("No valid data found to visualize.")
        return

    print("\n--- Daily Call Volume ---")
    print(f"{'Date':<12} | {'Count':<6} | {'Distribution':<20}")
    print("-" * 45)
    
    max_count = max(daily_counts.values()) if daily_counts else 0
    
    total = sum(daily_counts.values())
    
    for day in sorted_days:
        count = daily_counts[day]
        bar_len = int((count / max_count) * 20) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{day:<12} | {count:<6} | {bar}")
        
    print("-" * 45)
    print(f"Total Files: {total}")

if __name__ == "__main__":
    visualize()
