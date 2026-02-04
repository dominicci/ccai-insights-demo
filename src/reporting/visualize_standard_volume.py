
import os
import json
import glob
from collections import Counter
from datetime import datetime
import sys

def visualize_standard_volume():
    data_dir = "data/synthetic_transcripts/velofit/full_scale_run"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Scanning {data_dir} for Standard Item Calls...")
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    daily_counts = Counter()
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                # Filter for Standard Items
                is_standard = False
                if "conversation_info" in data and "metadata" in data["conversation_info"]:
                    meta = data["conversation_info"]["metadata"]
                    prod_cat = meta.get("product_category", "").lower()
                    prod_item = meta.get("product_item", "").lower()
                    
                    if prod_cat != "band" and "pulse" not in prod_item:
                        is_standard = True
                
                if not is_standard:
                    continue

                # Get Date
                timestamp_str = None
                if "start_timestamp" in data:
                    timestamp_str = data["start_timestamp"]
                elif "entries" in data and len(data["entries"]) > 0:
                     ent = data["entries"][0]
                     if "start_timestamp" in ent:
                         timestamp_str = ent["start_timestamp"]
                     elif "start_timestamp_usec" in ent:
                         ts = ent["start_timestamp_usec"] / 1_000_000
                         timestamp_str = datetime.fromtimestamp(ts).isoformat()
                
                if timestamp_str:
                    # Handle typical formats
                    try:
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    except:
                        dt = datetime.strptime(timestamp_str[:10], "%Y-%m-%d")
                    
                    day_str = dt.strftime("%Y-%m-%d")
                    
                    # Filter Date Range (Dec 1 to Jan 28)
                    if "2025-12-01" <= day_str <= "2026-01-28":
                        daily_counts[day_str] += 1
                        
        except Exception as e:
            pass

    # Sort and Fill Dates
    if not daily_counts:
        print("No Standard calls found in range.")
        return

    sorted_days = sorted(daily_counts.keys())
    
    print("\n--- Standard Item Call Volume (Dec 1 - Jan 28) ---")
    print(f"{'Date':<12} | {'Count':<6} | {'Volume':<20}")
    print("-" * 50)
    
    max_count = max(daily_counts.values())
    
    # Fill in missing days for cleaner plot? 
    # Or just show presence. Let's show all occurring days.
    
    total = 0
    for day in sorted_days:
        count = daily_counts[day]
        total += count
        bar_len = int((count / max_count) * 20) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"{day:<12} | {count:<6} | {bar}")
        
    print("-" * 50)
    print(f"Total Standard Calls: {total}")

if __name__ == "__main__":
    visualize_standard_volume()
