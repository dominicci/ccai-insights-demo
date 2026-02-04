
import os
import json
import glob
import re
from collections import defaultdict
from datetime import datetime
import sys

def aggregate_stats():
    data_dir = "data/synthetic_transcripts/velofit/full_scale_run"
    output_file = "data/dashboard_data.json"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Scanning {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    # Structure: date -> metrics
    daily_stats = defaultdict(lambda: {
        "standard": 0,
        "pulse_total": 0,
        "pulse_correct": 0,
        "pulse_incorrect": 0,
        "pulse_neutral": 0
    })
    
    # Regex for Policy
    r_90 = re.compile(r"90\s*(-)?\s*days?", re.IGNORECASE)
    r_10 = re.compile(r"10\s*(-)?\s*days?", re.IGNORECASE)

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                if "conversation_info" not in data or "metadata" not in data["conversation_info"]:
                    continue
                
                meta = data["conversation_info"]["metadata"]
                prod_cat = meta.get("product_category", "").lower()
                prod_item = meta.get("product_item", "").lower()
                
                # Determine Date
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
                         
                if not timestamp_str:
                    continue
                    
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                except:
                    dt = datetime.strptime(timestamp_str[:10], "%Y-%m-%d")
                
                day_str = dt.strftime("%Y-%m-%d")
                
                # Categorize
                if prod_cat == "band" or "pulse" in prod_item:
                    # Pulse Item
                    daily_stats[day_str]["pulse_total"] += 1
                    
                    # Policy Check
                    full_text = " ".join([e["text"] for e in data["entries"]])
                    has_90 = bool(r_90.search(full_text))
                    has_10 = bool(r_10.search(full_text))
                    
                    if has_10:
                        daily_stats[day_str]["pulse_correct"] += 1
                    elif has_90:
                        daily_stats[day_str]["pulse_incorrect"] += 1
                    else:
                        daily_stats[day_str]["pulse_neutral"] += 1
                        
                else:
                    # Standard Item
                    daily_stats[day_str]["standard"] += 1

        except Exception as e:
            pass

    # Convert to sorted list
    sorted_data = []
    for day in sorted(daily_stats.keys()):
        stats = daily_stats[day]
        stats["date"] = day
        sorted_data.append(stats)
        
    # Stats Summary
    total_calls = sum(d["standard"] + d["pulse_total"] for d in sorted_data)
    print(f"Aggregated {total_calls} calls across {len(sorted_data)} days.")
    
    with open(output_file, 'w') as f:
        json.dump(sorted_data, f, indent=2)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    aggregate_stats()
