
import os
import json
import glob
from collections import Counter
from datetime import datetime
import sys

def analyze_chaos():
    data_dir = "data/synthetic_transcripts/velofit/full_scale_run"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Scanning {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    chaos_calls = []
    daily_counts = Counter()
    
    start_date = datetime(2026, 1, 1).date()
    end_date = datetime(2026, 1, 7).date()
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                # Extract Timestamp
                timestamp_str = None
                timestamp_usec = None
                
                if "start_timestamp" in data:
                    timestamp_str = data["start_timestamp"]
                elif "entries" in data and len(data["entries"]) > 0:
                     ent = data["entries"][0]
                     if "start_timestamp" in ent:
                         timestamp_str = ent["start_timestamp"]
                     if "start_timestamp_usec" in ent:
                         timestamp_usec = ent["start_timestamp_usec"]

                dt = None
                if timestamp_str:
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                elif timestamp_usec:
                    dt = datetime.fromtimestamp(timestamp_usec / 1_000_000)
                
                if dt:
                    d = dt.date()
                    if start_date <= d <= end_date:
                        daily_counts[d.strftime("%Y-%m-%d")] += 1
                        
                        call_type = "Unknown"
                        if "conversation_info" in data and "metadata" in data["conversation_info"]:
                            meta = data["conversation_info"]["metadata"]
                            call_type = meta.get("call_type", "Unknown")
                        
                        chaos_calls.append(call_type)

        except Exception as e:
            pass

    print(f"\n--- Chaos Phase (Jan 1 - Jan 7) Analysis ---")
    print(f"Total Chaos Calls: {len(chaos_calls)}")
    
    if not chaos_calls:
        print("No calls found in this range.")
        return

    print("\n--- Daily Volume ---")
    years = sorted(daily_counts.keys())
    for d in years:
        print(f"{d}: {daily_counts[d]}")
        
    counts = Counter(chaos_calls)
    total = len(chaos_calls)
    
    print(f"\n{'Call Type':<50} | {'Count':<6} | {'%':<5}")
    print("-" * 70)
    for ctype, count in counts.most_common():
        pct = (count / total) * 100
        print(f"{ctype:<50} | {count:<6} | {pct:.1f}%")

if __name__ == "__main__":
    analyze_chaos()
