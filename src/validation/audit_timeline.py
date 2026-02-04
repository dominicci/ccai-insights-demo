
import os
import json
import glob
import re
from datetime import datetime

# Pulse Launched Dec 15, 2025
LAUNCH_DATE = datetime(2025, 12, 15).date()

import argparse

def audit_timelines():
    parser = argparse.ArgumentParser(description="Audit logical timelines in transcripts")
    parser.add_argument("--data_dir", type=str, default="data/synthetic_transcripts/velofit/mini_pilot_v2", help="Directory to audit")
    args = parser.parse_args()

    data_dir = args.data_dir
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    print(f"Auditing {len(files)} files in '{data_dir}' for Timeline Logic...")
    
    violations = []
    pulse_files_checked = 0
    
    # Regex to catch "X days ago"
    r_days = re.compile(r"(\d+)\s+days\s+ago", re.IGNORECASE)
    
    for fpath in files:
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
                
                # Handle missing Conversation Info or Metadata safely
                if "conversation_info" not in data or "metadata" not in data["conversation_info"]:
                     continue
                     
                meta = data["conversation_info"]["metadata"]
                
                # Only care about Pulse items
                if "VeloBand Pulse" not in meta.get("product_item", ""):
                    continue
                    
                pulse_files_checked += 1
                
                # Get Call Date
                timestamp_str = None
                if "start_timestamp" in data:
                     timestamp_str = data["start_timestamp"]
                elif "entries" in data and len(data["entries"]) > 0:
                     timestamp_str = data["entries"][0].get("start_timestamp")
                     # Fallback to calculate from usec if missing string
                     if not timestamp_str and "start_timestamp_usec" in data["entries"][0]:
                         ts = data["entries"][0]["start_timestamp_usec"] / 1_000_000
                         timestamp_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

                if not timestamp_str:
                    continue
                    
                # Handle iso format
                try:
                    call_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).date()
                except:
                   # Try simple YYYY-MM-DD
                   try:
                       call_dt = datetime.strptime(timestamp_str[:10], "%Y-%m-%d").date()
                   except:
                       continue
                
                if call_dt < LAUNCH_DATE:
                    # Pre-launch calls shouldn't happen for Pulse, but if they do, they are violations
                    violations.append(f"{os.path.basename(fpath)}: Call Date {call_dt} is before Launch {LAUNCH_DATE}")
                    continue
                    
                max_possible_days = (call_dt - LAUNCH_DATE).days
                
                # Scan text for mentions
                full_text = " ".join([e.get("text", "") for e in data["entries"]])
                matches = r_days.findall(full_text)
                
                for m in matches:
                    days_mentioned = int(m)
                    if days_mentioned > max_possible_days:
                        violations.append(f"{os.path.basename(fpath)}: Mentioned '{days_mentioned} days ago' on {call_dt}. Max possible: {max_possible_days}")
                        break # One violation per file is enough
                        
            except Exception as e:
                # print(f"Error reading {fpath}: {e}")
                pass
                
    print(f"\nChecked {pulse_files_checked} Pulse files.")
    print(f"Found {len(violations)} Violations.")
    
    if violations:
        print("\nVIOLATIONS MATCHED:")
        for v in violations[:20]:
            print(v)
        if len(violations) > 20:
            print(f"... and {len(violations)-20} more.")

if __name__ == "__main__":
    audit_timelines()
