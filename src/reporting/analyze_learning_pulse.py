
import os
import json
import glob
import re
from collections import Counter
from datetime import datetime
import sys

def analyze_learning_pulse():
    data_dir = "data/synthetic_transcripts/velofit/full_scale_run"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Scanning {data_dir} for Learning Phase (Jan 8+) Pulse Calls...")
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    pulse_calls = []
    outcomes = Counter()
    policy_mentions = {"10-day": 0, "90-day": 0, "both": 0, "neither": 0}
    
    # Date Cutoff
    LEARNING_START = datetime(2026, 1, 8).date()
    
    # Regex
    r_90 = re.compile(r"90\s*(-)?\s*days?", re.IGNORECASE)
    r_10 = re.compile(r"10\s*(-)?\s*days?", re.IGNORECASE)

    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
                # Check Metadata first
                if "conversation_info" not in data or "metadata" not in data["conversation_info"]:
                    continue
                
                meta = data["conversation_info"]["metadata"]
                prod_cat = meta.get("product_category", "").lower()
                prod_item = meta.get("product_item", "").lower()
                
                # Filter Pulse
                if prod_cat != "band" and "pulse" not in prod_item:
                    continue
                    
                # Filter Date
                timestamp_str = None
                if "start_timestamp" in data:
                    timestamp_str = data["start_timestamp"]
                elif "entries" in data:
                     ent = data["entries"][0]
                     if "start_timestamp" in ent:
                         timestamp_str = ent["start_timestamp"]
                     elif "start_timestamp_usec" in ent:
                         ts = ent["start_timestamp_usec"] / 1_000_000
                         timestamp_str = datetime.fromtimestamp(ts).isoformat()
                         
                if not timestamp_str:
                    continue
                    
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).date()
                except:
                    dt = datetime.strptime(timestamp_str[:10], "%Y-%m-%d").date()
                    
                if dt < LEARNING_START:
                    continue
                    
                # It is a Learning Phase Pulse Call
                pulse_calls.append(fpath)
                
                # 1. Outcome Analysis
                outcome = meta.get("outcome", "Unknown")
                outcomes[outcome] += 1
                
                # 2. Policy Mention Analysis
                full_text = " ".join([e["text"] for e in data["entries"]])
                has_90 = bool(r_90.search(full_text))
                has_10 = bool(r_10.search(full_text))
                
                if has_90 and has_10:
                    policy_mentions["both"] += 1
                elif has_90:
                    policy_mentions["90-day"] += 1
                elif has_10:
                    policy_mentions["10-day"] += 1
                    print(f"[Correct Enforcement Candidate]: {fpath}")
                else:
                    policy_mentions["neither"] += 1

        except Exception as e:
            pass

    total = len(pulse_calls)
    if total == 0:
        print("No Learning Phase Pulse calls found.")
        return

    print(f"\n--- Learning Phase (Jan 8+) Pulse Analysis ---")
    print(f"Total Pulse Calls: {total}")
    
    print(f"\nResolution Status:")
    print("-" * 30)
    for k, v in outcomes.most_common():
        pct = (v / total) * 100
        print(f"{k:<20} | {v:<5} | {pct:.1f}%")
        
    print(f"\nPolicy Keyword Analysis:")
    print("-" * 30)
    print(f"{'Mentioned 90-Day':<20} | {policy_mentions['90-day']:<5} | {(policy_mentions['90-day']/total)*100:.1f}%")
    print(f"{'Mentioned 10-Day':<20} | {policy_mentions['10-day']:<5} | {(policy_mentions['10-day']/total)*100:.1f}%")
    print(f"{'Mentioned Both':<20} | {policy_mentions['both']:<5} | {(policy_mentions['both']/total)*100:.1f}%")
    print(f"{'Mentioned Neither':<20} | {policy_mentions['neither']:<5} | {(policy_mentions['neither']/total)*100:.1f}%")
    
    # Insight Check
    compliance_miss_proxy = policy_mentions['90-day'] + policy_mentions['both']
    print(f"\n[Insight] Potential Compliance Misses (referenced 90-day): {compliance_miss_proxy} ({compliance_miss_proxy/total*100:.1f}%)")

if __name__ == "__main__":
    analyze_learning_pulse()
