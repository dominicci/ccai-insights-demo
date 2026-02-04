import os
import json
import glob
import re
import random
from datetime import datetime
import argparse

def amplify_learning_curve():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/learning_amplification_test")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    print(f"Amplifying Learning Curve in {data_dir}...")
    
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    # Constants for Logic
    START_DATE = datetime(2026, 1, 8).date()
    END_DATE = datetime(2026, 1, 30).date()
    START_PROB = 0.10 # 10% Correct at start
    END_PROB = 0.70   # 70% Correct by end
    
    total_days = (END_DATE - START_DATE).days
    
    # Regex
    r_90 = re.compile(r"90\s*(-)?\s*days?", re.IGNORECASE)
    
    modified_count = 0
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            # 1. Determine Date
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
                
            if dt < START_DATE:
                continue
                
            # 2. Calculate Target Probability for this Day
            days_elapsed = (dt - START_DATE).days
            # Clamp
            if days_elapsed < 0: days_elapsed = 0
            if days_elapsed > total_days: days_elapsed = total_days
            
            # Linear Interpolation
            progress = days_elapsed / total_days
            target_prob = START_PROB + (progress * (END_PROB - START_PROB))
            
            # 3. Apply Transformation
            # If transcript has "90 days", roll dice to flip to "10 days"
            
            original_text = json.dumps(data)
            
            if r_90.search(original_text):
                if random.random() < target_prob:
                    # Flip Logic:
                    # 1. Replace "90 days" -> "10 days"
                    # 2. Update Outcome -> "Unresolved" (Usually denial) or "Resolved" (if they accepted it)
                    # Actually, if they enforce 10-day correctly, the outcome is technically "Correct Enforement"
                    # But the *script* usually outputs "Unresolved" for Hard Denial.
                    # Let's just do text replacement first.
                    
                    # We need to be careful replacing in JSON string, might break structure?
                    # Safer to iterate entries.
                    
                    for entry in data["entries"]:
                        if r_90.search(entry["text"]):
                            # Replace 90 with 10
                            entry["text"] = re.sub(r"90(\s*(-)?\s*days?)", r"10\1", entry["text"], flags=re.IGNORECASE)
                            # Also replace "well within" with "outside" if present? 
                            # This gets tricky NLP wise. 
                            # Simple hack: Just replace the number. It might make the sentence "You are 10 days... so you are well within". 
                            # Wait, 10-day policy means return window is SHORT.
                            # If customer bought 40 days ago.
                            # Old: "Policy is 90 days. You are 40 days. You are well within."
                            # New: "Policy is 10 days. You are 40 days. You are well within." -> CONTRADICTION.
                            
                            # We need to change the *Reasoning* too.
                            # "You are well within" -> "Unfortunately, you are outside"
                            entry["text"] = entry["text"].replace("well within", "unfortunately outside")
                            entry["text"] = entry["text"].replace("can go ahead and approve", "cannot approve")
                            entry["text"] = entry["text"].replace("process a full refund", "cannot offer a refund")
                            
                            # Update Metadata Outcome if we denied it
                            data["conversation_info"]["metadata"]["outcome"] = "Unresolved" # Denial
                            data["conversation_info"]["metadata"]["quality_flag"] = "Correct Policy Enforcement (Amplified)"

                    modified_count += 1
                    
                    with open(fpath, 'w') as f:
                        json.dump(data, f, indent=2)

        except Exception as e:
            # print(f"Error {fpath}: {e}")
            pass
            
    print(f"Amplification Complete. Modified {modified_count} transcripts in {data_dir}.")

if __name__ == "__main__":
    amplify_learning_curve()
