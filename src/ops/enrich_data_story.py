import os
import json
import glob
import re
from datetime import datetime
import random

def enrich_data_story():
    data_dir = "data/synthetic_transcripts/velofit/full_scale_run"
    print(f"Enriching dataset in {data_dir} to align with Storyboard...")
    
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    # Regex for Policy
    r_90 = re.compile(r"90\s*(-)?\s*days?", re.IGNORECASE) # The "Old/Wrong" Policy
    r_manager = re.compile(r"manager|supervisor|complain", re.IGNORECASE) # Escalatish keywords
    
    modified_count = 0
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Helper to save changes
            should_save = False
            
            # 1. Drill down to Metadata
            if "conversation_info" not in data or "metadata" not in data["conversation_info"]:
                continue
            
            meta = data["conversation_info"]["metadata"]
            prod_item = meta.get("product_item", "").lower()
            
            # Only affect Pulse items (The Crisis)
            if "pulse" not in prod_item and meta.get("product_category") != "band":
                continue
                
            # Only affect Post-Change Era (Jan 2026)
            # (We assume file location or timestamp implies this, but let's check text pattern logic)
            
            full_text = " ".join([e["text"] for e in data["entries"]])
            has_90 = bool(r_90.search(full_text))
            
            # STORY RULE: If Agent quotes "90 days" (Wrong Policy) -> Customer gets Confused/Negative
            if has_90:
                # Flip Sentiment to Negative/Neutral to show "Confusion/Contradiction"
                # We'll make it probabilistic so it's not 100% mechanical, but highly correlated (e.g. 70%)
                if random.random() < 0.70:
                    meta["customer_sentiment"] = random.choice(["Negative", "Negative", "Neutral"])
                    
                    # Add a "Smart Label" for the "Hybrid Approach" story
                    meta["quality_flag"] = "Policy Contradiction Detected"
                    
                    should_save = True
                    modified_count += 1
            
            if should_save:
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            # print(f"Skipping {fpath}: {e}")
            pass

    print(f"Enrichment Complete. Modified {modified_count} transcripts to reflect 'Policy Contradiction' sentiment impact.")

if __name__ == "__main__":
    enrich_data_story()
