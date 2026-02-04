import os
import json
import shutil
import glob
from datetime import datetime

def setup_test_data():
    source_dir = "data/synthetic_transcripts/velofit/full_scale_run"
    dest_dir = "data/learning_amplification_test"
    
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    print(f"Scanning {source_dir}...")
    files = glob.glob(os.path.join(source_dir, "**/*.json"), recursive=True)
    
    count = 0
    target_count = 300
    
    # Target: Pulse items after Jan 8
    
    for fpath in files:
        if count >= target_count:
            break
            
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            if "conversation_info" not in data or "metadata" not in data["conversation_info"]:
                continue
                
            meta = data["conversation_info"]["metadata"]
            prod_item = meta.get("product_item", "").lower()
            
            if "pulse" not in prod_item and meta.get("product_category") != "band":
                continue
                
            # Check Date
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
                
            # Filter for Learning Phase (Jan 8 - Jan 30)
            if datetime(2026, 1, 8).date() <= dt <= datetime(2026, 1, 30).date():
                # COPY
                dest_path = os.path.join(dest_dir, os.path.basename(fpath))
                shutil.copy(fpath, dest_path)
                count += 1
                
        except Exception as e:
            pass
            
    print(f"Copied {count} Learning Phase Pulse calls to {dest_dir}")

if __name__ == "__main__":
    setup_test_data()
