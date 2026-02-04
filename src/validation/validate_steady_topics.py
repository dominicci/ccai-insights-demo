import os
import json
import glob
from collections import Counter

def validate_topics():
    data_dir = "data/upload_batches/01_steady_state"
    print(f"Scanning {data_dir} for Topic Keywords...")
    
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    topics = {
        "Product Inquiries": ["size", "frame", "height", "spec", "road series", "mountain series", "choose", "difference"],
        "Shipping & Tracking": ["order", "arrive", "track", "shipment", "delivery", "late", "where is my"],
        "Subscription & Billing": ["subscription", "membership", "charged", "credit card", "payment", "cancel", "fee"],
        "Standard Returns": ["return", "90 days", "ninety days", "policy", "refund", "process a return"],
        "Technical Support": ["app", "sync", "log within", "login", "connect", "device", "bluetooth", "error"]
    }
    
    topic_counts = Counter()
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Get full text
            full_text = " ".join([e["text"] for e in data["entries"]]).lower()
            
            # Check for matches
            # A file can match multiple topics (rare but possible)
            matched = False
            for topic, keywords in topics.items():
                if any(k in full_text for k in keywords):
                    topic_counts[topic] += 1
                    matched = True
                    
        except Exception:
            pass
            
    print(f"\n--- Topic Validation Results (N={len(files)}) ---")
    for topic, count in topic_counts.most_common():
        pct = (count / len(files)) * 100
        print(f"{topic}: {count} calls ({pct:.1f}%)")
        
    # Check if any have 0
    missing = [t for t in topics if topic_counts[t] < 5] # Threshold of 5
    if missing:
        print(f"\nWARNING: Low volume topics: {missing}")
    else:
        print("\nAll topics are well-represented.")

if __name__ == "__main__":
    validate_topics()
