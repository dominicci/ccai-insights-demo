import os
import json
import glob
import random

def validate_topic_purity():
    data_dir = "data/upload_batches/01_steady_state"
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    random.shuffle(files)
    
    # Definitions
    bill_keywords = ["subscription", "membership", "charged", "billing", "credit card"]
    tech_keywords = ["app", "sync", "bluetooth", "login", "password", "connect", "device"]
    return_keywords = ["return", "refund", "exchange", "policy", "send back"]
    
    pure_billing = []
    pure_tech = []
    
    print(f"Scanning {len(files)} files for PURE topics (excluding '{return_keywords[0]}', etc.)...\n")
    
    for fpath in files:
        if len(pure_billing) >= 3 and len(pure_tech) >= 3:
            break
            
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            full_text = " ".join([e["text"] for e in data["entries"]]).lower()
            
            has_bill = any(k in full_text for k in bill_keywords)
            has_tech = any(k in full_text for k in tech_keywords)
            has_return = any(k in full_text for k in return_keywords)
            
            if has_bill and not has_return:
                pure_billing.append((os.path.basename(fpath), full_text[:200]))
                
            if has_tech and not has_return:
                pure_tech.append((os.path.basename(fpath), full_text[:200]))
                
        except Exception:
            pass
            
    # Report
    print(f"--- Pure Billing Calls (No Returns) ---")
    if pure_billing:
        for name, snippet in pure_billing:
            print(f"File: {name}\nSnippet: {snippet}...\n")
    else:
        print("None found.\n")
        
    print(f"--- Pure Tech Support Calls (No Returns) ---")
    if pure_tech:
        for name, snippet in pure_tech:
            print(f"File: {name}\nSnippet: {snippet}...\n")
    else:
        print("None found.\n")

if __name__ == "__main__":
    validate_topic_purity()
