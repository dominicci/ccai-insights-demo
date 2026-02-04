
import os
import json
from collections import defaultdict

import sys

def check_consistency():
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/synthetic_transcripts/velofit/test_validation_jan28"
        
    agent_map = defaultdict(set)
    inconsistencies = []

    print(f"Scanning {data_dir}...")
    
    file_count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                file_count += 1
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        meta = data.get("conversation_info", {}).get("metadata", {})
                        
                        agent_id = meta.get("agent_id")
                        agent_name = meta.get("agent_name")
                        
                        if agent_id and agent_name:
                            agent_map[agent_id].add(agent_name)
                            
                        # Check Text Consistency
                        entries = data.get("entries", [])
                        agent_intro_found = False
                        for entry in entries:
                            if entry.get("role") == "AGENT":
                                text = entry.get("text", "")
                                if "my name is" in text.lower():
                                    agent_intro_found = True
                                    if agent_name.lower() not in text.lower():
                                        print(f"[FAIL] {file}: Metadata Name '{agent_name}' not found in intro: '{text}'")
                                        inconsistencies.append(file)
                                    break
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    print(f"Scanned {file_count} files.")
    
    print("\n--- Consistency Report (ID -> Names) ---")
    clean_id = True
    for aid, names in agent_map.items():
        if len(names) > 1:
            clean_id = False
            print(f"[FAIL] Agent ID {aid} has multiple names: {names}")
            inconsistencies.append(aid)
            
    if clean_id:
        print("[SUCCESS] All Agent IDs map to unique names.")

    print("\n--- Reverse Consistency Report (Name -> IDs) ---")
    name_map = defaultdict(set)
    for aid, names in agent_map.items():
        for name in names:
            name_map[name].add(aid)
            
    for name, aids in name_map.items():
        if len(aids) > 1:
            print(f"[INFO] Name '{name}' is used by multiple IDs: {aids}")
        # else:
            # print(f"[OK] Name '{name}' is unique to ID: {list(aids)[0]}")

if __name__ == "__main__":
    check_consistency()
