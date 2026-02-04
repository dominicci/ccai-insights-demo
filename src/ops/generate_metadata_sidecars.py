import os
import json
import glob
import shutil
import re
import argparse

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def convert_agent_info(agent_info_list):
    new_list = []
    for agent in agent_info_list:
        new_agent = {}
        for k, v in agent.items():
            new_agent[to_snake_case(k)] = v
        new_list.append(new_agent)
    return new_list

def generate_sidecars():
    parser = argparse.ArgumentParser(description="Generate CCAI Insights metadata sidecars.")
    parser.add_argument("--input", default="data/upload_batches/02_crisis_phase", help="Input directory containing JSON transcripts")
    parser.add_argument("--output", default="data/upload_batches/02_crisis_phase_metadata", help="Output directory for metadata files")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Scanning {input_dir}...")
    files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    
    count = 0
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Extract Components
            conv_info = data.get("conversation_info", {})
            quality_meta = conv_info.get("qualityMetadata", {})
            custom_meta = conv_info.get("metadata", {}) # Our custom fields
            
            # Identify Conversation ID
            # Use filename without extension as safe ID, or specific field if exists
            fname = os.path.basename(fpath)
            conv_id = fname.replace("synthetic_call_", "").replace(".json", "")
            
            # Construct Sidecar Object
            sidecar = {}
            
            # 1. Agent Info (Convert keys to snake_case)
            if "agentInfo" in quality_meta:
                sidecar["agent_info"] = convert_agent_info(quality_meta["agentInfo"])
            
            # 2. Customer Satisfaction (if present)
            if "customerSatisfactionRating" in quality_meta:
                sidecar["customer_satisfaction_rating"] = quality_meta["customerSatisfactionRating"]
                
            # 3. Conversation ID
            sidecar["conversation_id"] = conv_id
            
            # 4. Custom Metadata (Flattened into root)
            for k, v in custom_meta.items():
                sidecar[to_snake_case(k)] = v
                
            # Tag the dataset version
            sidecar["dataset_version"] = "v4_full_scale"
            
            # Write Sidecar
            dest_path = os.path.join(output_dir, fname)
            with open(dest_path, 'w') as f_out:
                json.dump(sidecar, f_out, indent=2)
                
            count += 1
            
        except Exception as e:
            print(f"Error {fpath}: {e}")
            
    print(f"Generated {count} metadata sidecars in {output_dir}")

if __name__ == "__main__":
    generate_sidecars()
