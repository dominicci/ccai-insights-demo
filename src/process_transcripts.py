# basic data exploration to understand the dataset

import json
import os
import argparse
import pandas as pd
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datasets import load_dataset, Features, Value, Sequence

# fetch the dataset from GCS gs://ssi-lab-sandbox-1-ccai-demo/call-center-transcripts-dataset/
# contains call_recordings.csv, data_description.csv, WAV files, and a README.md
# handle retry logic and exponential backoff
# handle the case where the bucket doesn't exist
# handle authentication errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception)), # Retry on general exceptions for network issues, but we'll catch specific ones inside
    reraise=True
)
def fetch_dataset(bucket_name, dataset_prefix):
    print(f"Attempting to fetch dataset from gs://{bucket_name}/{dataset_prefix}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        blobs = bucket.list_blobs(prefix=dataset_prefix)
        
        found_files = False
        for blob in blobs:
            found_files = True
            # Create local directory structure
            local_path = blob.name
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            if not blob.name.endswith('/'): # Skip directories
                print(f"Downloading {blob.name} to {local_path}...")
                blob.download_to_filename(local_path)
        
        if not found_files:
             print(f"Warning: No files found in gs://{bucket_name}/{dataset_prefix}")

    except (NotFound, Forbidden) as e:
        print(f"Error accessing GCS bucket: {e}")
        # We might want to re-raise if we want the retry logic to kick in, 
        # but for auth/not found errors, retrying might not help.
        # However, the user asked for retry logic, so let's stick to the decorator for transient issues
        # and maybe catch critical ones here if we want to stop early. 
        # For now, letting the retry decorator handle it or raising to stop.
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during fetch: {e}")
        raise e



# --- NEW VALIDATION FUNCTION ---
def is_valid_ccai_format(data):
    """
    Checks if the JSON data is already in valid CCAI Insights format.
    Criteria:
    1. Has 'entries' list.
    2. Entries have 'speakerId' and 'start_timestamp_usec'.
    """
    if "entries" not in data or not isinstance(data["entries"], list):
        return False
    
    if len(data["entries"]) > 0:
        first_entry = data["entries"][0]
        if "speakerId" in first_entry and "start_timestamp_usec" in first_entry:
            return True
    
    return False

# --- UPDATED PROCESSING LOGIC ---
def process_source_directory(source_path, output_dir):
    """
    Iterates through a directory and decides how to handle each file
    based on its extension and content.
    """
    os.makedirs(output_dir, exist_ok=True)
    count_processed = 0
    count_passthrough = 0

    # Handle single file or directory
    files_to_process = []
    if os.path.isfile(source_path):
        files_to_process.append(source_path)
    elif os.path.isdir(source_path):
        for root, _, files in os.walk(source_path):
            for file in files:
                files_to_process.append(os.path.join(root, file))

    print(f"Scanning {len(files_to_process)} files in '{source_path}'...")

    for file_path in files_to_process:
        filename = os.path.basename(file_path)
        
        # CASE 1: JSON FILES (Could be Synthetic or Raw)
        if filename.lower().endswith(".json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # DECISION: Is it already perfect?
                if is_valid_ccai_format(data):
                    # PASS-THROUGH: Just copy to output dir
                    print(f"  [PASS-THROUGH] Valid CCAI JSON detected: {filename}")
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w') as f_out:
                        json.dump(data, f_out, indent=4)
                    count_passthrough += 1
                else:
                    # RE-PROCESS: It's JSON but likely raw/monolithic
                    # (Assumes 'text' field exists in the JSON root or entries)
                    # Implementation detail: Extract text and run segment_text
                    print(f"  [PROCESS] Raw JSON detected: {filename}")
                    # ... Logic to extract text from raw JSON would go here ...
                    # For now, let's assume raw JSONs are just simple text wrappers
                    pass 

            except Exception as e:
                print(f"Error reading JSON {filename}: {e}")

        # CASE 2: CSV FILES (Raw Data)
        elif filename.lower().endswith(".csv"):
            print(f"  [PROCESS] CSV detected: {filename}")
            try:
                df = pd.read_csv(file_path, dtype={'Order Number': object}).fillna("N/A")
                for index, row in df.iterrows():
                    # Handle missing columns gracefully
                    c_id = row.get('id', f"csv_{index}")
                    text = row.get('Transcript', "")
                    c_type = row.get('Type', "General")
                    c_name = row.get('Name', "Unknown")
                    order_num = row.get('Order Number', "N/A")
                    
                    if text:
                        process_single_transcript(c_id, text, c_type, c_name, order_num, output_dir)
                        count_processed += 1
            except Exception as e:
                print(f"Error processing CSV {filename}: {e}")

    print(f"\\nSummary: {count_passthrough} files passed through, {count_processed} files processed.")


import re

# ... (existing imports but re is new)

def segment_text(text, customer_name="Unknown"):
    """
    Segments text using an IVR-aware state machine and keyword scoring.
    1. Starts in IVR_SYSTEM mode.
    2. Switches to Conversation mode upon detecting human anchor phrases.
    3. Assigns roles based on keyword scoring (Positive=Agent, Negative=Customer).
    4. Uses customer_name metadata as a super keyword for self-identification.
    5. Calculates timestamps based on character length (~15 chars/sec).
    """
    
    # --- CONFIGURATION ---
    # Regex triggers that signal the IVR is done and a human has picked up
    ivr_handoff_triggers = [
        r"this is", 
        r"my name is", 
        r"how can i help",
        r"speaking with you"
    ]

    # Keyword Scoring: Positive = Agent, Negative = Customer
    keyword_map = [
        (r"thank you for calling", 5),
        (r"how can i help", 5),
        (r"my name is", 3),
        (r"appointments", 2),
        (r"hold on", 3),
        (r"text message", 3),
        (r"reach you back", 3),
        (r"correct\?", 2),
        (r"i'm calling", -5),  # Strong customer indicator
        (r"i need", -4),
        (r"i want", -4),
        (r"what time", -4),
        (r"you open", -4),
        (r"my number", -3),
        (r"you close", -4),
        (r"look at car", -3)
    ]

    # --- PRE-PROCESSING ---
    # Clean up text and split by terminal punctuation (. ? !)
    # The lookbehind (?<=[.?!]) ensures we keep the punctuation with the sentence.
    clean_text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.?!])\s+', clean_text)

    entries = []
    
    # State Variables
    is_ivr_active = True 
    current_role = "IVR_SYSTEM"  # Start as system
    previous_role = None
    consecutive_turns = 0
    current_timestamp = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_lower = sentence.lower()
        
        # 1. CHECK FOR HANDOFF (IVR -> HUMAN)
        if is_ivr_active:
            for trigger in ivr_handoff_triggers:
                if re.search(trigger, sentence_lower):
                    is_ivr_active = False 
                    current_role = "AGENT"  # Humans usually speak first after pickup
                    break
        
        # Track for monologue breaking
        if current_role == previous_role:
            consecutive_turns += 1
        else:
            consecutive_turns = 1
        
        # 2. METADATA OVERRIDE (Highest Priority)
        # If the customer's name appears in the text, it's likely the customer introducing themselves.
        # Exclude "Unknown Customer" or "N/A" placeholders.
        name_match = False
        if customer_name and customer_name.lower() not in ["unknown customer", "n/a", "unknown"]:
            # Check if the full name appears in the sentence
            if customer_name.lower() in sentence_lower:
                current_role = "CUSTOMER"
                name_match = True
                is_ivr_active = False  # <--- ADD THIS LINE TO KILL THE IVR

        # 3. DETERMINE ROLE (Only run if metadata didn't catch it)
        if not name_match:
            if is_ivr_active:
                current_role = "IVR_SYSTEM"
                speaker_id = "SYSTEM"
                user_id = 0
            else:
                # Score the sentence to see if we should switch speakers
                score = 0
                for regex, weight in keyword_map:
                    if re.search(regex, sentence_lower):
                        score += weight
                
                # Threshold Logic
                if score >= 2:
                    current_role = "AGENT"
                elif score <= -2:
                    current_role = "CUSTOMER"
                else:
                    # NEUTRAL SCORE LOGIC (Stickiness + Monologue Breaker)
                    if previous_role and consecutive_turns >= 3 and len(sentence.split()) < 5:
                        # Break monologue: if same speaker for 3+ turns and short sentence, switch
                        current_role = "CUSTOMER" if previous_role == "AGENT" else "AGENT"
                    else:
                        # Keep previous speaker (stickiness)
                        current_role = previous_role if previous_role else "AGENT"
        
        # Set speaker_id and user_id based on final role
        if current_role == "IVR_SYSTEM":
            speaker_id = "SYSTEM"
            user_id = 0
        else:
            speaker_id = "AGENT" if current_role == "AGENT" else "CUSTOMER"
            user_id = 2 if current_role == "AGENT" else 1

        # 4. CALCULATE DURATION
        # Estimate: 15 chars ~ 1 second (1,000,000 usec)
        char_count = len(sentence)
        duration_usec = int((char_count / 15) * 1000000)
        
        # Ensure minimum duration of 0.5 seconds to avoid zero-length glitches
        duration_usec = max(duration_usec, 500000)

        entry = {
            "text": sentence.strip(),
            "speakerId": speaker_id,
            "role": current_role,
            "user_id": user_id,
            "start_timestamp_usec": current_timestamp
        }
        
        entries.append(entry)
        
        # Update state for next iteration
        previous_role = current_role
        current_timestamp += duration_usec

    return entries

def process_single_transcript(conversation_id, text, call_type, customer_name, order_number, output_dir):
    
    segmented_entries = segment_text(text, customer_name)
    
    conversation_data = {
        "conversation_info": {
            "conversation_id": conversation_id,
            "metadata": {
                "call_type": call_type,
                "customer_name": customer_name,
                "order_number": order_number
            }
        },
        "entries": segmented_entries
    }
    
    file_path = os.path.join(output_dir, f"{conversation_id}.json")
    with open(file_path, 'w') as f:
        json.dump(conversation_data, f, indent=4)

# upload the JSON files to GCS bucket gs://ssi-lab-sandbox-1-ccai-demo/call-center-transcripts-dataset/
# handle retry logic and exponential backoff
# handle the case where the bucket doesn't exist
# handle authentication errors
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception)),
    reraise=True
)
def upload_json_files(bucket_name, dataset_prefix, output_dir):
    print(f"Attempting to upload JSON files to gs://{bucket_name}/{dataset_prefix}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        if not os.path.exists(output_dir):
            print(f"Output directory {output_dir} does not exist. Nothing to upload.")
            return

        for filename in os.listdir(output_dir):
            if filename.endswith(".json"):
                local_path = os.path.join(output_dir, filename)
                blob_name = f"{dataset_prefix}{filename}" # Upload to the same prefix/folder
                blob = bucket.blob(blob_name)
                
                print(f"Uploading {local_path} to gs://{bucket_name}/{blob_name}...")
                blob.upload_from_filename(local_path)
        
        print("Upload completed successfully.")

    except (NotFound, Forbidden) as e:
        print(f"Error accessing GCS bucket during upload: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during upload: {e}")
        raise e


import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and process call center data.")
    # Changed argument from 'local_dataset_dir' to generic 'source_path'
    parser.add_argument("--source_path", type=str, default=None, help="Path to input file or directory (defaults to data/synthetic_transcripts)")
    parser.add_argument("--output_dir", type=str, default="transcripts_json", help="Output directory")
    parser.add_argument("--bucket_name", type=str, default="ssi-lab-sandbox-1-ccai-demo", help="GCS bucket name")
    parser.add_argument("--dataset_prefix", type=str, default="call-center-transcripts-dataset/", help="GCS prefix")
    
    args = parser.parse_args()

    # Determine source path
    if args.source_path:
        source_path = pathlib.Path(args.source_path)
    else:
        # Default: ProjectRoot/data/synthetic_transcripts
        script_dir = pathlib.Path(__file__).parent.resolve()
        source_path = script_dir.parent / "data" / "synthetic_transcripts"

    print(f"Processing data from: {source_path}")

    # 1. Process/Ingest Data
    # Convert path object to string for existing function
    process_source_directory(str(source_path), args.output_dir)

    # 2. Upload Results
    try:
        upload_json_files(args.bucket_name, args.dataset_prefix, args.output_dir)
    except Exception as e:
        print(f"Upload failed: {e}")




