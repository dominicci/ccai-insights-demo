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


# load the dataset from the CSV file
def load_csv_dataset(local_dataset_dir, bucket_name, dataset_prefix):
    csv_path = os.path.join(local_dataset_dir, "call_recordings.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Attempting to fetch...")
        try:
            fetch_dataset(bucket_name, dataset_prefix)
        except Exception as e:
            print(f"Failed to fetch dataset: {e}")
            return None

    if os.path.exists(csv_path):
         df = pd.read_csv(csv_path, dtype={'Order Number': object})
         return df
    else:
        print("Failed to load dataset: CSV file still missing after fetch attempt.")
        return None

# iterate through the CSV and build the JSON structure.
# handle the case where the CSV doesn't exist
# handle the case where the CSV has invalid data
def process_transcripts(data_source, output_dir, source_type="csv"):
    
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0

    if source_type == "csv":
        print("Processing CSV dataset...")
        if data_source is None:
            print("No dataframe provided to process.")
            return

        # Fill NaN values with "N/A" to avoid invalid JSON and match requirements
        df_filled = data_source.fillna("N/A")

        # iterate through the CSV and build the JSON structure.
        for index, row in df_filled.iterrows():
            conversation_id = row['id']
            process_single_transcript(
                conversation_id,
                row['Transcript'],
                row['Type'],
                row['Name'],
                row['Order Number'],
                output_dir
            )
            count += 1
            
    elif source_type == "huggingface":
        print("Processing Hugging Face dataset...")
        # Iterate through the streaming dataset
        for i, row in enumerate(data_source):
            # Generate an ID since the dataset doesn't seem to have one
            conversation_id = f"hf_call_{i+1:04d}"
            
            # The new dataset has 'text' but lacks metadata like 'Type', 'Name', 'Order Number'
            # We'll use placeholders for now
            process_single_transcript(
                conversation_id,
                row['text'],
                "General Inquiry", # Placeholder
                "Unknown Customer", # Placeholder
                "N/A",
                output_dir
            )
            count += 1
            
            # Limit for demo purposes if needed, or remove to process all
            if count >= 100: 
                break

    print(f"Successfully processed {count} transcripts to '{output_dir}' directory.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process call center transcripts.")
    parser.add_argument("--dataset_name", type=str, default="AIxBlock/92k-real-world-call-center-scripts-english", help="Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, default="transcripts_json", help="Output directory for JSON files")
    parser.add_argument("--bucket_name", type=str, default="ssi-lab-sandbox-1-ccai-demo", help="GCS bucket name")
    parser.add_argument("--dataset_prefix", type=str, default="call-center-transcripts-dataset/", help="GCS dataset prefix")
    parser.add_argument("--local_dataset_dir", type=str, default="call-center-transcripts-dataset", help="Local directory for CSV dataset")
    parser.add_argument("--source_type", type=str, choices=["csv", "huggingface"], default="csv", help="Source of the dataset")

    args = parser.parse_args()

    if args.source_type == "csv":
        df = load_csv_dataset(args.local_dataset_dir, args.bucket_name, args.dataset_prefix)
        if df is not None:
            process_transcripts(df, args.output_dir, source_type="csv")
    
    elif args.source_type == "huggingface":
        print(f"Loading dataset {args.dataset_name} from Hugging Face (streaming)...")
        try:
            # Use streaming=True to avoid schema errors with mixed types
            dataset = load_dataset(args.dataset_name, streaming=True)
            
            # Use manual iteration with error handling to skip malformed rows
            count = 0
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            iterator = iter(dataset['train'])
            while True:
                try:
                    row = next(iterator)
                    
                    # Generate an ID since the dataset doesn't seem to have one
                    # We use count as index approximation since enumerate is complicated with skip
                    conversation_id = f"hf_call_{count+1:04d}"
                    
                    # The new dataset has 'text' but lacks metadata like 'Type', 'Name', 'Order Number'
                    process_single_transcript(
                        conversation_id,
                        row['text'],
                        "General Inquiry", # Placeholder
                        "Unknown Customer", # Placeholder
                        "N/A",
                        output_dir
                    )
                    count += 1
                    
                    # Limit for demo purposes if needed, or remove to process all
                    if count >= 100: 
                        break
                        
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Skipping row due to load error: {e}")
                    continue

        except Exception as e:
            print(f"Failed to load/process Hugging Face dataset: {e}")

    try:
        upload_json_files(args.bucket_name, args.dataset_prefix, args.output_dir)
    except Exception as e:
        print(f"Upload failed after retries: {e}")


