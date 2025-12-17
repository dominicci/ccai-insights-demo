# basic data exploration to understand the dataset

import json
import os
import pandas as pd
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datasets import load_dataset


BUCKET_NAME = "ssi-lab-sandbox-1-ccai-demo"
DATASET_PREFIX = "call-center-transcripts-dataset/"
LOCAL_DATASET_DIR = "call-center-transcripts-dataset"
OUTPUT_DIR = "transcripts_json"

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
def fetch_dataset():
    print(f"Attempting to fetch dataset from gs://{BUCKET_NAME}/{DATASET_PREFIX}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        blobs = bucket.list_blobs(prefix=DATASET_PREFIX)
        
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
             print(f"Warning: No files found in gs://{BUCKET_NAME}/{DATASET_PREFIX}")

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
def load_dataset():
    csv_path = os.path.join(LOCAL_DATASET_DIR, "call_recordings.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Attempting to fetch...")
        try:
            fetch_dataset()
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
def process_transcripts(df):
    if df is None:
        print("No dataframe provided to process.")
        return

    '''
    For each row in your CSV, you will create a JSON file that follows this structure. 
    You will need to parse the single Transcript column in your CSV into multiple entries (dialog turns).
    Example: "Hello, I'm Sarah Miller. I'm calling to inquire about the AC-7892 air conditioner unit..."
    
    Required JSON Output for a single conversation
    {
        "conversation_info": {
            "conversation_id": "call_recording_01",
            "metadata": {
            "call_type": "Product Inquiry",
            "customer_name": "Sarah Miller",
            "order_number": "N/A"
            }
        },
        "entries": [
            {
            "text": "Hello, I'm Sarah Miller. I'm calling to inquire about the AC-7892 air conditioner unit...",
            "role": "CUSTOMER",
            "user_id": 1,
            "start_timestamp_usec": 0 // Use 0 for the first turn, then estimate subsequent turns
            },
            {
            "text": "Thank you for calling. I can certainly help you with the AC-7892. What is your question?",
            "role": "AGENT",
            "user_id": 2,
            "start_timestamp_usec": 8000000 // A logical time increment for the next speaker (8 seconds later)
            }
            // ... add more turns by separating the customer and agent dialogue ...
        ]
    }
    '''
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fill NaN values with "N/A" to avoid invalid JSON and match requirements
    df_filled = df.fillna("N/A")

    # iterate through the CSV and build the JSON structure.
    for index, row in df_filled.iterrows():
        conversation_id = row['id']
        
        conversation_data = {
            "conversation_info": {
                "conversation_id": conversation_id,
                "metadata": {
                    "call_type": row['Type'],
                    "customer_name": row['Name'],
                    "order_number": row['Order Number']
                }
            },
            "entries": [
                {
                    "text": row['Transcript'],
                    "role": "CUSTOMER",
                    "user_id": 1,
                    "start_timestamp_usec": 0
                }
            ]
        }
        
        file_path = os.path.join(OUTPUT_DIR, f"{conversation_id}.json")
        with open(file_path, 'w') as f:
            json.dump(conversation_data, f, indent=4)

    print(f"Successfully processed {len(df)} transcripts to '{OUTPUT_DIR}' directory.")

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
def upload_json_files():
    print(f"Attempting to upload JSON files to gs://{BUCKET_NAME}/{DATASET_PREFIX}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        
        if not os.path.exists(OUTPUT_DIR):
            print(f"Output directory {OUTPUT_DIR} does not exist. Nothing to upload.")
            return

        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith(".json"):
                local_path = os.path.join(OUTPUT_DIR, filename)
                blob_name = f"{DATASET_PREFIX}{filename}" # Upload to the same prefix/folder
                blob = bucket.blob(blob_name)
                
                print(f"Uploading {local_path} to gs://{BUCKET_NAME}/{blob_name}...")
                blob.upload_from_filename(local_path)
        
        print("Upload completed successfully.")

    except (NotFound, Forbidden) as e:
        print(f"Error accessing GCS bucket during upload: {e}")
        raise e
    except Exception as e:
        print(f"An unexpected error occurred during upload: {e}")
        raise e


if __name__ == "__main__":

    df = load_dataset()
    
    if df is not None:
        process_transcripts(df)
        try:
            upload_json_files()
        except Exception as e:
            print(f"Upload failed after retries: {e}")


