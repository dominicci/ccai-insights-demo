# basic data exploration to understand the dataset

import json
import os
import argparse
import pandas as pd
from google.cloud import storage
from google.api_core.exceptions import NotFound, Forbidden
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datasets import load_dataset

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

def process_single_transcript(conversation_id, text, call_type, customer_name, order_number, output_dir):
    conversation_data = {
        "conversation_info": {
            "conversation_id": conversation_id,
            "metadata": {
                "call_type": call_type,
                "customer_name": customer_name,
                "order_number": order_number
            }
        },
        "entries": [
            {
                "text": text,
                "role": "CUSTOMER",
                "user_id": 1,
                "start_timestamp_usec": 0
            }
        ]
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
            process_transcripts(dataset['train'], args.output_dir, source_type="huggingface")
        except Exception as e:
            print(f"Failed to load/process Hugging Face dataset: {e}")

    try:
        upload_json_files(args.bucket_name, args.dataset_prefix, args.output_dir)
    except Exception as e:
        print(f"Upload failed after retries: {e}")


