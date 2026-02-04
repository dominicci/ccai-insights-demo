import os
import json
import shutil
import glob

# Configuration
SOURCE_DIR = 'data/synthetic_transcripts/velofit'
DEST_A = 'data/synthetic_transcripts/velofit/sorted/history_pre_change'
DEST_B = 'data/synthetic_transcripts/velofit/sorted/current_post_change'

# Agent IDs for "Old Days" (History Pre Change)
OLD_DAYS_AGENT_IDS = {201, 202, 203, 204, 205, 206}

def sort_transcripts():
    # Ensure destination directories exist
    os.makedirs(DEST_A, exist_ok=True)
    os.makedirs(DEST_B, exist_ok=True)

    json_files = glob.glob(os.path.join(SOURCE_DIR, '*.json'))
    
    count_history = 0
    count_current = 0
    errors = 0

    print(f"Scanning {len(json_files)} files in {SOURCE_DIR}...")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            agent_id = None
            # Find the first entry where role is AGENT
            if 'entries' in data:
                for entry in data['entries']:
                    if entry.get('role') == 'AGENT':
                        agent_id = entry.get('user_id')
                        break
            
            if agent_id is None:
                # Fallback or skip if no agent found (though based on schema there should be one)
                # print(f"Skipping {file_path}: No AGENT role found.")
                continue

            # Determine destination
            if agent_id in OLD_DAYS_AGENT_IDS:
                shutil.move(file_path, os.path.join(DEST_A, os.path.basename(file_path)))
                count_history += 1
            else:
                shutil.move(file_path, os.path.join(DEST_B, os.path.basename(file_path)))
                count_current += 1

        except json.JSONDecodeError:
            print(f"Error decoding JSON: {file_path}")
            errors += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            errors += 1

    print("-" * 30)
    print(f"Moved {count_history} files to History (Pre-Change)")
    print(f"Moved {count_current} files to Current (Post-Change)")
    if errors > 0:
        print(f"Encountered {errors} errors.")
    
    scan_unrelated_calls()

def scan_unrelated_calls():
    print("\nScanning for unrelated calls (missing 'VeloFit')...")
    sorted_dir = os.path.dirname(DEST_A) # data/synthetic_transcripts/velofit/sorted
    unrelated_files = []

    for root, dirs, files in os.walk(sorted_dir):
        for file in files:
            if not file.endswith('.json'):
                continue
            
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for "VeloFit" case-insensitive
                if "velofit" not in content.lower():
                    unrelated_files.append(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    count = len(unrelated_files)
    if count == 0:
        print("No unrelated calls found.")
        return

    print(f"Found {count} unrelated calls:")
    for f in unrelated_files:
        print(f" - {os.path.basename(f)}")
    
    choice = input(f"\nDo you want to move these {count} files to another folder? (y/n): ").strip().lower()
    if choice == 'y':
        dest_folder = input("Enter destination folder path: ").strip()
        if not dest_folder:
            print("Invalid path. Aborting move.")
            return
        
        try:
            os.makedirs(dest_folder, exist_ok=True)
            for f in unrelated_files:
                shutil.move(f, os.path.join(dest_folder, os.path.basename(f)))
            print(f"Successfully moved {count} files to {dest_folder}")
        except Exception as e:
            print(f"Error moving files: {e}")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    sort_transcripts()
