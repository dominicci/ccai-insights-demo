#!/usr/bin/env python3
"""
Script to fix incorrect speaker segmentation in CCAI JSON transcript files.
Reads conversation JSONs with monolithic text entries and re-segments them into
properly tagged AGENT/CUSTOMER turns.
"""

import json
import re
import argparse
import os
from pathlib import Path


def segment_conversation(monolithic_text):
    """
    Segments a monolithic conversation text block into alternating AGENT/CUSTOMER turns.
    
    Args:
        monolithic_text: Full conversation as a single string
        
    Returns:
        List of segmented turn objects with 'text', 'role', and 'speakerId' fields
    """
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', monolithic_text)
    
    if not parts:
        parts = [monolithic_text]
    
    # Merge greeting fragments and short utterances
    merged_parts = []
    i = 0
    while i < len(parts):
        current = parts[i].strip()
        
        # Check if this should be merged with next segment
        if i + 1 < len(parts):
            next_part = parts[i + 1].strip()
            
            # Merge greeting fragments or organizational info
            should_merge = (
                (len(current) < 40 and any(phrase in current.lower() for phrase in 
                    ['thank you for calling', 'good morning', 'good afternoon', 'good evening'])) or
                ('this is' in current.lower() and 'how' not in current.lower()) or
                (current.endswith('[ORGANIZATION].') or current.endswith('[LOCATION].'))
            )
            
            if should_merge:
                current = current + ' ' + next_part
                i += 1  # Skip next since we merged it
        
        if current:
            merged_parts.append(current)
        i += 1
    
    # Determine starting role based on first segment
    first_segment = merged_parts[0].lower() if merged_parts else ""
    
    # Agent indicators
    agent_indicators = [
        'thank you for calling',
        'how can i help',
        'how may i help',
        'good morning',
        'good afternoon', 
        'good evening',
        'this is'
    ]
    
    # Start with AGENT if first segment contains agent indicators
    if any(indicator in first_segment for indicator in agent_indicators):
        current_role = "AGENT"
    else:
        # Default to AGENT (most transcripts start with agent greeting)
        current_role = "AGENT"
    
    # Build segmented entries
    entries = []
    for part in merged_parts:
        if not part.strip():
            continue
            
        entry = {
            "text": part.strip(),
            "speakerId": current_role,
            "role": current_role
        }
        entries.append(entry)
        
        # Alternate role
        current_role = "CUSTOMER" if current_role == "AGENT" else "AGENT"
    
    return entries


def process_single_file(input_path, output_path):
    """
    Process a single JSON file, fixing its segmentation.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    """
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r') as f:
        conversation = json.load(f)
    
    # Extract monolithic text from single entry
    if len(conversation.get('entries', [])) == 1:
        monolithic_text = conversation['entries'][0]['text']
        
        # Re-segment the conversation
        new_entries = segment_conversation(monolithic_text)
        
        # Replace entries with segmented version
        conversation['entries'] = new_entries
        
        # Write corrected JSON
        with open(output_path, 'w') as f:
            json.dump(conversation, f, indent=4)
        
        print(f"  ✓ Segmented into {len(new_entries)} turns")
    else:
        print(f"  ⚠ Skipping: already has {len(conversation.get('entries', []))} entries")


def process_directory(input_dir, output_dir):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory for output JSON files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    for json_file in json_files:
        output_file = output_path / json_file.name
        try:
            process_single_file(json_file, output_file)
        except Exception as e:
            print(f"  ✗ Error processing {json_file}: {e}")
    
    print(f"\n✓ Processing complete. Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix speaker segmentation in CCAI JSON transcript files"
    )
    parser.add_argument(
        'input',
        help='Input JSON file or directory'
    )
    parser.add_argument(
        'output',
        help='Output JSON file or directory'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Process single file
        process_single_file(input_path, output_path)
    elif input_path.is_dir():
        # Process directory
        process_directory(input_path, output_path)
    else:
        print(f"Error: {args.input} is not a valid file or directory")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
