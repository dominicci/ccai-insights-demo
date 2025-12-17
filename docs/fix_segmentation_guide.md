# Fix Segmentation Script

## Overview
`fix_segmentation.py` is a standalone utility to re-segment CCAI JSON transcript files that have monolithic text entries. It extracts the full conversation from a single entry and splits it into properly tagged AGENT/CUSTOMER turns.

## Features
- ✅ Automatically merges greeting fragments
- ✅ Detects agent greetings using common call center phrases
- ✅ Splits conversations on sentence boundaries
- ✅ Assigns alternating AGENT/CUSTOMER roles
- ✅ Preserves `conversation_info` metadata
- ✅ Processes single files or entire directories

## Usage

### Single File
```bash
python fix_segmentation.py input.json output.json
```

### Batch Processing (Directory)
```bash
python fix_segmentation.py transcripts_json/ corrected_transcripts/
```

## Input Format
The script expects JSON files with this structure:
```json
{
  "conversation_info": { ... },
  "entries": [
    {
      "text": "Entire conversation in one block...",
      "role": "CUSTOMER",
      "user_id": 1,
      "start_timestamp_usec": 0
    }
  ]
}
```

## Output Format
The script produces:
```json
{
  "conversation_info": { ... },
  "entries": [
    {
      "text": "Thank you for calling...",
      "speakerId": "AGENT",
      "role": "AGENT"
    },
    {
      "text": "Yes, I need help with...",
      "speakerId": "CUSTOMER",
      "role": "CUSTOMER"
    }
    // ... more turns
  ]
}
```

## Segmentation Logic
1. **Sentence Splitting**: Splits on `.!?` followed by capital letters or `[` (PII tags)
2. **Greeting Merging**: Combines fragments like "Thank you for calling." + "[ORGANIZATION]." + "This is [NAME]"
3. **Role Detection**: Identifies AGENT based on greeting phrases ("thank you for calling", "how can I help", etc.)
4. **Alternation**: Strictly alternates roles after initial speaker is determined

## Examples

### Process Current Transcripts
```bash
# Re-segment all files in transcripts_json/
python fix_segmentation.py transcripts_json/ corrected_transcripts/
```

### Test on Sample
```bash
# Test on a single file
python fix_segmentation.py test_input.json test_output.json
```

## Notes
- The script skips files that already have multiple entries
- Processing preserves the original `conversation_info` metadata
- Agent detection assumes standard call center greeting patterns
