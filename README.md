# CCAI Insights - Transcript Processor

This project processes call center transcripts from CSV format into structured JSON files and integrates with Google Cloud Storage (GCS) for data retrieval and storage.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Google Cloud credentials (if running in an environment that requires explicit auth)

## Setup

This project uses `uv` for dependency management.

### 1. Install Dependencies

You can install dependencies into a virtual environment.

**Default `.venv`:**
```bash
uv sync
```

**Custom `.insights` environment:**
If you prefer to use a custom virtual environment name (e.g., `.insights`):

```bash
# Create the environment
uv venv .insights

# Sync dependencies to it
UV_PROJECT_ENVIRONMENT=.insights uv sync
```

## Execution

### Running the Script

The script `process_transcripts.py` processes transcripts into JSON and uploads them to GCS. It supports two data sources:
1.  **Local CSV**: `call-center-transcripts-dataset/call_recordings.csv` (Default)
2.  **Hugging Face**: `AIxBlock/92k-real-world-call-center-scripts-english`

**Using `uv run` (Recommended):**

```bash
# Run with defaults (loads local CSV)
./.insights/bin/python process_transcripts.py

# Run with Hugging Face dataset
./.insights/bin/python process_transcripts.py --source_type huggingface

# Run with custom output directory
./.insights/bin/python process_transcripts.py --output_dir my_custom_output
```

**Using the Python Executable Directly:**

```bash
source .insights/bin/activate
python process_transcripts.py --help
```

## Configuration / CLI Arguments

You can configure the script using command-line arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--source_type` | `csv` | Source of the dataset: `csv` or `huggingface` |
| `--dataset_name` | `AIxBlock/92k...` | Hugging Face dataset name (if source is `huggingface`) |
| `--output_dir` | `transcripts_json` | Local directory for generated JSON files |
| `--bucket_name` | `ssi-lab-sandbox-1-ccai-demo` | GCS bucket name for upload |
| `--dataset_prefix` | `call-center-transcripts-dataset/` | Prefix for GCS upload |
| `--local_dataset_dir` | `call-center-transcripts-dataset` | Local directory for CSV dataset |
