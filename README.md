# CCAI Synthetic Data Generator

A toolkit for generating high-quality synthetic contact center transcripts (JSON) to demonstrate Google Cloud CCAI Insights features like Topic Modeling, Smart Highlights, and Quality AI.

## Key Features

-   **Valid CCAI JSON Schema**: Produces files that are strictly compliant with Google Cloud CCAI Insights ingestion requirements.
-   **Realistic Workforce Simulation**: Assigns consistent Agent IDs from a predefined pool and generates unique Customer IDs for every call.
-   **Keyword Injection**: Naturally integrates specific keywords (e.g., "tracking number", "cancel subscription") into dialogues to trigger Topic Models.
-   **Variable Outcomes**: Simulates realistic call friction with a 90% Resolved / 10% Unresolved split.
-   **Multi-Agent Transfers**: Supports complex scenarios where calls are transferred from Tier 1 to Tier 2 agents or supervisors (e.g., Agent 201 -> Agent 205).

## Usage Guide

### 1. Generate Synthetic Data
Run the generator script to create a batch of synthetic JSON transcripts. You can specify the number of files and the output directory.

```bash
# Generate 100 synthetic transcripts
# Defaults to data/synthetic_transcripts
python src/generate_synthetic_ccai_data.py --count 100 
```

*Note: Requires `OPENAI_API_KEY` or `GOOGLE_API_KEY` to be set in your environment.*

### 2. Process & Ingest
The `process_transcripts.py` script handles the ingestion pipeline. It automatically detects if files are valid CCAI JSON (pass-through) or need segmentation (CSV processing), and uploads them to Google Cloud Storage.

```bash
# Defaults source to data/synthetic_transcripts
python src/process_transcripts.py --output_dir data/processed_output
```

## Google Cloud CCAI Insights Limits

When planning your data pipeline and ingestion strategy, be aware of the following platform limits:

### Ingestion Limits
-   **Import Rate**: The quickstart method allows up to **18,000 conversations per hour**.
-   **Bulk Import**: Batch limit of **10,000 conversations per JSON file** when importing from Cloud Storage.

### Audio & Text Constraints
-   **Async Audio**: Maximum duration of **480 minutes (8 hours)** per file.
-   **Telephony**: Maximum call duration of **3.5 hours**.
-   **Chat**: 
    -   Dialogflow responses are capped at **4,000 characters**.
    -   Intent detection input is capped at **256 characters**.

### Feature Quotas
-   **Quality AI**: Scorecards can evaluate a maximum of **50 questions per conversation**.
-   **Topic Modeling**: Requires a technical minimum of **100 conversations** to train (though **1,000+** is recommended for best results).

## Future Roadmap

-   **BigQuery Streaming**: Investigate streaming export to BigQuery to handle datasets larger than the standard UI export limits.
