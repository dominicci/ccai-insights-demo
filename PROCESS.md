# CCAI Synthetic Data Pipeline: Order of Execution

This document outlines the standard operational workflow for generating, verifying, and reporting on synthetic contact center data using the **VeloFit** profile.

## 1. Environment Setup

Ensure your environment is configured with the necessary API keys and dependencies.

```bash
# 1. Install dependencies (using uv, pip, or similar)
uv pip install -r requirements.txt

# 2. Export API Key (Google Gemini is preferred)
export GOOGLE_API_KEY="your-api-key-here"
# OR
export OPENAI_API_KEY="your-api-key-here"
```

---

## 2. Generation (The Core Step)
**Script**: `src.synth_data.main`

This is the primary entry point. It generates synthetic conversations in CCAI insights-compliant JSON format.

```bash
# Standard Execution (VeloFit Profile)
# - Generates 10 conversations
# - Uses 3 parallel workers
# - Saves to data/synthetic_transcripts
uv run python -m src.synth_data.main --count 10 --profile velofit
```

**Common Flags**:
*   `--count`: Number of files to generate (Default: 5).
*   `--workers`: Number of parallel threads (Default: 3).
*   `--start-date` / `--end-date`: Date range for the simulation (Format: YYYY-MM-DD).

---

## 3. Operations & Validation (Optional)

After generation, you can run various utility scripts located in `src/ops/` and `src/validation/`.

### Sort & Split
If you need to separate files (e.g., for uploading in batches):
```bash
uv run python src/ops/split_for_upload.py --input_dir data/synthetic_transcripts --max_files 1000
```

### Validate JSON
Ensure the generated files meet strict schema requirements before uploading:
```bash
uv run python src/validation/validate_json_schema.py --input_dir data/synthetic_transcripts
```

---

## 4. Reporting & Visualization
**Script**: `src/reporting/`

Generate dashboards or statistics based on the generated batch.

```bash
# Generate aggregated stats
uv run python src/reporting/aggregate_stats.py
```

---

## 5. Legacy Generation (Deprecated)
If you need to generate data using the old "Generic" profile:

```bash
uv run python src/legacy/generate_generic_data.py --count 5
```
