# Logic Audit & Validation Guide

The `src/validate_synthetic_data.py` script is a specialized tool used to verify that the synthetic data generation logic is performing as expected. It performs a "Logic Audit" by scanning generated JSON files and calculating key performance metrics.

## Key Metrics Evaluated

1.  **Routing Split Check**:
    - Calculates the percentage of calls handled by Tier 200 (Rookies) vs. Tier 300+ (Standard).
    - **Target**: ~50% split.

2.  **Knowledge Base Integrity**:
    - Scans Rookie calls for modern policy terms (e.g., "hygiene", "bio-contaminant").
    - **Target**: 0 violations (Rookies should only know the outdated 90-day policy).

3.  **Behavioral Consistency**:
    - Compares average conversation length (turns).
    - **Hypothesis**: Rookie calls should be significantly shorter due to "Blind Approvals."

4.  **Outcome Distribution**:
    - Compares "Resolved" vs. "Unresolved" rates between agent tiers.
    - **Hypothesis**: Rookies should have a much higher resolution rate (false positives) compared to Standard agents (who enforce denials).

5.  **Scenario Distribution**:
    - Counts and displays the frequency of each scenario type.
    - **Target**: Verify that "Exceptions" are capped at ~10% of total volume.

## Usage

You can run the audit against any output directory.

```bash
# Basic Usage (Positional argument for directory)
uv run python src/validate_synthetic_data.py data/synthetic_transcripts/my_test_run/velofit
```

## Interpreting Results

If you see high **KB Integrations Violations**, it indicates that the LLM is "leaking" modern policy knowledge into Rookie personas, requiring adjustment of the system prompt or scenario overrides.
