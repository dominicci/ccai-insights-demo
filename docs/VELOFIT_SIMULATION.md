# VeloFit Simulation Overview

The VeloFit simulation is an advanced path for generating synthetic data designed to test compliance, policy adherence, and agent diagnostic skills.

## Core Mechanics

### 1. Global Routing Fork (50/50 Split)
Every conversation selection for the VeloFit profile undergoes a 50/50 probability check to simulate operational variance:

- **Path A: Rookie / Outdated (The "Problem" Path)**
    - **Selection**: Tier 200 Agents (IDs 201-206).
    - **Knowledge Base**: `KB_OUTDATED` (Uses a 90-day return policy).
    - **Behavior**: "Blind Approval" script. Agents discard specific scenario instructions and simply approve returns immediately if they are within 90 days.
    - **Goal**: Simulates compliance misses and financial leakage.

- **Path B: Standard / Updated (The "Competent" Path)**
    - **Selection**: Tier 300+ Agents.
    - **Knowledge Base**: `KB_UPDATED` (Uses the current 10/20 day policy).
    - **Behavior**: Follows a strict **"Diagnose & Solve"** model (Clarify -> Diagnose -> Resolve).
    - **Goal**: Simulates high-quality, professional agent interactions.

### 2. Weighted Scenario Selection
To ensure a realistic distribution of data for analysis, scenarios are weighted by "Tracks":
- **Track A: Defect Pivot (30%)**: Handling claims of broken items.
- **Track B: Preference Probe (30%)**: Handling buyers' remorse via education or resale.
- **Track C: Exceptions (10%)**: Verified Medical/Military or Business Fault cases.
- **Track D: Hard Stop (30%)**: Clear denials for calls > 20 days (Bio-contaminant risk).

### 3. Customer Personas
To avoid repetitive "Angry Customer" patterns, one of six emotional personas is injected into every call:
- **Aggressive**: High tension, demands supervisors (forced for Hard Denials).
- **Sad/Guilt Trip**: Tries to leverage sympathy.
- **Negotiator**: Asks for partial credit if the return is denied.
- **Confused**: Relies heavily on the agent for guidance.
- **Busy/Direct**: Brief turns, value-driven.
- **Passive-Aggressive**: Subtle sarcasm and frustration.

## Usage
Generate VeloFit specific data using the `--profile` flag:
```bash
uv run python src/generate_synthetic_ccai_data.py --count 20 --profile velofit
```
