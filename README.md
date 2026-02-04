# CCAI Synthetic Data Generator (VeloFit Edition)

A toolkit for generating high-quality synthetic contact center transcripts (JSON) to demonstrate Google Cloud CCAI Insights features like Topic Modeling, Smart Highlights, and Quality AI.

This project simulates **VeloFit**, a fictional fitness wearable company, generating realistic customer service calls including "Happy Path" returns, policy disputes, and complex multi-agent escalations.

## 🚀 Quick Start

**1. Setup Environment**
```bash
export GOOGLE_API_KEY="your-key"
```

**2. Run Generator**
```bash
# Generates 10 synthetic calls
uv run python -m src.synth_data.main --count 10
```

**3. Check Output**
Files will be generated in `data/synthetic_transcripts/`.

---

## 📂 Project Structure

*   `src/synth_data/`: **Core Package**. Contains the modularized generation logic.
    *   `main.py`: Entry point.
    *   `config.py`: Profiles, personas, and product data.
    *   `generators.py`: Prompt engineering and LLM iteration.
*   `src/ops/`: Operational utilities (splitting files, data enrichment).
*   `src/validation/`: Schemes validation scripts.
*   `src/reporting/`: Dashboard and stats generation.
*   `src/legacy/`: Deprecated scripts (e.g., Generic profile).

## 📖 Documentation

*   [**PROCESS.md**](PROCESS.md): Detailed **Order of Execution** and workflow guide.
*   [**SETUP_GUIDE.md**](docs/SETUP_GUIDE.md): Manual configuration steps for Google Cloud Console.
*   [**VELOFIT_SIMULATION.md**](docs/VELOFIT_SIMULATION.md): Details on the simulation logic (Time-phased error rates, personas).

## ✨ Key Features

*   **Valid CCAI JSON**: Strictly compliant with Google Cloud ingestion formats.
*   **Time-Phased Logic**: Simulates "Steady State", "Launch Chaos", and "Learning Curve" phases based on call dates.
*   **Dynamic Personas**: Agents and Customers behave according to specific personas (e.g., "Aggressive Customer", "Rookie Agent").
*   **Escalation Simulation**: Automates realistic transfers from Tier 1 agents to Supervisors.
