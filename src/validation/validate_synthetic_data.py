import json
import os
import glob
from pathlib import Path

def validate_synthetic_data(directory="data/synthetic_transcripts/velofit"):
    """
    Performs a Logic Audit on generated synthetic VeloFit calls.
    """
    path = Path(directory)
    if not path.exists():
        print(f"Error: Directory {directory} does not exist.")
        return

    files = list(path.glob("*.json"))
    if not files:
        print(f"No JSON files found in {directory}.")
        return

    total_calls = len(files)
    tier_200_calls = 0
    tier_300_400_calls = 0
    
    # KB Integrity
    new_kb_mentions_by_rookie = 0
    
    # Behavioral Consistency
    tier_200_turns = []
    tier_300_400_turns = []
    
    # Outcome Distribution
    tier_200_outcomes = {"Resolved": 0, "Escalated": 0, "Denied": 0, "Unresolved": 0}
    tier_300_400_outcomes = {"Resolved": 0, "Escalated": 0, "Denied": 0, "Unresolved": 0}

    # Scenario Distribution
    scenario_counts = {}

    new_kb_keywords = ["10 days", "hygiene", "bio-contaminant", "sanitation", "policy change"]

    print(f"--- Logic Audit of {total_calls} Calls in {directory} ---")

    for file_path in files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

            # --- NEW PARSING LOGIC (Aligned with CCAI Format) ---
            metadata = data.get("conversation_info", {}).get("metadata", {})
            scenario_name = metadata.get("call_type", "Unknown")
            outcome = metadata.get("outcome", "Unknown")
            entries = data.get("entries", [])
            num_turns = len(entries)

            # Track Scenario Distribution
            scenario_counts[scenario_name] = scenario_counts.get(scenario_name, 0) + 1

            # Find primary agent ID (from the first AGENT entry)
            agent_id = 0
            for entry in entries:
                if entry.get("role") == "AGENT":
                    agent_id = entry.get("user_id", 0)
                    break

            # 1. Routing Split Check
            is_rookie = 201 <= agent_id <= 206
            if is_rookie:
                tier_200_calls += 1
                tier_200_turns.append(num_turns)
                if outcome in tier_200_outcomes:
                    tier_200_outcomes[outcome] += 1
                
                # 2. KB Integrity Check
                full_text = " ".join([t.get("text", "").lower() for t in entries])
                for kw in new_kb_keywords:
                    if kw in full_text:
                        new_kb_mentions_by_rookie += 1
                        break
            else:
                tier_300_400_calls += 1
                tier_300_400_turns.append(num_turns)
                if outcome in tier_300_400_outcomes:
                    tier_300_400_outcomes[outcome] += 1

    # --- Print Results ---

    # 1. Routing Split
    rookie_pct = (tier_200_calls / total_calls * 100) if total_calls > 0 else 0
    print(f"\n[1] ROUTING SPLIT CHECK")
    print(f"    Total Calls: {total_calls}")
    print(f"    Tier 200 (Rookies): {tier_200_calls} ({rookie_pct:.1f}%)")
    print(f"    Tier 300/400 (Standard): {tier_300_400_calls} ({100-rookie_pct:.1f}%)")
    print(f"    Target Routing Split: 50/50")

    # 2. KB Integrity
    print(f"\n[2] KB INTEGRITY CHECK (New Terms used by Rookies)")
    print(f"    Violations Found: {new_kb_mentions_by_rookie}")
    print(f"    Target Violations: 0 (Rookies should only know 90-day policy)")

    # 3. Behavioral Consistency
    avg_200 = sum(tier_200_turns) / len(tier_200_turns) if tier_200_turns else 0
    avg_300 = sum(tier_300_400_turns) / len(tier_300_400_turns) if tier_300_400_turns else 0
    print(f"\n[3] BEHAVIORAL CONSISTENCY (Avg Turns)")
    print(f"    Rookie Avg Length: {avg_200:.1f} turns")
    print(f"    Standard Avg Length: {avg_300:.1f} turns")
    print(f"    Hypothesis: Rookies should be significantly shorter (Blind Approvals)")

    # 4. Outcome Distribution
    print(f"\n[4] OUTCOME DISTRIBUTION")
    print(f"    ROOKIES:   {tier_200_outcomes}")
    print(f"    STANDARD:  {tier_300_400_outcomes}")
    print(f"    Hypothesis: Rookies should have a much higher 'Resolved' rate.")

    # 5. Scenario Distribution
    print(f"\n[5] SCENARIO DISTRIBUTION")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"    - {scenario}: {count} ({(count/total_calls*100):.1f}%)")
    print(f"    Target: Exceptions should be ~10% total.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit synthetic CCAI data for logic and behavioral consistency.")
    parser.add_argument("dir", nargs="?", default="data/synthetic_transcripts/velofit", help="Directory containing JSON transcripts (default: data/synthetic_transcripts/velofit)")
    args = parser.parse_args()
    validate_synthetic_data(args.dir)
