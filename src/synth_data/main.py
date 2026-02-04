# src/synth_data/main.py

import os
import random
import json
import argparse
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Local Imports
from .config import (
    PRODUCT_CATEGORIES, 
    CUSTOMER_PERSONAS, 
    VELOFIT_SCENARIOS, 
    AGENTS_TIER_200, 
    AGENTS_TIER_300, 
    AGENT_TIER_400, 
    AGENTS_SUPERVISOR
)
from .context import (
    OPERATIONAL_CONTEXT, 
    KB_OUTDATED, 
    KB_UPDATED, 
    get_time_phased_logic, 
    calculate_days_since_launch
)
from .utils import (
    get_agent_name, 
    get_product_category_and_item, 
    random_timestamp_in_range
)
from .generators import generate_raw_conversation
from .processors import assemble_call_data


def process_single_call(i, total_count, args, start_date, end_date, output_dir, company_name):
    print(f"\nGenerating conversation {i+1}/{total_count}...")
    
    # Determine Base Start Time
    base_start_time_usec = random_timestamp_in_range(start_date, end_date)

    # A. SELECT SCENARIO & KEYWORDS
    scenario_type_label = "STANDARD"
    is_multi_agent = False
    
    # 2. Determine Product & Agent (VeloFit Context)
    random.seed(i) # Ensure reproducibility for this call
    agent_id = random.randint(200, 250)
    agent_name = get_agent_name(agent_id)
    random.seed() # Reset seed

    prod_category, prod_item = get_product_category_and_item()
    target_product_type = "pulse" if prod_category == "band" else "standard"

    # --- 3. Time-Phased Logic ---
    call_date = datetime.fromtimestamp(base_start_time_usec / 1_000_000)
    logic_params = get_time_phased_logic(call_date)

    # Adjust Scenario Selection based on Time Phase
    selected_scenario_pair = None # Tuple (name, data)
    
    # Pulse Logic (Jan 1+)
    if logic_params["pulse_probability"] > 0 and random.random() < logic_params["pulse_probability"]:
        # It's a Pulse call
        # Check for error (Compliance Miss)
        if logic_params["force_compliance_miss"] or random.random() < logic_params["error_rate"]:
            # Pick the specific Compliance Miss scenario
            candidates = [(k, v) for k, v in VELOFIT_SCENARIOS.items() if "Pulse" in k and "Compliance Miss" in k]
            if candidates:
                selected_scenario_pair = random.choice(candidates)
            else:
                 # Fallback if specific scenario not found
                 candidates = [(k, v) for k, v in VELOFIT_SCENARIOS.items() if "Pulse" in k]
                 if candidates: selected_scenario_pair = random.choice(candidates)

        else:
            # Correct handling of Pulse (Correct Denial or Defect)
            candidates = [(k, v) for k, v in VELOFIT_SCENARIOS.items() if "Pulse" in k and "Compliance Miss" not in k]
            
            # FILTER: Remove "20+ Days" scenario if technically impossible
            days_since_launch = calculate_days_since_launch(call_date)
            if days_since_launch < 20: 
                 # Cannot have owned it for 20 days yet
                 candidates = [c for c in candidates if "20+ Days" not in c[0]]

            if candidates:
                selected_scenario_pair = random.choice(candidates)
            else:
                 candidates = [(k, v) for k, v in VELOFIT_SCENARIOS.items() if "Pulse" in k]
                 if candidates: selected_scenario_pair = random.choice(candidates)
    
    # If not Pulse (or pre-Jan), pick Standard Scenario
    if not selected_scenario_pair:
         # Standard Logic - Filter out Pulse scenarios
         candidates = [(k, v) for k, v in VELOFIT_SCENARIOS.items() if "Pulse" not in k]
         
         if not candidates:
             # Fallback to anything
             candidates = list(VELOFIT_SCENARIOS.items())
             print(f"[Warn] No Standard scenarios found. Using all.")

         if candidates:
             selected_scenario_pair = random.choice(candidates)

    if not selected_scenario_pair:
         # Extreme fallback
         selected_scenario_pair = list(VELOFIT_SCENARIOS.items())[0]

    scenario_name = selected_scenario_pair[0]
    scenario_data = selected_scenario_pair[1]
    
    # 3.1. Align Product with Scenario
    if "Pulse" in scenario_name:
        if prod_category != "band":
            prod_category = "band"
            prod_item = random.choice(PRODUCT_CATEGORIES["band"])
    else:
        # Check for other implied products in scenario name
        found_match = False
        for cat, items in PRODUCT_CATEGORIES.items():
            if cat.lower() in scenario_name.lower():
                prod_category = cat
                prod_item = random.choice(items)
                found_match = True
                break
        
        # Standard Scenario Fallback
        if not found_match and prod_category == "band":
            # Resample standard category
            while prod_category == "band":
                prod_category, prod_item = get_product_category_and_item()
    
    keywords = scenario_data.get('keywords', [])
    initial_user_message = scenario_data.get('instruction', "")
    scenario_instruction = initial_user_message
        
    # Inject Product into User Message/Prompt
    # We'll prepend the context "I am calling about my [Product Item]..."
    
    scenario_type = scenario_data.get("type", "")
    
    if "shipping" in scenario_type:
        policy_context = """
        **Operational Context**:
        - Shipping Policy: Standard shipping takes 3-5 business days. 
        - Delays: If delayed > 48 hours, offer shipping refund ($10).
        """
    elif "billing" in scenario_type:
        policy_context = """
        **Billing Context**:
        - Price Adjustments: Allowed within 7 days of purchase.
        - Pending Charges: Fall off within 3-5 business days.
        """
    elif "sales" in scenario_type:
        policy_context = """
        **Sales Goal**:
        - Consultative Sales Approach. 
        - Provide expert advice, build trust, and encourage purchase.
        - Do NOT discuss returns unless asked.
        """
    else:
        # Default Return Logic (Pulse vs Standard)
        if target_product_type == "pulse":
            policy_context = f"""
            **Time-Specific Policy Context (VeloBand Pulse)**:
            - Days 1-10: Valid Return (Full Refund).
            - Days 11-20 (Gray Zone): 
                * Buyer's Remorse -> DENY. Offer Education or Marketplace.
                * Defect Claim -> TRANSFER to Tech Support.
                * Proven Business Fault/Medical/Military -> APPROVE Exception.
            - Days 20+ (Hard Stop): Strict Denial. Reason: Hygiene/Bio-contaminant risk. NO EXCEPTIONS.
            """
        else:
            policy_context = f"""
            **Time-Specific Policy Context (Standard Item)**:
            - Standard 90-day return window applies.
            - No hygiene restrictions unless item is visibly damaged/used.
            """
    scenario_instruction += policy_context
    
    # F. CUSTOMER PERSONA INJECTION
    persona_name, persona_desc = random.choice(list(CUSTOMER_PERSONAS.items()))

    if "Hard Denial" in scenario_name and random.random() < 0.4:
        persona_name = "Aggressive"
        persona_desc = CUSTOMER_PERSONAS["Aggressive"]

    scenario_instruction += f"\n\nCUSTOMER PERSONA: {persona_name}\nBEHAVIOR: {persona_desc}\nIMPORTANT: The customer's tone and reaction to the agent must match this persona."

    # B. INITIAL OUTCOME
    outcome_status = "Resolved"
    if random.random() < 0.1:
        outcome_status = "Unresolved"
        
    # C. AGENT SELECTION & LIMITS
    # Determine Customer Tier
    customer_tier = "Standard"
    if random.random() < 0.2:
        customer_tier = "Premium"
        
    # Assign Agents based on Tier
    agent_tier_level = 300 # Default
    
    # 5% chance of Tier 200 (Rookie)
    if random.random() < 0.05:
         agent_id = random.choice(AGENTS_TIER_200)
         agent_tier_level = 200
         agent_name = get_agent_name(agent_id)
         
    # 20% chance of Tier 400 (Star)
    elif random.random() < 0.2:
         agent_id = random.choice(AGENT_TIER_400)
         agent_tier_level = 400
         agent_name = get_agent_name(agent_id)
         
    # Ensure ID consistency if not overridden
    if agent_tier_level == 300 and agent_id not in AGENTS_TIER_300:
         # Map generic ID to 300 names if applicable or keep dynamic
         pass

    # Agent Limits
    agent_limit_instruction = "You have a spending limit of $50 for goodwill credits."
    if agent_tier_level == 200:
        agent_limit_instruction = "You have a spending limit of $25. You MUST consult a knowledge base for every decision."
    elif agent_tier_level == 400:
        agent_limit_instruction = "You have a spending limit of $100. You are an expert agent."
        
    # D. KNOWLEDGE BASE SELECTION
    # If Tier 200, force Outdated KB
    if agent_tier_level == 200:
        kb_text = KB_OUTDATED
    else:
        # Standard/Star agents use Updated KB
        kb_text = KB_UPDATED

    # E. ESCALATION LOGIC
    is_escalated = False
    secondary_agent_id = 501
    second_agent_role = "SUPERVISOR"
    
    # Dynamic Escalation Triggers
    should_escalate = False
    
    # 1. Aggressive Persona + Hard Denial = High Escalation Risk
    if persona_name == "Aggressive" and "Hard Denial" in scenario_name:
        if random.random() < 0.7: should_escalate = True
            
    # 2. Pulse Exception = Medium Esc Risk (Needs Manager Approval sometimes)
    if "Pulse Exception" in scenario_name:
        if random.random() < 0.4: should_escalate = True
            
    # 3. Billing Dispute = Medium Esc Risk
    if "Billing Dispute" in scenario_name:
        if random.random() < 0.3: should_escalate = True
        
    if should_escalate:
        is_escalated = True
        is_multi_agent = True
        secondary_agent_id = random.choice(AGENTS_SUPERVISOR) # 501 or 502
        second_agent_role = "SUPERVISOR"
        outcome_instruction = f"""
            COLLABORATION INSTRUCTION:
            - Start with {agent_name} (Agent 1) trying to help.
            - The Customer becomes dissatisfied or demands a manager.
            - Agent 1 transfers the call: "Please hold while I get a supervisor." (Role: AGENT_1)
            - SUPERVISOR ({get_agent_name(secondary_agent_id)}) joins the call (Role: SUPERVISOR).
            - Supervisor resolves the issue firmly but politely.
            
            CRITICAL SUPERVISOR BEHAVIOR:
            - Supervisor must ACKNOWLEDGE that the policy (10-day/20-day) IS CORRECT and valid.
            - However, Supervisor uses **Manager Override Authority** to grant a 'One-Time Goodwill Exception'.
            - Supervisor must NOT say the item is 'standard' or that the policy doesn't apply.
        """
    else:
        outcome_instruction = "The Agent should resolve the issue autonomously."
        if outcome_status == "Unresolved":
            outcome_instruction = "The Agent cannot solve the issue. The customer hangs up frustrated."

    # F. RANDOM CONTEXT
    days_owned = random.randint(3, 45)
    
    # GENERATION
    raw_convo = generate_raw_conversation(
        scenario_name=scenario_name,
        keywords=keywords,
        outcome_instruction=outcome_instruction,
        is_multi_agent=is_multi_agent,
        company_name=company_name,
        scenario_instruction=scenario_instruction,
        second_agent_role=second_agent_role,
        days=days_owned,
        kb_text=kb_text,
        agent_limit_instruction=agent_limit_instruction,
        agent_name=agent_name,
        agent_id=agent_id,
        prod_item=prod_item,
        prod_category=prod_category,
        call_date_str=call_date.strftime("%Y-%m-%d")
    )
    
    if not raw_convo:
        return
        
    # ASSEMBLY
    final_json = assemble_call_data(
        raw_turns=raw_convo,
        scenario_name=scenario_name,
        outcome_status=outcome_status,
        is_multi_agent=is_multi_agent,
        customer_tier=customer_tier,
        is_escalated=is_escalated,
        primary_agent_id=agent_id,
        secondary_agent_id=secondary_agent_id,
        base_start_time_usec=base_start_time_usec,
        phase=logic_params.get("phase", "unknown"),
        product_category=prod_category,
        product_item=prod_item,
        agent_name=agent_name
    )
    
    # SAVE
    filename = f"synthetic_call_{final_json['conversation_info']['conversation_id']}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(final_json, f, indent=2)
        
    print(f"  [Success] Saved conversation {i+1} to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic customer service calls (CCA Insights).")
    parser.add_argument("--count", type=int, default=5, help="Number of conversations to generate.")
    parser.add_argument("--output", type=str, default="data/synthetic_transcripts", help="Output directory.")
    parser.add_argument("--profile", type=str, default="velofit", help="Scenario profile (velofit only).")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel worker threads.")
    
    # Date Range Arguments
    parser.add_argument("--start-date", type=str, default="2026-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, default="2026-01-31", help="End date (YYYY-MM-DD).")
    
    args = parser.parse_args()
    
    # Validate Profile
    if args.profile != "velofit":
        print("[Error] Only 'velofit' profile is supported in this version.")
        return

    company_name = "VeloFit"
    
    # Create Output Directory
    os.makedirs(args.output, exist_ok=True)
    print(f"--- Starting Synthetic Data Generation for {args.count} calls [Profile: {args.profile}] ---")
    print(f" Output Directory: {args.output}")
    print(f" Parallel Workers: {args.workers}")
    print(f" Date Range: {args.start_date} to {args.end_date}")
    
    # Parse Dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        print("[Error] Invalid date format. Use YYYY-MM-DD.")
        return

    print(f"Spinning up {args.workers} workers...\n")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i in range(args.count):
            futures.append(
                executor.submit(
                    process_single_call, 
                    i, 
                    args.count, 
                    args, 
                    start_date, 
                    end_date, 
                    args.output,
                    company_name
                )

            )
        
        # Check for errors
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"[Critical Error] Thread failed: {e}")

            
    print("\n--- Batch generation complete ---")

if __name__ == "__main__":
    main()
