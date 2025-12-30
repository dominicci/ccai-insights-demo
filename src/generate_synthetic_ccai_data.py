import os
import json
import random
import uuid
import time
import argparse
from typing import List, Dict, Any

# --- GLOBAL GROUNDING RULES ---
OPERATIONAL_CONTEXT = """
1. TIMELINE REALITY: 
   - Standard shipping takes 3-5 business days. 
   - The Customer HAS the item in their possession.
   - "Unopened Box" excuse is INVALID unless combined with "Late Gift" exception.

2. RETURN POLICY LOGIC (THE 20-DAY SPLIT):
   - Days 1-10: Valid Return (Full Refund).
   - Days 11-20 (Gray Zone): 
       * Buyer's Remorse -> DENY. Offer Education or Marketplace.
       * Defect Claim -> TRANSFER to Tech Support.
       * Proven Business Fault/Medical/Military -> APPROVE Exception.
   - Days 20+ (Hard Stop): Strict Denial. Reason: Hygiene/Bio-contaminant risk. NO EXCEPTIONS.

3. AGENT WORKFLOW (MANDATORY):
   - Step 1: CLARIFY. You MUST ask "What specifically is the issue?" or "Is it a technical fault or a preference?"
   - Step 2: DIAGNOSE. 
       * If "Defect": Pivot to troubleshooting/transfer.
       * If "Preference": Educate on features.
   - Step 3: RESOLVE.
       * Only offer credits for Business Faults or Verified Exceptions.
       * For Buyer's Remorse, offer the "Resale Marketplace" as the solution.

4. FINANCIAL LIMITS (STRICT):
   - Goodwill Credits for "Changing Mind" are BANNED.
   - Max Credit for Business Fault (Website Error): $50 (Tier 300) / $100 (Tier 400).
   - Max Credit for Verified Exception (Medical/Military): Full Refund/Store Credit allowed.

5. EMOTIONAL LOGIC:
   - If an Agent denies a request, the Customer MUST react according to their assigned PERSONA (e.g., Sad, Angry, or Negotiating).
   - Do NOT default to "Angry" unless the persona is "Aggressive".
   - Customers should ONLY switch to "Positive" if the Agent offers a tangible solution.
"""

# --- KNOWLEDGE BASES ---
KB_OUTDATED = """
INTERNAL KNOWLEDGE BASE [LAST UPDATED: 2023]
--------------------------------------------
RETURN POLICY:
- All items are eligible for return within 90 days of delivery.
- Items must be in original condition.
"""

KB_UPDATED = """
INTERNAL KNOWLEDGE BASE [LAST UPDATED: TODAY]
---------------------------------------------
RETURN POLICY ALERT (VELOBAND PULSE):
- CRITICAL: VeloBand Pulse has a STRICT 10-DAY return window.
- REASON (Days 11-20): Device activation links hardware to user account.
- REASON (Days 20+): Hygiene protocols prevent resale.
- The standard 90-day policy does NOT apply to the Pulse.

STANDARD POLICY (Other Items):
- Jerseys/Shoes: 90 days.
"""

SCENARIO_PROFILES = {
    "generic": {
        "config": {
            "company_name": "StellarTech"
        },
        "standard": {
            "Order Status Inquiry": ["tracking number", "scheduled delivery", "shipping update", "package location"],
            "Product Return Request": ["return label", "original packaging", "refund policy", "RMA number"],
            "Store Hours & Location": ["opening time", "closing time", "weekend hours", "parking availability"],
            "Password Reset Help": ["cant login", "reset link", "forgot password", "username", "locked out"],
            "Product Feature Questions": ["battery life", "compatibility", "warranty", "user manual", "features"]
        },
        "transfer": {
            "Billing Dispute Escalation": ["speak to a supervisor", "overcharged", "incorrect amount", "unauthorized charge", "credit back"],
            "Service Cancellation Retention": ["cancel my subscription", "too expensive", "better offer", "retention team", "close account"],
            "Technical Support Tier 2": ["hardware failure", "error code", "advanced troubleshooting", "tier 2", "technical specialist"]
        }
    },
    "velofit": {
        "config": { "company_name": "VeloFit" },
        "scenarios": {
            # --- TRACK A: THE DEFECT PIVOT ---
            "Pulse Defect (Pivot to Support)": {
                "keywords": ["not working", "won't sync", "broken", "defective"],
                "instruction": "Customer claims item is defective (Day 11-15). Agent asks clarifying questions. Customer agrees to transfer to Tech Support."
            },
            "Pulse Defect (Refuses Support)": {
                "keywords": ["just want money", "don't have time", "garbage", "refund"],
                "instruction": "Customer claims defect but refuses troubleshooting. Agent explains policy: 'Without verification, standard 10-day rule applies.' Agent DENIES return."
            },

            # --- TRACK B: THE PREFERENCE PROBE ---
            "Pulse Preference (Education Win)": {
                "keywords": ["hard to use", "confusing", "expected more", "user manual"],
                "instruction": "Customer doesn't like the interface (Day 12). Agent asks specific usage questions. Agent explains how to use the feature correctly. Customer is satisfied."
            },
            "Pulse Preference (Marketplace Pivot)": {
                "keywords": ["changed my mind", "don't use it", "impulse buy", "resale"],
                "instruction": "Customer simply changed mind (Day 18). Agent denies return. Offers 'VeloFit Resale Marketplace'. Customer accepts."
            },

            # --- TRACK C: THE EXCEPTIONS (High Tier Only) ---
            "Pulse Exception (Medical/Military)": {
                "keywords": ["hospital", "deployment", "emergency", "surgery"],
                "instruction": "Customer missed window due to legitimate Life Event (Day 11-20). Agent verifies and grants ONE-TIME Full Credit/Refund exception."
            },
            "Pulse Exception (Business Fault)": {
                "keywords": ["website error", "couldn't print", "system down", "screenshot"],
                "instruction": "Customer tried to return on Day 9 but website crashed. Agent validates the known outage. Approves return."
            },

            # --- TRACK D: THE HARD STOP (20+ DAYS) ---
            "Pulse Hard Denial (20+ Days)": {
                "keywords": ["hygiene", "policy", "too late", "sanitation"],
                "instruction": "Customer calls on Day 25. Agent strictly denies based on Hygiene/Bio-safety. No credits offered."
            }
        }
    }
}

CUSTOMER_PERSONAS = {
    "Aggressive": "Customer is angry, raises voice, demands manager, threatens bad reviews.",
    "Sad/Guilt Trip": "Customer is disappointed and sad. Uses guilt ('I really needed this money', 'I've been loyal'). Does not yell.",
    "Negotiator": "Customer is calm and rational. Tries to bargain for partial credit or a discount code. Treats policy as a negotiation.",
    "Confused": "Customer acts helpless/overwhelmed. Claims they didn't understand the website or policy. asks 'Can't you just help me?'.",
    "Busy/Direct": "Customer is impatient. Cuts off small talk. Wants a yes/no answer immediately. Annoyed by wasted time.",
    "Passive-Aggressive": "Customer gives short, cold answers ('Fine', 'Whatever'). Accepts defeat but makes snide remarks."
}

# 200-Level: Risky/Untrained (6 Agents)
# Assigned to: Misinformation & Unauthorized Overrides
AGENTS_TIER_200 = [201, 202, 203, 204, 205, 206] # Risky, Outdated KB, $25 Limit

# 300-Level: Mixed Bag (4 Agents)
# Split: 301/302 (Not up-to-date), 303/304 (Up-to-date but blunt)
AGENTS_TIER_300 = [301, 302, 303, 304]           # Mixed KB, $50 Limit

# 400-Level: The Star (2 Agents)
# Up-to-date, effectively de-escalates. Can handle any non-fail scenario.
AGENT_TIER_400 = [401, 402]                      # Stars, Updated KB, $100 Limit

# 500-Level: Supervisors (2 Agents)
# Only assigned as 'secondary_agent' in escalations.
AGENTS_SUPERVISOR = [501, 502]                   # Supervisor, Updated KB, $150 Limit

# --- LLM Clients ---

def get_llm_response(prompt: str) -> str:
    """
    Tries to get a response from Gemini (Google) first, then OpenAI.
    Returns the raw string content.
    """
    
    # 1. Try Google Gemini
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if google_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            model = genai.GenerativeModel("gemini-2.5-pro")
            response = model.generate_content(prompt)
            print(" Using Google Gemini...")
            return response.text
        except ImportError:
            print("  [Warning] google-generativeai not installed. Skipping Gemini.")
        except Exception as e:
            print(f"  [Error] Gemini generation failed: {e}")

    # 2. Try OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            completion = client.chat.completions.create(
                model="gpt-4o-mini",  # or gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates synthetic data in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            print(" Using OpenAI...")
            return completion.choices[0].message.content
        except ImportError:
            print("  [Warning] openai library not installed. Skipping OpenAI.")
        except Exception as e:
            print(f"  [Error] OpenAI generation failed: {e}")

    raise EnvironmentError("No valid API key found (GOOGLE_API_KEY or OPENAI_API_KEY) or libraries missing.")

# --- Phase 1: Generation ---

def generate_raw_conversation(scenario_name: str, keywords: List[str], outcome_instruction: str, is_multi_agent: bool, company_name: str, scenario_instruction: str = "", second_agent_role: str = "AGENT_2", days: int = 15, kb_text: str = "", agent_limit_instruction: str = "") -> List[Dict[str, str]]:
    """
    Generates a raw conversation list using the LLM.
    """
    
    system_prompt = f"""
    Generate a JSON transcript for a customer service call.
    
    BRAND IDENTITY: You are a customer service agent for {company_name}. Never refer to any other company name.
    
    OPERATIONAL CONTEXT:
    {OPERATIONAL_CONTEXT}
    
    KNOWLEDGE BASE:
    {kb_text}
    
    YOUR AUTHORITY:
    {agent_limit_instruction}
    
    CONTEXT: Customer received item {days} days ago.
    
    SCENARIO: {scenario_name}
    
    CRITICAL INSTRUCTION: You MUST naturally weave the following keywords into the dialogue:
    Keywords: {", ".join(keywords)}
    
    {scenario_instruction}
    
    {outcome_instruction}
    
    RULES:
    - Start with role "AUTOMATED_AGENT" (e.g. "Thank you for calling {company_name}...").
    - Use role "CUSTOMER" for the caller.
    - {'Use roles "AGENT_1" and "' + second_agent_role + '" to show the transfer.' if is_multi_agent else 'Use role "AGENT" for the representative.'}
    - Output ONLY a JSON list of objects: {{"role": "...", "text": "..."}}
    - Do NOT include markdown formatting like ```json ... ```. Just the raw JSON string.
    """

    try:
        raw_response = get_llm_response(system_prompt)
        # Clean up potential markdown formatting if the LLM ignores instructions
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        return json.loads(clean_response)
    
    except json.JSONDecodeError:
        print("  [Error] Failed to parse LLM output as JSON.")
        return []
    except Exception as e:
        print(f"  [Error] Generation failed: {e}")
        return []

# --- Phase 2: Assembly ---

def assemble_call_data(raw_turns: List[Dict[str, str]], scenario_name: str, outcome_status: str, is_multi_agent: bool, customer_tier: str = "Standard", is_escalated: bool = False, primary_agent_id: int = 201, secondary_agent_id: int = 501) -> Dict[str, Any]:
    """
    Process raw turns into the final CCAI Insights JSON format with timestamps.
    """
    conversation_id = str(uuid.uuid4())
    current_time_usec = 0
    entries = []
    
    # 1. Setup IDs for this specific call
    customer_id = random.randint(10000, 99999)
    # primary_agent_id and secondary_agent_id are now passed from main loop
    
    for turn in raw_turns:
        raw_role = turn.get("role", "").upper()
        text = turn.get("text", "")
        
        # 1. Determine Speaker ID and API Role
        if raw_role in ["AUTOMATED_AGENT", "IVR_SYSTEM"]:
            final_role = "AUTOMATED_AGENT"
            user_id = 0
            speaker_id = "AUTOMATED_AGENT"
            
        elif raw_role == "CUSTOMER":
            final_role = "CUSTOMER"
            user_id = customer_id
            speaker_id = "CUSTOMER"
            
        elif raw_role in ["AGENT"]:
            final_role = "AGENT"
            user_id = primary_agent_id
            speaker_id = "AGENT"
        
        elif raw_role == "AGENT_1":
            final_role = "AGENT"
            user_id = primary_agent_id
            speaker_id = "AGENT"
            
        elif raw_role == "AGENT_2" or raw_role == "SUPERVISOR":
            final_role = "AGENT" # API requires "AGENT"
            user_id = secondary_agent_id # ID change indicates transfer
            speaker_id = "AGENT"
        else:
             # Default fallback
             final_role = "CUSTOMER"
             user_id = customer_id
             speaker_id = "CUSTOMER"

        # 2. Calculate Duration (1 sec = 1,000,000 usec)
        # Estimate: 15 characters per second
        char_count = len(text)
        if char_count == 0:
            continue
            
        duration_usec = int((char_count / 15) * 1_000_000)
        
        # Ensure min duration (0.5s) to avoid zero-length glitches
        if duration_usec < 500_000:
            duration_usec = 500_000

        # 3. Build Entry
        entries.append({
            "text": text,
            "speakerId": speaker_id,
            "role": final_role,
            "user_id": user_id,
            "start_timestamp_usec": current_time_usec
        })

        # 4. Increment Time (add a small pause between speakers)
        current_time_usec += duration_usec + 200_000 # +0.2s pause

    # Inject conversation-level metadata
    final_object = {
        "conversation_info": {
            "conversation_id": conversation_id,
            "metadata": {
                "call_type": scenario_name,
                "customer_sentiment": "Negative" if outcome_status == "Unresolved" else "Positive", # Simple inference
                "outcome": outcome_status,
                "generated_by": "synthetic_script_v3_quality",
                "is_transfer": str(is_multi_agent),
                "customer_tier": customer_tier,
                "is_escalated": is_escalated
            }
        },
        "entries": entries
    }
    
    return final_object

# --- Main Loop ---

import pathlib

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic call center data.")
    parser.add_argument("--count", type=int, default=10, help="Number of synthetic calls to generate")
    parser.add_argument("--profile", type=str, default="generic", choices=["generic", "velofit"], help="Configuration profile to use")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated files (defaults to data/synthetic_transcripts)")
    
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        base_output_dir = pathlib.Path(args.output_dir)
    else:
        # Default: ProjectRoot/data/synthetic_transcripts
        # Script is in ProjectRoot/src/
        script_dir = pathlib.Path(__file__).parent.resolve()
        base_output_dir = script_dir.parent / "data" / "synthetic_transcripts"

    # Append profile subfolder for organization
    output_dir = base_output_dir / args.profile

    print(f"--- Starting Synthetic Data Generation for {args.count} calls [Profile: {args.profile}] ---")
    print(f" Output Directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Load scenarios for the selected profile
    profile_cfg = SCENARIO_PROFILES[args.profile]
    company_name = profile_cfg["config"]["company_name"]
    
    # Standard and Transfer are used by the generic profile
    standard_scenarios = profile_cfg.get("standard", {})
    transfer_scenarios = profile_cfg.get("transfer", {})
    
    # Scenarios is used by the velofit profile
    all_scenarios = profile_cfg.get("scenarios", {})

    for i in range(args.count):
        print(f"\nGenerating conversation {i+1}/{args.count}...")
        
        # A. SELECT SCENARIO & KEYWORDS
        scenario_type_label = "STANDARD"
        is_multi_agent = False
        
        if args.profile == "velofit":
            # VELOFIT SPECIFIC SELECTION (Weighted Tracks)
            weights = {
                "Pulse Defect (Pivot to Support)": 0.15,
                "Pulse Defect (Refuses Support)": 0.15,
                "Pulse Preference (Education Win)": 0.15,
                "Pulse Preference (Marketplace Pivot)": 0.15,
                "Pulse Exception (Medical/Military)": 0.05,
                "Pulse Exception (Business Fault)": 0.05,
                "Pulse Hard Denial (20+ Days)": 0.30
            }
            scenario_name = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            
            scenario_data = all_scenarios[scenario_name]
            keywords = scenario_data["keywords"]
            scenario_instruction = scenario_data.get("instruction", "")

            # F. CUSTOMER PERSONA INJECTION
            # Randomly select a persona to vary the reaction
            persona_name, persona_desc = random.choice(list(CUSTOMER_PERSONAS.items()))

            # Override: Hard Denials naturally trigger aggression more often
            if "Hard Denial" in scenario_name and random.random() < 0.4:
                persona_name = "Aggressive"
                persona_desc = CUSTOMER_PERSONAS["Aggressive"]

            # Append to instruction
            scenario_instruction += f"\n\nCUSTOMER PERSONA: {persona_name}\nBEHAVIOR: {persona_desc}\nIMPORTANT: The customer's tone and reaction to the agent must match this persona."
        else:
            # GENERIC PROFILE LOGIC
            # 80% Standard / 20% Transfer
            if transfer_scenarios and random.random() < 0.2:
                scenario_name, keywords = random.choice(list(transfer_scenarios.items()))
                is_multi_agent = True
                scenario_type_label = "TRANSFER"
                scenario_instruction = ""
            else:
                scenario_name, keywords = random.choice(list(standard_scenarios.items()))
                scenario_instruction = ""
            persona_name = "Standard" # Fallback for generic

        # B. INITIAL OUTCOME (Refined by profile/path later)
        if random.random() < 0.1:
            outcome_instruction = "OUTCOME: The issue is NOT resolved. The customer remains frustrated or angry. The agent tries to help but fails. The customer eventually hangs up."
            outcome_status = "Unresolved"
        else:
            outcome_instruction = "OUTCOME: The issue is successfully resolved. The customer expresses gratitude and leaves happy."
            outcome_status = "Resolved"

        # VELOFIT SENTIMENT INJECTION
        if args.profile == "velofit" and "VeloBand Pulse" in scenario_name:
            sentiment_prefix = "SENTIMENT INSTRUCTION: Customer should be frustrated and impatient. Agent must remain professional. "
            outcome_instruction = sentiment_prefix + outcome_instruction

        # MODIFIERS (Dynamic Logic V4 - Final Grounding)
        customer_tier = "Standard"
        is_escalated = False
        days = 15 # Default
        primary_agent_id = 201 # Default
        secondary_agent_id = random.choice(AGENTS_SUPERVISOR)
        agent_limit_instruction = "Max Credit $25." # Default

        if args.profile == "velofit":
            # 1. Loyalty Check (20% Platinum)
            if random.random() < 0.2:
                customer_tier = "Platinum"

            # --- THE GLOBAL ROUTING FORK (50/50 Path Split) ---
            if random.random() < 0.5:
                # PATH A: ROOKIE / OUTDATED (The Problem Path)
                kb_text = KB_OUTDATED
                primary_agent_id = random.choice(AGENTS_TIER_200)
                agent_limit_instruction = "$25 Max."
                
                # SCENARIO OVERRIDE: Discard diagnostic instructions for Rookies
                scenario_instruction = (
                    "SCENARIO: Customer wants to return item. Agent is a Rookie (Tier 200) using an Outdated 90-day policy. "
                    "Agent MUST NOT ask diagnostic questions, MUST NOT troubleshoot, and MUST NOT pivot to Tech Support. "
                    "Agent simply approves the return immediately because it is within 90 days."
                )
                keywords = ["90 days", "return label", "no problem", "emailing you"]
                outcome_instruction = "OUTCOME: Agent approves return immediately without friction."
                outcome_status = "Resolved"
                
                # Conflict Timeline (12-89 days)
                # This ensures the policies conflict: Old Policy (90 days) says Yes, New (10/20) says No.
                days = random.randint(12, 89)
            else:
                # PATH B: STANDARD / UPDATED (The Competent Path)
                kb_text = KB_UPDATED
                
                # B. SCENARIO & AGENT MAPPING (Competent Logic)
                if "Exception" in scenario_name:
                    # Only Stars handle exceptions well
                    primary_agent_id = random.choice(AGENT_TIER_400)
                    agent_limit_instruction = "Exception Case: Auth to offer Full Refund/Credit if verified."
                    outcome_status = "Resolved"

                elif "Hard Denial" in scenario_name:
                    # Tier 300+ handles denials professionally
                    primary_agent_id = random.choice(AGENTS_TIER_300 + AGENT_TIER_400)
                    agent_limit_instruction = "Strict Denial. No Credits."
                    outcome_status = "Unresolved" # Denied

                elif "Defect" in scenario_name or "Preference" in scenario_name:
                    # Tier 300+ handles the Core Diagnostic Work
                    primary_agent_id = random.choice(AGENTS_TIER_300 + AGENT_TIER_400)

                    if primary_agent_id in AGENTS_TIER_300:
                        agent_limit_instruction = "Max Credit $50 (Business Fault Only). Focus on Troubleshooting."
                    else:
                        agent_limit_instruction = "Max Credit $100 (Business Fault Only). Focus on Retention/Marketplace."
                    
                    if "Refuses Support" in scenario_name:
                        outcome_status = "Unresolved"
                    else:
                        outcome_status = "Resolved"
                else:
                    # Fallback
                    primary_agent_id = random.choice(AGENTS_TIER_300)
                    agent_limit_instruction = "Max Credit $50."
                    outcome_status = "Resolved"

                # Update outcome instruction based on status
                if outcome_status == "Unresolved":
                    outcome_instruction = "OUTCOME: Agent CLARIFIES and DIAGNOSES, but ultimately DENIES the return based on policy. Customer accepts (or grumbles) but the return is NOT approved."
                else:
                    outcome_instruction = "OUTCOME: Agent successfully resolves the issue (via troubleshooting, marketplace education, or approved exception)."

                # D. TIMELINE CALCULATION (Competent Alignment)
                if "Hard Denial" in scenario_name:
                    days = random.randint(21, 45) # 20+ Days
                elif "Defect" in scenario_name or "Preference" in scenario_name or "Exception" in scenario_name:
                    days = random.randint(11, 19) # Gray Zone
                elif "Valid/Happy" in scenario_name:
                    days = random.randint(2, 9)   # Safe Zone
                else:
                    days = random.randint(11, 15) # Default Fallback

            # Loyalty Instruction Enrichment (Applies to both paths)
            if customer_tier == "Platinum":
                loyalty_instruction = " Customer is VIP. Agent offers perks/overrides."
                scenario_instruction += loyalty_instruction
        else:
            kb_text = "" # No KB for generic profile
            agent_limit_instruction = ""

        print(f"  Scenario [{scenario_type_label}]: {scenario_name}")
        print(f"  Persona: {persona_name}")
        print(f"  Outcome: {outcome_status}")
        print(f"  Tier: {customer_tier}")
        print(f"  Agent ID: {primary_agent_id}")
        if is_escalated: print(f"  Status: Escalated to {secondary_agent_id}")
        
        # Phase 1
        raw_turns = generate_raw_conversation(scenario_name, keywords, outcome_instruction, is_multi_agent, company_name, scenario_instruction, second_agent_role="SUPERVISOR" if is_escalated else "AGENT_2", days=days, kb_text=kb_text, agent_limit_instruction=agent_limit_instruction)
        
        if not raw_turns:
            print("  Skipping due to generation error.")
            continue
            
        # Phase 2
        call_data = assemble_call_data(raw_turns, scenario_name, outcome_status, is_multi_agent, customer_tier, is_escalated, primary_agent_id, secondary_agent_id)
        
        # Save
        conv_id = call_data["conversation_info"]["conversation_id"]
        filename = output_dir / f"synthetic_call_{conv_id}.json"
        
        with open(filename, "w") as f:
            json.dump(call_data, f, indent=2)
            
        print(f"  Saved to {filename}")
        
        # Sleep briefly to be nice to the API
        time.sleep(1)
        
    print("\n--- Batch generation complete ---")


if __name__ == "__main__":
    main()
