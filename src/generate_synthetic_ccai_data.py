import os
import json
import random
import uuid
import time
import argparse
from typing import List, Dict, Any

# --- Configuration ---
# FORMAT: "Scenario Name": ["keyword1", "keyword2", "keyword3"]

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
        "config": {
            "company_name": "VeloFit"
        },
        "standard": {
            "Return Policy Inquiry": ["return window", "restocking fee", "shipping label", "30 days", "VeloFit return policy"],
            "Technical Support": ["broken pedal", "wheel wobble", "frame crack", "defective", "VeloFit Racer"],
            "Order Cancellation": ["cancel order", "refund my card", "stop delivery", "changed my mind"]
        },
        "transfer": {}
    }
}

AGENT_POOL = [201, 202, 203, 204, 205]

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
            model = genai.GenerativeModel("gemini-2.0-flash")
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

def generate_raw_conversation(scenario_name: str, keywords: List[str], outcome_instruction: str, is_multi_agent: bool, company_name: str) -> List[Dict[str, str]]:
    """
    Generates a raw conversation list using the LLM.
    """
    
    system_prompt = f"""
    Generate a JSON transcript for a customer service call.
    
    BRAND IDENTITY: You are a customer service agent for {company_name}. Never refer to any other company name.
    
    SCENARIO: {scenario_name}
    
    CRITICAL INSTRUCTION: You MUST naturally weave the following keywords into the dialogue:
    Keywords: {", ".join(keywords)}
    
    {outcome_instruction}
    
    RULES:
    - Start with role "AUTOMATED_AGENT" (e.g. "Thank you for calling {company_name}...").
    - Use role "CUSTOMER" for the caller.
    - {'Use roles "AGENT_1" and "AGENT_2" to show the transfer.' if is_multi_agent else 'Use role "AGENT" for the representative.'}
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

def assemble_call_data(raw_turns: List[Dict[str, str]], scenario_name: str, outcome_status: str, is_multi_agent: bool) -> Dict[str, Any]:
    """
    Process raw turns into the final CCAI Insights JSON format with timestamps.
    """
    conversation_id = str(uuid.uuid4())
    current_time_usec = 0
    entries = []
    
    # 1. Setup IDs for this specific call
    customer_id = random.randint(10000, 99999)
    primary_agent_id = random.choice(AGENT_POOL)
    secondary_agent_id = random.choice([a for a in AGENT_POOL if a != primary_agent_id])
    
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
            
        elif raw_role == "AGENT_2":
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
                "is_transfer": str(is_multi_agent)
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
    standard_scenarios = profile_cfg["standard"]
    transfer_scenarios = profile_cfg["transfer"]

    for i in range(args.count):
        print(f"\nGenerating conversation {i+1}/{args.count}...")
        
        # A. SELECT SCENARIO & KEYWORDS
        scenario_type_label = "STANDARD"
        is_multi_agent = False
        
        if args.profile == "velofit":
            # VELOFIT SPECIFIC WEIGHTING
            # 20% Return Policy Inquiry, 80% other standard scenarios
            # Note: velofit currently has no transfer scenarios in the requirements
            if random.random() < 0.2:
                scenario_name = "Return Policy Inquiry"
                keywords = standard_scenarios[scenario_name]
            else:
                # Pick from remaining standard scenarios
                other_scenarios = {k: v for k, v in standard_scenarios.items() if k != "Return Policy Inquiry"}
                scenario_name, keywords = random.choice(list(other_scenarios.items()))
        else:
            # GENERIC PROFILE LOGIC
            # 80% Standard / 20% Transfer
            if transfer_scenarios and random.random() < 0.2:
                scenario_name, keywords = random.choice(list(transfer_scenarios.items()))
                is_multi_agent = True
                scenario_type_label = "TRANSFER"
            else:
                scenario_name, keywords = random.choice(list(standard_scenarios.items()))

        # B. DETERMINE OUTCOME & SENTIMENT
        # 10% of calls should end unresolved
        if random.random() < 0.1:
            outcome_instruction = "OUTCOME: The issue is NOT resolved. The customer remains frustrated or angry. The agent tries to help but fails. The customer eventually hangs up."
            outcome_status = "Unresolved"
        else:
            outcome_instruction = "OUTCOME: The issue is successfully resolved. The customer expresses gratitude and leaves happy."
            outcome_status = "Resolved"

        # VELOFIT SENTIMENT INJECTION
        if args.profile == "velofit" and scenario_name in ["Return Policy Inquiry", "Order Cancellation"]:
            sentiment_prefix = "SENTIMENT INSTRUCTION: Customer should be frustrated and impatient. Agent must remain professional. "
            outcome_instruction = sentiment_prefix + outcome_instruction

        print(f"  Scenario [{scenario_type_label}]: {scenario_name}")
        print(f"  Outcome: {outcome_status}")
        
        # Phase 1
        raw_turns = generate_raw_conversation(scenario_name, keywords, outcome_instruction, is_multi_agent, company_name)
        if not raw_turns:
            print("  Skipping due to generation error.")
            continue
            
        # Phase 2
        call_data = assemble_call_data(raw_turns, scenario_name, outcome_status, is_multi_agent)
        
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
