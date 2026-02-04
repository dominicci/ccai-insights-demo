# src/legacy/generate_generic_data.py

import os
import json
import random
import uuid
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Retaining the basic LLM Client for legacy support
# In a real scenario, this might import from src.synth_data.llm to avoid code duplication
# But for strict separation, I'll assume we want it standalone or minimal.
# Let's import the new LLM wrapper to verify it works there too.
try:
    from src.synth_data.llm import get_llm_response
except ImportError:
    # Fallback if run directly without package context
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.synth_data.llm import get_llm_response

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

def generate_raw_conversation(scenario_name: str, keywords: List[str], is_multi_agent: bool, company_name: str) -> List[Dict[str, str]]:
    """Legacy generator for generic profile."""
    
    system_prompt = f"""
    Generate a JSON transcript for a customer service call.
    
    BRAND IDENTITY: You are a customer service agent for {company_name}.
    
    SCENARIO: {scenario_name}
    KEYWORDS: {", ".join(keywords)}
    
    RULES:
    - Start with role "AUTOMATED_AGENT".
    - Use role "CUSTOMER" and "AGENT".
    - {'Use "AGENT_2" for transfer.' if is_multi_agent else ''}
    - Output ONLY JSON list.
    """
    
    try:
        raw_response = get_llm_response(system_prompt)
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        return json.loads(clean_response)
    except Exception as e:
        print(f"Error: {e}")
        return []

def process_generic_call(i, output_dir):
    profile = SCENARIO_PROFILES["generic"]
    company_name = profile["config"]["company_name"]
    
    # Selection
    is_multi_agent = False
    if random.random() < 0.2:
        scenario_name, keywords = random.choice(list(profile["transfer"].items()))
        is_multi_agent = True
    else:
        scenario_name, keywords = random.choice(list(profile["standard"].items()))

    raw_data = generate_raw_conversation(scenario_name, keywords, is_multi_agent, company_name)
    
    if raw_data:
        filename = f"generic_call_{uuid.uuid4()}.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump({"entries": raw_data}, f, indent=2)
        print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output", type=str, default="data/legacy_generic")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        for i in range(args.count):
            executor.submit(process_generic_call, i, args.output)

if __name__ == "__main__":
    main()
