# src/synth_data/generators.py

import json
from typing import List, Dict
from .llm import get_llm_response
from .context import OPERATIONAL_CONTEXT

def generate_raw_conversation(
    scenario_name: str, 
    keywords: List[str], 
    outcome_instruction: str, 
    is_multi_agent: bool, 
    company_name: str, 
    scenario_instruction: str = "", 
    second_agent_role: str = "AGENT_2", 
    days: int = 15, 
    kb_text: str = "", 
    agent_limit_instruction: str = "", 
    agent_name: str = "Alex", 
    agent_id: int = 201, 
    prod_item: str = "item", 
    prod_category: str = "product", 
    call_date_str: str = "2024-01-01"
) -> List[Dict[str, str]]:
    """
    Generates a raw conversation list using the LLM.
    """
    
    system_prompt = f"""
    Generate a JSON transcript for a customer service call.
    
    BRAND IDENTITY: You are a customer service agent for {company_name}. Never refer to any other company name.
    
    **Current Date**: {call_date_str}
    **Agent Name**: {agent_name} (ID: {agent_id})
    **Customer Product**: {prod_item} ({prod_category})
    
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
    - The Agent MUST introduce themselves as "{agent_name}".
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
