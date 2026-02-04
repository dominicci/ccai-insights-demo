# src/synth_data/processors.py

import uuid
import random
from typing import List, Dict, Any

def assemble_call_data(
    raw_turns: List[Dict[str, str]], 
    scenario_name: str, 
    outcome_status: str, 
    is_multi_agent: bool, 
    customer_tier: str = "Standard", 
    is_escalated: bool = False, 
    primary_agent_id: int = 201, 
    secondary_agent_id: int = 501, 
    base_start_time_usec: int = 0, 
    phase: str = "unknown", 
    product_category: str = "unknown", 
    product_item: str = "unknown", 
    agent_name: str = "Agent"
) -> Dict[str, Any]:
    """
    Process raw turns into the final CCAI Insights JSON format with timestamps.
    """
    conversation_id = str(uuid.uuid4())
    current_time_usec = base_start_time_usec
    entries = []
    
    # 1. Setup IDs for this specific call
    customer_id = random.randint(10000, 99999)
    # primary_agent_id and secondary_agent_id are passed from main loop
    
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
            "agent_id": str(primary_agent_id), # Native Field
            "agent_info": [ # Sibling to metadata
                {
                    "agent_id": str(primary_agent_id),
                    "display_name": agent_name
                }
            ],
            "metadata": {
                "call_type": scenario_name,
                "customer_sentiment": "Positive", # Placeholder, ideally derived from content
                "outcome": outcome_status,
                "generated_by": "synthetic_script_v3_quality",
                "is_transfer": str(is_multi_agent),
                "customer_tier": customer_tier,
                "is_escalated": is_escalated,
                "agent_id": str(primary_agent_id),
                "phase": phase,
                "product_category": product_category,
                "product_item": product_item,
                "agent_name": agent_name,
                "agent_info": [
                    {
                        "agent_id": str(primary_agent_id),
                        "display_name": agent_name
                    }
                ],
                # CAMELCASE DUPLICATE FOR BULK IMPORT (Strict REST Spec)
                "qualityMetadata": {
                    "agentInfo": [
                        {
                            "agentId": str(primary_agent_id),
                            "displayName": agent_name
                        }
                    ]
                }
            }
        },
        "entries": entries
    }
    
    return final_object
