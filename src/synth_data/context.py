# src/synth_data/context.py

from datetime import datetime

# --- GLOBAL GROUNDING RULES ---
OPERATIONAL_CONTEXT = """
1. TIMELINE REALITY: 
   - Standard shipping takes 3-5 business days. 
   - The Customer HAS the item in their possession.
   - "Unopened Box" excuse is INVALID unless combined with "Late Gift" exception.

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

def get_time_phased_logic(call_date: datetime) -> dict:
    """
    Returns logic parameters based on the date.
    Phases:
    1. Steady State (Dec 1 - Dec 31): Low error rate, Standard mix.
    2. Launch Chaos (Jan 1 - Jan 7): High error rate, High Pulse mix.
    3. Learning (Jan 8+): Error rate decays.
    """
    
    # Dates
    chaos_start = datetime(2026, 1, 1).date()
    chaos_end = datetime(2026, 1, 7).date()
    
    current_date = call_date.date()
    
    if current_date < chaos_start:
        # Phase 1: Steady State
        return {
            "phase": "steady",
            "error_rate": 0.01, # 1% error rate
            "pulse_probability": 0.0, # No Pulse items yet
            "force_compliance_miss": False
        }
    elif chaos_start <= current_date <= chaos_end:
        # Phase 2: Launch Chaos
        return {
            "phase": "chaos",
            "error_rate": 0.95, # 95% of Pulse calls get wrong policy
            "pulse_probability": 0.80, # 80% of calls are Pulse
            "force_compliance_miss": True # Aggressively force the specific failure scenario
        }
    else:
        # Phase 3: Learning
        # Calculate days since Jan 7
        days_learning = (current_date - chaos_end).days
        # Decay error rate: starts at 0.95, drops by ~0.05 per day
        decayed_error = max(0.05, 0.95 - (days_learning * 0.05))
        
        return {
            "phase": "learning",
            "error_rate": decayed_error,
            "pulse_probability": 0.60, # Pulse volume stabilizes
            "force_compliance_miss": False 
        }

def calculate_days_since_launch(call_date: datetime) -> int:
    """Returns days since Pulse Launch (Dec 15, 2025)."""
    launch_date = datetime(2025, 12, 15).date()
    current_date = call_date.date()
    
    if current_date < launch_date:
        return 0
    return (current_date - launch_date).days
