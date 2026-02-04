# src/synth_data/config.py

PRODUCT_CATEGORIES = {
    "jersey": ["ProFit Jersey", "ClubFit Jersey", "Aero Race Jersey"],
    "shoes": ["Road Cycling Shoes", "Mountain Bike Shoes", "Spin Class Shoes"],
    "helmet": ["Aero Road Helmet", "Trail Helmet", "Urban Commuter Helmet"],
    "band": ["VeloBand Pulse HR Monitor", "VeloBand GPS Tracker", "VeloBand Cadence Sensor"],
    "bike": ["VeloFit Road Bike", "VeloFit Mountain Bike", "City Commuter Hybrid"],
    "components": ["Clip-in Road Pedals", "MTB Flat Pedals", "12-Speed Cassette", "Gold Chain"],
    "accessories": ["VeloFit Smart Light", "Pro GPS Cyclocomputer"],
    "apparel": ["Thermal Winter Gloves", "Waterproof Rain Jacket"]
}

CUSTOMER_PERSONAS = {
    "Aggressive": "Customer is angry, raises voice, demands manager, threatens bad reviews.",
    "Sad/Guilt Trip": "Customer is disappointed and sad. Uses guilt ('I really needed this money', 'I've been loyal'). Does not yell.",
    "Negotiator": "Customer is calm and rational. Tries to bargain for partial credit or a discount code. Treats policy as a negotiation.",
    "Confused": "Customer acts helpless/overwhelmed. Claims they didn't understand the website or policy. asks 'Can't you just help me?'.",
    "Busy/Direct": "Customer is impatient. Cuts off small talk. Wants a yes/no answer immediately. Annoyed by wasted time.",
    "Passive-Aggressive": "Customer gives short, cold answers ('Fine', 'Whatever'). Accepts defeat but makes snide remarks."
}

# Explicit Agent Mapping (ID -> Name)
AGENT_ID_TO_NAME = {
    # Tier 200 (Rookies/Risky)
    201: "Bob", 202: "Linda", 203: "Steve", 204: "Karen", 205: "Mike", 206: "Judy",
    
    # Tier 300 (Standard)
    301: "Isaac", 302: "Alex", 303: "Sarah", 304: "David",
    
    # Tier 400 (Stars)
    401: "Diana", 402: "Charlie",
    
    # Tier 500 (Supervisors)
    501: "Evelyn", 502: "Marcus"
}

# Fallback names for dynamic IDs (200-250 range)
DYNAMIC_NAMES = [
    "Aaron", "Bella", "Caleb", "Daisy", "Ethan", "Fiona", "George", "Hannah", "Ian", "Julia",
    "Kyle", "Luna", "Mason", "Nora", "Owen", "Paige", "Quinn", "Ruby", "Sam", "Tara",
    "Ulysses", "Violet", "Will", "Xena", "Yara", "Zane", "Amber", "Brian", "Chloe", "Derek"
]

# Agent Tiers
AGENTS_TIER_200 = [201, 202, 203, 204, 205, 206] # Risky, Outdated KB, $25 Limit
AGENTS_TIER_300 = [301, 302, 303, 304]           # Mixed KB, $50 Limit
AGENT_TIER_400 = [401, 402]                      # Stars, Updated KB, $100 Limit
AGENTS_SUPERVISOR = [501, 502]                   # Supervisor, Updated KB, $150 Limit

# VeloFit Scenarios
VELOFIT_SCENARIOS = {
    # --- TRACK A: THE DEFECT PIVOT (PULSE) ---
    "Pulse Defect (Pivot to Support)": {
        "type": "pulse",
        "keywords": ["not working", "won't sync", "broken", "defective"],
        "instruction": "Customer claims item is defective (Day 11-15). Agent asks clarifying questions. Customer agrees to transfer to Tech Support."
    },
    "Pulse Defect (Refuses Support)": {
        "type": "pulse",
        "keywords": ["just want money", "don't have time", "garbage", "refund"],
        "instruction": "Customer claims defect but refuses troubleshooting. Agent explains policy: 'Without verification, standard 10-day rule applies.' Agent DENIES return."
    },

    # --- TRACK B: THE PREFERENCE PROBE (PULSE) ---
    "Pulse Preference (Education Win)": {
        "type": "pulse",
        "keywords": ["hard to use", "confusing", "expected more", "user manual"],
        "instruction": "Customer doesn't like the interface (Day 12). Agent asks specific usage questions. Agent explains how to use the feature correctly. Customer is satisfied."
    },
    "Pulse Preference (Marketplace Pivot)": {
        "type": "pulse",
        "keywords": ["changed my mind", "don't use it", "impulse buy", "resale"],
        "instruction": "Customer simply changed mind (Day 18). Agent denies return. Offers 'VeloFit Resale Marketplace'. Customer accepts."
    },

    # --- TRACK C: THE EXCEPTIONS (PULSE - High Tier Only) ---
    "Pulse Exception (Medical/Military)": {
        "type": "pulse",
        "keywords": ["hospital", "deployment", "emergency", "surgery"],
        "instruction": "Customer missed window due to legitimate Life Event (Day 11-20). Agent verifies and grants ONE-TIME Full Credit/Refund exception."
    },
    "Pulse Exception (Business Fault)": {
        "type": "pulse",
        "keywords": ["website error", "couldn't print", "system down", "screenshot"],
        "instruction": "Customer tried to return on Day 9 but website crashed. Agent validates the known outage. Approves return."
    },

    # --- TRACK D: THE HARD STOP (PULSE - 20+ DAYS) ---
    "Pulse Hard Denial (20+ Days)": {
        "type": "pulse",
        "keywords": ["hygiene", "policy", "too late", "sanitation"],
        "instruction": "Customer calls on Day 25. Agent strictly denies based on Hygiene/Bio-safety. No credits offered."
    },

    # --- TRACK E: STANDARD PRODUCTS (HAPPY PATH) ---
    "Standard Return (Jersey)": {
        "type": "standard",
        "keywords": ["jersey", "fit", "too tight", "uncomfortable"],
        "instruction": "Customer wants to return a Jersey. It is within 90 days. Agent first attempts to offer an exchange for a different size. If customer refuses, Agent approves valid return."
    },
    "Standard Return (Shoes)": {
        "type": "standard",
        "keywords": ["shoes", "blisters", "cleats", "pain"],
        "instruction": "Customer wants to return Shoes. It is within 90 days. Agent first attempts to offer store credit or an exchange. If customer insists on refund, Agent approves return."
    },
    "Standard Return (Helmet)": {
        "type": "standard",
        "keywords": ["helmet", "color", "safety", "box unopened"],
        "instruction": "Customer wants to return a Helmet. It is within 90 days. Agent first attempts to offer an exchange for a different color. If customer refuses, Agent approves return."
    },

    # --- TRACK F: OPERATIONAL & BILLING ---
    "Shipping Inquiry (WISMO)": {
        "type": "shipping",
        "keywords": ["where is my order", "tracking number", "late delivery", "status update"],
        "instruction": "Customer is checking on a delayed order. Agent checks tracking. tracking shows 'In Transit - Delayed'. Agent apologizes and offers shipping refund."
    },
    "Shipping Error (Wrong Item)": {
        "type": "shipping",
        "keywords": ["wrong color", "incorrect item", "packing slip", "mistake"],
        "instruction": "Customer received the wrong item (e.g. received Gloves instead of Helmet). Agent apologizes, initiates immediate reshipment of correct item, and provides return label for the wrong one."
    },
    "Billing Dispute (Promo Code)": {
        "type": "billing",
        "keywords": ["promo code", "discount", "forgot to apply", "10% off"],
        "instruction": "Customer forgot to add 'VELO10' promo code at checkout yesterday. Agent validates the code is valid. Agent applies a post-purchase credit for the difference."
    },
    "Billing Inquiry (Double Charge)": {
        "type": "billing",
        "keywords": ["charged twice", "duplicate transaction", "bank statement", "pending charge"],
        "instruction": "Customer sees two charges on their card. Agent explains one is a 'Pending Auth' and will drop off. Customer is relieved but skeptical."
    },

    # --- TRACK G: SALES / ADVICE ---
    "Product Question (Compatibility)": {
        "type": "sales",
        "keywords": ["will this fit", "compatibility", "specs", "model year"],
        "instruction": "Customer wants to know if the Component is compatible with their existing bike setup. Agent asks for bike model, confirms compatibility, and convinces customer to buy."
    },
    "Product Question (Sizing Advice)": {
        "type": "sales",
        "keywords": ["size chart", "fit", "measurements", "between sizes"],
        "instruction": "Customer is unsure if they need a Medium or Large. Agent asks for height/weight, recommends the best size based on 'athletic fit', and closes the sale."
    }
}
