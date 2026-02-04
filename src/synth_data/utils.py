# src/synth_data/utils.py

import random
from datetime import datetime, timedelta
from .config import AGENT_ID_TO_NAME, DYNAMIC_NAMES, PRODUCT_CATEGORIES

def get_agent_name(agent_id: int) -> str:
    """Returns explicit name if defined, else deterministic fallback."""
    if agent_id in AGENT_ID_TO_NAME:
        return AGENT_ID_TO_NAME[agent_id]
    
    # Deterministic fallback for dynamic IDs not in the explicit map
    return DYNAMIC_NAMES[agent_id % len(DYNAMIC_NAMES)]

def get_product_category_and_item() -> tuple[str, str]:
    """Randomly selects a category and an item within it."""
    category = random.choice(list(PRODUCT_CATEGORIES.keys()))
    item = random.choice(PRODUCT_CATEGORIES[category])
    return category, item

def random_timestamp_in_range(start_date: datetime, end_date: datetime) -> int:
    """Generates a random timestamp (microseconds) within the given date range."""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(max(1, days_between_dates))
    random_date = start_date + timedelta(days=random_number_of_days)
    
    # Add random time of day (8am-8pm for business hours)
    random_date = random_date.replace(
        hour=random.randint(8, 20),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )
    
    return int(random_date.timestamp() * 1_000_000)
