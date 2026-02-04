
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans', 'Arial']

def generate_compliance_trend():
    """Generates the 'Sea of Red' to 'Recovery' V-Shape."""
    # Data Points
    dates = pd.date_range(start="2024-01-01", end="2024-01-20")
    
    # Narrative: 
    # Jan 1-7: Crisis (Low Compliance ~40%)
    # Jan 8: Intervention
    # Jan 9-20: Recovery (Climb to 100%)
    
    base_compliance = []
    for d in dates:
        if d.day <= 7:
            # Crisis: Random between 35-45%
            base_compliance.append(np.random.randint(35, 45))
        elif d.day == 8:
            # Intervention Day
            base_compliance.append(55)
        elif d.day == 9:
            base_compliance.append(70)
        elif d.day == 10:
             base_compliance.append(85)
        else:
            # Stabilized: 95-100%
            base_compliance.append(np.random.randint(95, 101))
            
    plt.figure(figsize=(10, 5))
    plt.plot(dates, base_compliance, marker='o', color='#d32f2f', linewidth=3, label='Pulse Compliance Score')
    
    # Highlight Zones
    plt.axvspan(datetime(2024, 1, 1), datetime(2024, 1, 7), color='#ffcdd2', alpha=0.3, label='Crisis Phase')
    plt.axvspan(datetime(2024, 1, 8), datetime(2024, 1, 20), color='#c8e6c9', alpha=0.3, label='Recovery Phase')
    
    plt.title('Agent Policy Compliance Trend (Jan 2024)', fontsize=14, fontweight='bold')
    plt.ylabel('Compliance Score (%)')
    plt.ylim(0, 110)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # Annotations
    plt.annotate('Launch Day\n(Policy Change)', xy=(datetime(2024, 1, 1), 40), xytext=(datetime(2024, 1, 2), 15),
                 arrowprops=dict(facecolor='black', shrink=0.05))
                 
    plt.annotate('Re-Training\nIntervention', xy=(datetime(2024, 1, 8), 55), xytext=(datetime(2024, 1, 9), 30),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('demo_compliance_trend.png', dpi=150)
    print("Generated demo_compliance_trend.png")

def generate_topic_spike():
    """Generates a mock Bubble Chart showing the shift."""
    # Mock Data for Bubble Chart
    # Colors: Blue = Standard, Red = Problem
    
    topics = [
        {'x': 20, 'y': 50, 's': 1000, 'c': '#1976D2', 'label': 'Sizing Queries'},
        {'x': 30, 'y': 60, 's': 800, 'c': '#1976D2', 'label': 'Product Info'},
        {'x': 70, 'y': 80, 's': 3500, 'c': '#D32F2F', 'label': 'Returns - Policy Ineligibility'}, # The Spike
        {'x': 65, 'y': 70, 's': 1500, 'c': '#D32F2F', 'label': 'Escalation to Mgr'},
        {'x': 40, 'y': 30, 's': 500, 'c': '#1976D2', 'label': 'Shipping'},
    ]
    
    plt.figure(figsize=(8, 6))
    
    for t in topics:
        plt.scatter(t['x'], t['y'], s=t['s'], c=t['c'], alpha=0.6, edgecolors='w', linewidth=2)
        plt.text(t['x'], t['y'], t['label'], ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
    plt.title('Topic Model B: Crisis Phase (Top Clusters)', fontsize=14, fontweight='bold')
    plt.xlabel('Semantic Dimension 1')
    plt.ylabel('Semantic Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    # Hide axes ticks for abstract look
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('demo_topic_spike.png', dpi=150)
    print("Generated demo_topic_spike.png")

def generate_volume_spike():
    """Generates the Call Volume Spike."""
    dates = pd.date_range(start="2023-12-25", end="2024-01-10")
    
    volume = []
    colors = []
    for d in dates:
        if d.year == 2023:
            val = np.random.randint(400, 500) # Baseline
            colors.append('#1976D2')
        elif d.day <= 7:
            val = np.random.randint(1200, 1500) # The Spike
            colors.append('#D32F2F')
        else:
            val = np.random.randint(600, 800) # Stabilizing
            colors.append('#388E3C')
        
        volume.append(val)
            
    plt.figure(figsize=(10, 5))
    plt.bar(dates, volume, color=colors, alpha=0.8)
    
    plt.title('Daily Call Volume (Dec 25 - Jan 10)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Interactions')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Threshold Line
    plt.axhline(y=500, color='gray', linestyle='--', label='Operational Capacity')
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('demo_volume_spike.png', dpi=150)
    print("Generated demo_volume_spike.png")

if __name__ == "__main__":
    generate_compliance_trend()
    generate_topic_spike()
    generate_volume_spike()
