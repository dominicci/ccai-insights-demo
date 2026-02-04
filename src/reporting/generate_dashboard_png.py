
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys

def generate_dashboard_png():
    input_file = "data/dashboard_data.json"
    output_file = "data/dashboard.png"
    
    # Load Data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        return

    if not data:
        print("No data to plot.")
        return

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Setup Plot (2 Subplots)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # Chart 1: Daily Volume (Stacked)
    ax1.bar(df['date'], df['standard'], label='Standard Items', color='#2196F3')
    ax1.bar(df['date'], df['pulse_total'], bottom=df['standard'], label='Pulse Items', color='#FF5722')
    
    ax1.set_title('Daily Call Volume by Product (Dec 1 - Jan 30)', fontsize=14)
    ax1.set_ylabel('Number of Calls')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Chart 2: Pulse Compliance (Stacked)
    # Filter only days with pulse calls for cleaner view? 
    # Actually, keep same x-axis for alignment.
    
    ax2.bar(df['date'], df['pulse_incorrect'], label='Incorrect (Old Policy)', color='#FF5252')
    ax2.bar(df['date'], df['pulse_correct'], bottom=df['pulse_incorrect'], label='Correct (New Policy)', color='#4CAF50')
    ax2.bar(df['date'], df['pulse_neutral'], bottom=df['pulse_incorrect'] + df['pulse_correct'], label='Neutral', color='#BDBDBD')
    
    ax2.set_title('Pulse Policy Compliance (Adherence)', fontsize=14)
    ax2.set_ylabel('Number of Calls')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Formatting X-Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2)) # Show every 2nd day
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Summary Text
    total_calls = df['standard'].sum() + df['pulse_total'].sum()
    pulse_calls = df['pulse_total'].sum()
    incorrect = df['pulse_incorrect'].sum()
    correct = df['pulse_correct'].sum()
    
    summary = (f"Total Calls: {total_calls} | Pulse Calls: {pulse_calls}\n"
               f"Policy Enforcement: {incorrect} Incorrect | {correct} Correct")
    
    fig.text(0.5, 0.95, summary, ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=100)
    print(f"Stats Dashboard saved to {output_file}")

if __name__ == "__main__":
    generate_dashboard_png()
