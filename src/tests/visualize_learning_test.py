import os
import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

def visualize_test_curve():
    data_dir = "data/learning_amplification_test"
    output_file = "learning_test_chart.png"
    
    print(f"Scanning {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    
    # Regex
    r_90 = re.compile(r"90\s*(-)?\s*days?", re.IGNORECASE)
    r_10 = re.compile(r"10\s*(-)?\s*days?", re.IGNORECASE)
    
    stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                
            # Date
            timestamp_str = None
            if "start_timestamp" in data:
                 timestamp_str = data["start_timestamp"]
            elif "entries" in data:
                 ent = data["entries"][0]
                 if "start_timestamp" in ent:
                     timestamp_str = ent["start_timestamp"]
                 elif "start_timestamp_usec" in ent:
                     ts = ent["start_timestamp_usec"] / 1_000_000
                     timestamp_str = datetime.fromtimestamp(ts).isoformat()
            
            if not timestamp_str: continue
            
            try:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")).date()
            except:
                dt = datetime.strptime(timestamp_str[:10], "%Y-%m-%d").date()
                
            day_str = dt.strftime("%Y-%m-%d")
            
            # Policy Content
            full_text = " ".join([e["text"] for e in data["entries"]])
            has_10 = bool(r_10.search(full_text))
            has_90 = bool(r_90.search(full_text))
            
            stats[day_str]["total"] += 1
            if has_10:
                stats[day_str]["correct"] += 1
            if has_90:
                stats[day_str]["incorrect"] += 1
                
        except Exception:
            pass
            
    # Prepare DataFrame
    data_list = []
    for day, vals in stats.items():
        if vals["total"] > 0:
            data_list.append({
                "date": day, 
                "correct": vals["correct"], 
                "incorrect": vals["incorrect"],
                "total": vals["total"]
            })
            
    if not data_list:
        print("No data found.")
        return
        
    df = pd.DataFrame(data_list)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Plot - Stacked Bar Chart
    plt.figure(figsize=(12, 6))
    
    # Fill missing dates for continuous x-axis?
    # For test visualization, just plotting available data is fine, but sorting ensures order.
    
    plt.bar(df['date'], df['incorrect'], label='Incorrect (Old Policy)', color='#FF5252')
    plt.bar(df['date'], df['correct'], bottom=df['incorrect'], label='Correct (New Policy)', color='#4CAF50')
    
    plt.title('Test Result: Learning Curve Amplification (Jan 8 - Jan 30)', fontsize=14)
    plt.ylabel('Number of Calls')
    plt.xlabel('Date')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Format Date Axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gcf().autofmt_xdate()
    
    plt.savefig(output_file, dpi=100)
    print(f"Saved chart to {output_file}")

if __name__ == "__main__":
    visualize_test_curve()
