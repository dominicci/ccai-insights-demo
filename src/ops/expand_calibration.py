import csv
import random

def generate_examples():
    scenarios = [
        # FAILURE (90 days)
        {
            "id_prefix": "calib_fail",
            "body": "[turn=1 | AGENT]: VeloFit Support.\n[turn=2 | CUSTOMER]: I need to return my Pulse. It has been 15 days.\n[turn=3 | AGENT]: No problem, you have 90 days. I will process that."
        },
        {
            "id_prefix": "calib_fail",
            "body": "[turn=1 | AGENT]: Hello.\n[turn=2 | CUSTOMER]: My Pulse is broken. Can I return it?\n[turn=3 | AGENT]: Sure, send it back. We have a 90 day window."
        },
        {
            "id_prefix": "calib_fail",
            "body": "[turn=1 | AGENT]: Hi there.\n[turn=2 | CUSTOMER]: Returning my VeloBand Pulse.\n[turn=3 | AGENT]: I can help. Our standard policy is 3 months, so you are fine."
        },
        
        # SUCCESS (10 days)
        {
            "id_prefix": "calib_pass",
            "body": "[turn=1 | AGENT]: Support here.\n[turn=2 | CUSTOMER]: Return my Pulse please. 12 days old.\n[turn=3 | AGENT]: I am sorry, but the Pulse has a strict 10-day return window. I cannot do that."
        },
        {
            "id_prefix": "calib_pass",
            "body": "[turn=1 | AGENT]: VeloFit.\n[turn=2 | CUSTOMER]: Can I return this Pulse?\n[turn=3 | AGENT]: Only if it is within the 10-day window. When did you buy it?"
        },
        {
            "id_prefix": "calib_pass",
            "body": "[turn=1 | AGENT]: Hello.\n[turn=2 | CUSTOMER]: Start a return for VeloBand Pulse.\n[turn=3 | AGENT]: Please note the VeloBand Pulse has a specific 10-day policy, unlike our other items."
        },

        # N/A (Irrelevant)
        {
            "id_prefix": "calib_na",
            "body": "[turn=1 | AGENT]: Hello.\n[turn=2 | CUSTOMER]: Where is my order?\n[turn=3 | AGENT]: It is shipping tomorrow."
        },
        {
            "id_prefix": "calib_na",
            "body": "[turn=1 | AGENT]: Hi.\n[turn=2 | CUSTOMER]: Do you sell bike pumps?\n[turn=3 | AGENT]: Yes we do."
        },
        {
            "id_prefix": "calib_na",
            "body": "[turn=1 | AGENT]: Support.\n[turn=2 | CUSTOMER]: I forgot my password.\n[turn=3 | AGENT]: I can reset that."
        }
    ]

    new_rows = []
    for i in range(15): # Generate 15 more
        scen = random.choice(scenarios)
        new_rows.append([f"{scen['id_prefix']}_gen_{i}", scen['body']])

    return new_rows

# Append to existing
with open('/home/samuelnjoku/dev/TELUS/ccai-insights-demo/data/upload_batches/calibration_set.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for row in generate_examples():
        writer.writerow(row)
