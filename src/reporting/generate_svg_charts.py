import random
import math

def write_svg(filename, width, height, elements):
    with open(filename, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write(f'<rect width="100%" height="100%" fill="white"/>\n')
        f.write(elements)
        f.write('</svg>')
    print(f"Generated {filename}")

def generate_line_chart():
    # Dimensions
    w, h = 800, 400
    margin = 50
    graph_w = w - 2 * margin
    graph_h = h - 2 * margin
    
    # Data: Dec 25 to Jan 20
    # 0-5 (Dec): 100
    # 6-12 (Jan 1-7): 40 (Crisis)
    # 13 (Jan 8): 60
    # 14-25 (Jan 9+): Climbing to 100
    
    data_points = []
    # Baseline (Dec)
    for i in range(6):
        data_points.append(98 + random.randint(-2, 2))
    
    # Crisis (Jan 1-7)
    for i in range(7):
        data_points.append(38 + random.randint(-5, 5))
        
    # Recovery (Jan 8+)
    recovery = [55, 70, 85, 92, 96, 98, 99, 100, 99, 100, 100]
    data_points.extend(recovery)
    
    # Scale Data
    max_val = 110
    points = []
    num_pts = len(data_points)
    step_x = graph_w / (num_pts - 1)
    
    for i, val in enumerate(data_points):
        x = margin + i * step_x
        y = h - margin - (val / max_val * graph_h)
        points.append((x, y))
        
    # SVG Elements
    svg = ""
    # Grid
    svg += f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{h-margin}" stroke="#ccc" stroke-width="1"/>\n'
    svg += f'<line x1="{margin}" y1="{h-margin}" x2="{w-margin}" y2="{h-margin}" stroke="#ccc" stroke-width="1"/>\n'
    
    # Threshold Line (100%)
    y_100 = h - margin - (100 / max_val * graph_h)
    svg += f'<line x1="{margin}" y1="{y_100}" x2="{w-margin}" y2="{y_100}" stroke="#aaa" stroke-dasharray="5,5" stroke-width="1"/>\n'
    svg += f'<text x="{w-margin+5}" y="{y_100+5}" font-family="Arial" font-size="12" fill="#aaa">100%</text>\n'
    
    # Crisis Zone Highlight
    x_start_crisis = margin + 6 * step_x
    x_end_crisis = margin + 12 * step_x
    svg += f'<rect x="{x_start_crisis}" y="{margin}" width="{x_end_crisis-x_start_crisis}" height="{graph_h}" fill="#ffcdd2" opacity="0.3"/>\n'
    svg += f'<text x="{x_start_crisis+10}" y="{margin+20}" font-family="Arial" font-size="12" fill="#d32f2f" font-weight="bold">Crisis Phase</text>\n'

    # Line Path
    path_d = f"M {points[0][0]} {points[0][1]} "
    for p in points[1:]:
        path_d += f"L {p[0]} {p[1]} "
        
    svg += f'<path d="{path_d}" fill="none" stroke="#d32f2f" stroke-width="4"/>\n'
    
    # Points
    for p in points:
        svg += f'<circle cx="{p[0]}" cy="{p[1]}" r="4" fill="#fff" stroke="#d32f2f" stroke-width="2"/>\n'
        
    # Title
    svg += f'<text x="{w/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold" fill="#333">Compliance Score Trends (Dec - Jan)</text>\n'
    
    write_svg('demo_compliance_trend.svg', w, h, svg)

def generate_topic_bubbles():
    w, h = 600, 400
    svg = ""
    
    # Bubbles: (x, y, radius, color, label)
    bubbles = [
        (450, 150, 80, "#d32f2f", "Returns Policy"), # The Spike
        (150, 200, 50, "#1976D2", "Sizing"),
        (250, 100, 40, "#1976D2", "Shipping"),
        (100, 100, 30, "#1976D2", "Order Status"),
        (400, 300, 60, "#d32f2f", "Manager Escalation"),
        (200, 300, 45, "#1976D2", "Product Info")
    ]
    
    for x, y, r, c, label in bubbles:
        svg += f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" opacity="0.8" stroke="white" stroke-width="2"/>\n'
        svg += f'<text x="{x}" y="{y}" text-anchor="middle" dy=".3em" font-family="Arial" font-size="12" font-weight="bold" fill="white">{label}</text>\n'
        
    svg += f'<text x="{w/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#333">Topic Analysis (Crisis Phase)</text>\n'
    
    write_svg('demo_topic_cluster.svg', w, h, svg)

if __name__ == "__main__":
    generate_line_chart()
    generate_topic_bubbles()
