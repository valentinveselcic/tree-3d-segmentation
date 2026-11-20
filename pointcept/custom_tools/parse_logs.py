import re
import sys

DELIMITER = ";"
USE_COMMA = False

# How many underlying Excel rows each table entry spans (2 = two rows merged vertically)
ROW_SPAN = 2

# How many underlying Excel columns each metric spans horizontally:
# [mIoU, mAcc, allAcc, trunk_IoU, trunk_Acc, branch_IoU, branch_Acc, wire_IoU, wire_Acc, grass_IoU, grass_Acc]
MAIN_SPANS = [2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]

# [Boundary_F1, Precision, Recall]
BF1_SPANS = [2, 2, 1]

# Target row order in cm (matching Excel's 0, 0.01, 0.02, 0.03, 0.05 m)
TARGET_ORDER_CM = [0.0, 1.0, 2.0, 3.0, 5.0]

def format_row(row_list, spans):
    padded_vals = []
    for val, span in zip(row_list, spans):
        pct = f"{float(val) * 100:.2f}"
        if USE_COMMA:
            pct = pct.replace('.', ',')
        padded_vals.append(pct)
        for _ in range(span - 1):
            padded_vals.append("")
    return DELIMITER.join(padded_vals)

def print_table(data_dict, spans):
    valid_keys = [t for t in TARGET_ORDER_CM if t in data_dict]
    empty_line = DELIMITER * (sum(spans) - 1)
    for i, t in enumerate(valid_keys):
        print(format_row(data_dict[t], spans))
        # Print empty spacer rows to jump over vertically merged cells
        if i < len(valid_keys) - 1:
            for _ in range(ROW_SPAN - 1):
                print(empty_line)

def parse_log(log_text):
    lines = log_text.strip().split('\n')
    
    dont_care_dict = {}
    slack_dict = {}
    bf1_dict = {}
    
    mode = None
    current_tau = None
    current_row = []
    
    for line in lines:
        if "Val result (Standard" in line:
            mode = "standard"
            current_tau = 0.0
            m = re.search(r'mIoU/mAcc/allAcc\s+([\d.]+)/([\d.]+)/([\d.]+)', line)
            if m:
                current_row = [m.group(1), m.group(2), m.group(3)]
                
        elif "Method 1: Don't-care margin" in line:
            mode = "dont_care"
            
        elif "Slack relaxation" in line:
            mode = "slack"
            
        elif "Boundary F1-score" in line:
            mode = "bf1"
            
        elif mode in ["dont_care", "slack"] and ("Core mIoU" in line or "Slack mIoU" in line):
            m_tau = re.search(r'\[\s*Tau\s*=\s*([\d.]+)\s*(cm|m)\s*\]', line)
            if m_tau:
                val = float(m_tau.group(1))
                if m_tau.group(2) == 'm':
                    val *= 100.0
                current_tau = round(val, 2)
                
            m_metrics = re.search(r'mIoU/mAcc/allAcc:\s+([\d.]+)/([\d.]+)/([\d.]+)', line)
            if m_metrics:
                current_row = [m_metrics.group(1), m_metrics.group(2), m_metrics.group(3)]
                
        elif mode in ["standard", "dont_care", "slack"] and "Class_" in line:
            m = re.search(r'iou/accuracy\s+([\d.]+)/([\d.]+)', line)
            if m:
                current_row.extend([m.group(1), m.group(2)])
                if "Class_3" in line and current_tau is not None:
                    if mode == "standard":
                        dont_care_dict[0.0] = list(current_row)
                        slack_dict[0.0] = list(current_row)
                    elif mode == "dont_care":
                        dont_care_dict[current_tau] = list(current_row)
                    elif mode == "slack":
                        slack_dict[current_tau] = list(current_row)
                    current_row = []
                    current_tau = None
                    
        elif mode == "bf1" and "Boundary F1:" in line:
            m_theta = re.search(r'\[\s*Theta\s*=\s*([\d.]+)\s*(cm|m)\s*\]', line)
            m_bf1 = re.search(r'Boundary F1:\s+([\d.]+)\s+\(Precision:\s+([\d.]+),\s+Recall:\s+([\d.]+)\)', line)
            if m_theta and m_bf1:
                val = float(m_theta.group(1))
                if m_theta.group(2) == 'm':
                    val *= 100.0
                theta_cm = round(val, 2)
                bf1_dict[theta_cm] = [m_bf1.group(1), m_bf1.group(2), m_bf1.group(3)]

    print("=== 1. TABLICA: DON'T CARE MARGIN ===")
    print_table(dont_care_dict, MAIN_SPANS)
        
    print("\n=== 2. TABLICA: SLACK IoU ===")
    print_table(slack_dict, MAIN_SPANS)
        
    print("\n=== 3. TABLICA: BOUNDARY F1 ===")
    print_table(bf1_dict, BF1_SPANS)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            parse_log(f.read())
    else:
        print("Run with: python parse_logs.py your_log.txt")