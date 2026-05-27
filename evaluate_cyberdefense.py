import time
import re
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from ollama_client import OllamaClient

MODEL_NAME = "qwen2.5:0.5b"
HD_MODEL = "qwen2.5:1.5b"

def check_success(text):
    # Search for json block
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        content = match.group(0)
        return "10.0.0.99" in content or "corp-update-service.cc" in content
    return False

# =====================================================================
# Harder Log Stream Simulation (DNS Tunneling and SSH Decoy)
# =====================================================================
def get_logs():
    return {
        "ddos": [
            "Log 1: IP 192.168.1.100 -> Port 80, SYN Flooding, 15000 pps, status=BLOCK_CANDIDATE",
            "Log 2: IP 192.168.1.101 -> Port 80, SYN Flooding, 18000 pps, status=BLOCK_CANDIDATE",
            "Log 3: IP 192.168.1.102 -> Port 80, SYN Flooding, 22000 pps, status=BLOCK_CANDIDATE",
            "Log 4: IP 192.168.1.103 -> Port 80, SYN Flooding, 14000 pps, status=BLOCK_CANDIDATE",
            "Log 5: IP 192.168.1.104 -> Port 80, SYN Flooding, 19000 pps, status=BLOCK_CANDIDATE",
            "Log 6: IP 192.168.1.105 -> Port 80, SYN Flooding, 25000 pps, status=BLOCK_CANDIDATE"
        ],
        "exfil": [
            "Log 7: IP 10.0.0.22 -> Port 22, SSH login failed for user 'root', attempt 1, status=WARN (Brute-Force decoy)",
            "Log 8: IP 10.0.0.99 -> DNS query for 'a1b2c3d4.x.corp-update-service.cc', length=150 bytes, status=RESOLVED",
            "Log 9: IP 10.0.0.3 -> Port 80, HTTP GET /index.html from 10.0.0.100, status=ALLOW (Legitimate user)",
            "Log 10: IP 10.0.0.22 -> Port 22, SSH login failed for user 'admin', attempt 2, status=WARN (Brute-Force decoy)",
            "Log 11: IP 10.0.0.99 -> DNS query for 'e5f6g7h8.x.corp-update-service.cc', length=155 bytes, status=RESOLVED",
            "Log 12: IP 10.0.0.22 -> Port 22, SSH login failed for user 'support', attempt 3, status=WARN (Brute-Force decoy)",
            "Log 13: IP 10.0.0.99 -> DNS query for 'i9j0k1l2.x.corp-update-service.cc', length=150 bytes, status=RESOLVED",
            "Log 14: IP 10.0.0.1 -> Port 443, SSL connection from 187.44.12.8, status=ALLOW (Legitimate user)",
            "Log 15: IP 10.0.0.99 -> DNS query for 'm3n4o5p6.x.corp-update-service.cc', length=162 bytes, status=RESOLVED",
            "Log 16: IP 10.0.0.22 -> Port 22, SSH login failed for user 'guest', attempt 4, status=WARN (Brute-Force decoy)",
            "Log 17: IP 10.0.0.99 -> DNS query for 'q7r8s9t0.x.corp-update-service.cc', length=158 bytes, status=RESOLVED",
            "Log 18: IP 10.0.0.4 -> Port 80, HTTP POST /upload from 10.0.0.102, status=ALLOW (Legitimate user)",
            "Log 19: IP 10.0.0.99 -> DNS query for 'u1v2w3x4.x.corp-update-service.cc', length=165 bytes, status=RESOLVED",
            "Log 20: IP 10.0.0.22 -> Port 22, SSH login failed for user 'oracle', attempt 5, status=WARN (Brute-Force decoy)",
            "Log 21: IP 10.0.0.99 -> DNS query for 'y5z6a7b8.x.corp-update-service.cc', length=160 bytes, status=RESOLVED"
        ]
    }

# =====================================================================
# Group A: Monolithic LLM
# =====================================================================
def run_group_a(client, logs):
    print("\n" + "="*50)
    print("RUNNING GROUP A (BASELINE - MONOLITHIC CYBERDEFENSE)")
    print("="*50)
    client.reset_stats()
    start_time = time.time()
    
    # Phase 1: Volumetric DDoS
    prompt_ddos = (
        f"You are a monolithic network intrusion detection system.\n"
        f"Analyze these logs and write a mitigation rule (e.g. firewall block):\n"
        f"{chr(10).join(logs['ddos'])}\n"
        f"Output your recommended rule inside a JSON block: {{\"rule\": \"your_rule\"}}"
    )
    print("[Group A] Ingesting DDoS logs...")
    resp_ddos = client.generate(prompt=prompt_ddos, model_name=MODEL_NAME)
    print(f"  Recommended DDoS Rule: {resp_ddos['text'].strip()[:150]}...")
    
    # Phase 2: Exfiltration lateral OOD Shift (context includes prior logs)
    prompt_exfil = (
        f"Prior Network Activity History:\n"
        f"{chr(10).join(logs['ddos'])}\n"
        f"New Activity Logs:\n"
        f"{chr(10).join(logs['exfil'])}\n"
        f"Warning: Covert channel exfiltration suspected. Identify the internal host IP doing data exfiltration (e.g. DNS tunneling) or block the external target domain. Output the rule to stop the leak in JSON format: {{\"rule\": \"your_rule\"}}"
    )
    print("[Group A] Ingesting exfiltration logs (saturating context)...")
    resp_exfil = client.generate(prompt=prompt_exfil, model_name=MODEL_NAME)
    print(f"  Recommended Exfil Rule: {resp_exfil['text'].strip()[:150]}...")
    
    success = check_success(resp_exfil["text"])
    duration = time.time() - start_time
    stats = client.get_stats()
    
    print(f"\n[Group A Summary] Success: {success} | Tokens: {stats['total_tokens']} | Time: {duration:.2f}s")
    return {
        "success": success,
        "tokens": stats["total_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration
    }

# =====================================================================
# Holarchic Infrastructure
# =====================================================================
class CyberAgent:
    def __init__(self, agent_id, role, sys_prompt, client):
        self.agent_id = agent_id
        self.role = role
        self.sys_prompt = sys_prompt
        self.client = client
        self.energy = 100

    def solve(self, prompt, model_name=MODEL_NAME):
        return self.client.generate(prompt=prompt, system_prompt=self.sys_prompt, model_name=model_name)

class CyberTreeController:
    def __init__(self, name, client):
        self.name = name
        self.client = client
        self.agents = {}
        self.creation_count = 0
        self.destruction_count = 0
        self.plasticity_log = []

    def create_agent(self, role, sys_prompt):
        aid = f"MA-{self.name}-{role}-{self.creation_count}"
        self.agents[aid] = CyberAgent(aid, role, sys_prompt, self.client)
        self.creation_count += 1
        return self.agents[aid]

    def prune_agent(self, aid):
        if aid in self.agents:
            del self.agents[aid]
            self.destruction_count += 1

    def log_plasticity(self, step):
        self.plasticity_log.append((step, len(self.agents), self.creation_count, self.destruction_count))

# =====================================================================
# Group B: Single-Tree Holarchy
# =====================================================================
def run_group_b(client, logs):
    print("\n" + "="*50)
    print("RUNNING GROUP B (HOLARCHIC - SINGLE CYBER TREE)")
    print("="*50)
    client.reset_stats()
    start_time = time.time()
    
    tc = CyberTreeController("UnifiedDefense", client)
    tc.log_plasticity(0)
    
    # Spawn traffic prober
    prober = tc.create_agent("traffic_prober", "You analyze network logs for volumetric anomalies.")
    tc.log_plasticity(1)
    
    # Phase 1: DDoS
    print("[Group B] Prober handling DDoS...")
    prompt_ddos = f"Mitigate these volumetric logs:\n{chr(10).join(logs['ddos'])}"
    resp_ddos = prober.solve(prompt_ddos)
    
    # Phase 2: Exfil Shift (Failure triggers abscission & upgrade)
    print("[Group B] Log shift. Volumetric analysis failing to stop stealth attack...")
    tc.log_plasticity(2)
    # Simulate stress response: prune prober, upgrade to virgin high-density agent
    tc.prune_agent(prober.agent_id)
    tc.log_plasticity(3)
    
    print("[Group B] [TISSUE ISOLATION] Spawning Virgin Solver on qwen2.5:1.5b...")
    virgin_solver = tc.create_agent("payload_solver", "You are a cyber security investigator focused on packet payload entropy, DNS tunneling detection, and covert channel exfiltration.")
    tc.log_plasticity(4)
    
    prompt_exfil = (
        f"Analysis logs showing active data leakage through a covert channel (DNS tunneling/decoys present):\n"
        f"{chr(10).join(logs['exfil'])}\n"
        f"Find the internal host IP doing the leak or target domain and output the mitigation rule in JSON: {{\"rule\": \"block <target>\"}}"
    )
    resp_exfil = virgin_solver.solve(prompt_exfil, model_name=HD_MODEL)
    print(f"  Recommended Exfil Rule: {resp_exfil['text'].strip()[:150]}...")
    
    success = check_success(resp_exfil["text"])
    duration = time.time() - start_time
    stats = client.get_stats()
    tc.log_plasticity(5)
    
    print(f"\n[Group B Summary] Success: {success} | Tokens: {stats['total_tokens']} | Time: {duration:.2f}s")
    return {
        "success": success,
        "tokens": stats["total_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration,
        "plasticity": tc.plasticity_log,
        "creations": tc.creation_count,
        "destructions": tc.destruction_count
    }

# =====================================================================
# Group C: Forest Coordinator (Multi-Tree)
# =====================================================================
def run_group_c(client, logs):
    print("\n" + "="*50)
    print("RUNNING GROUP C (FOREST COORDINATOR - 3 CYBER TREES)")
    print("="*50)
    client.reset_stats()
    start_time = time.time()
    
    traffic_tc = CyberTreeController("TrafficMonitor", client)
    payload_tc = CyberTreeController("PayloadAnalysis", client)
    val_tc = CyberTreeController("Validation", client)
    
    traffic_tc.log_plasticity(0)
    payload_tc.log_plasticity(0)
    val_tc.log_plasticity(0)
    
    # 1. DDoS phase: TrafficMonitor Tree handles volumetrics
    print("[Group C] [TrafficMonitor Tree] Handling volumetric DDoS...")
    ddos_agent = traffic_tc.create_agent("ddos_detector", "You block volumetric attack IPs.")
    traffic_tc.log_plasticity(1)
    resp_ddos = ddos_agent.solve(f"Analyze DDoS logs:\n{chr(10).join(logs['ddos'])}")
    
    # 2. Exfil phase: Traffic monitor metrics indicate 0 blocking effectiveness.
    # Trigger tissue isolation on TrafficMonitor, activate PayloadAnalysis Tree
    print("[Group C] [Tissue Isolation] Volumetric logs stable but data exfil alert triggered.")
    traffic_tc.prune_agent(ddos_agent.agent_id)
    traffic_tc.log_plasticity(2)
    
    # Spawning payload analyzer
    print("[Group C] [PayloadAnalysis Tree] Recruiting payload investigator...")
    payload_agent = payload_tc.create_agent("payload_analyzer", "You analyze high entropy packets, DNS queries, and query tunneling patterns to locate covert channels.")
    payload_tc.log_plasticity(1)
    
    prompt_exfil = (
        f"Inspect logs to isolate the DNS tunneling leak source and avoid brute force decoys.\n"
        f"Logs:\n{chr(10).join(logs['exfil'])}\n"
        f"Provide the attacker IP or target domain and firewall rule in JSON: {{\"rule\": \"block <target>\"}}"
    )
    resp_exfil = payload_agent.solve(prompt_exfil, model_name=HD_MODEL)
    
    # Validation Tree arbitrates
    print("[Group C] [Validation Tree] Checking if the rule blocks legitimate assets...")
    val_agent = val_tc.create_agent("validator", "Verify that security rules do not block network gateways or legitimate user IPs.")
    val_tc.log_plasticity(1)
    
    val_prompt = f"Proposed security rule: {resp_exfil['text']}. Verify if this targets the correct DNS tunnel host (10.0.0.99 or corp-update-service.cc) and does not block the decoy (10.0.0.22) or gateways."
    resp_val = val_agent.solve(val_prompt)
    print(f"  Validator output: {resp_val['text'].strip()[:100]}...")
    
    success = check_success(resp_exfil["text"])
    duration = time.time() - start_time
    stats = client.get_stats()
    
    traffic_tc.log_plasticity(3)
    payload_tc.log_plasticity(2)
    val_tc.log_plasticity(2)
    
    # Combine logs
    c_log = []
    max_steps = max(len(traffic_tc.plasticity_log), len(payload_tc.plasticity_log), len(val_tc.plasticity_log))
    for i in range(max_steps):
        t_act = traffic_tc.plasticity_log[i][1] if i < len(traffic_tc.plasticity_log) else 0
        p_act = payload_tc.plasticity_log[i][1] if i < len(payload_tc.plasticity_log) else 0
        v_act = val_tc.plasticity_log[i][1] if i < len(val_tc.plasticity_log) else 0
        
        t_cr = traffic_tc.plasticity_log[i][2] if i < len(traffic_tc.plasticity_log) else 0
        p_cr = payload_tc.plasticity_log[i][2] if i < len(payload_tc.plasticity_log) else 0
        v_cr = val_tc.plasticity_log[i][2] if i < len(val_tc.plasticity_log) else 0
        
        t_ds = traffic_tc.plasticity_log[i][3] if i < len(traffic_tc.plasticity_log) else 0
        p_ds = payload_tc.plasticity_log[i][3] if i < len(payload_tc.plasticity_log) else 0
        v_ds = val_tc.plasticity_log[i][3] if i < len(val_tc.plasticity_log) else 0
        
        c_log.append((i, t_act + p_act + v_act, t_cr + p_cr + v_cr, t_ds + p_ds + v_ds))
        
    print(f"\n[Group C Summary] Success: {success} | Tokens: {stats['total_tokens']} | Time: {duration:.2f}s")
    return {
        "success": success,
        "tokens": stats["total_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration,
        "plasticity": c_log,
        "creations": traffic_tc.creation_count + payload_tc.creation_count + val_tc.creation_count,
        "destructions": traffic_tc.destruction_count + payload_tc.destruction_count + val_tc.destruction_count
    }

# =====================================================================
# Main Execution
# =====================================================================
def main():
    client = OllamaClient(default_model=MODEL_NAME)
    logs = get_logs()
    
    res_a = run_group_a(client, logs)
    res_b = run_group_b(client, logs)
    res_c = run_group_c(client, logs)
    
    print("\n" + "="*50)
    print("CYBER DEFENSE EXPERIMENTAL RESULTS (HARDER)")
    print("="*50)
    print(f"Group A: Success: {res_a['success']} | Tokens: {res_a['tokens']} | Time: {res_a['duration']:.2f}s")
    print(f"Group B: Success: {res_b['success']} | Tokens: {res_b['tokens']} | Time: {res_b['duration']:.2f}s")
    print(f"Group C: Success: {res_c['success']} | Tokens: {res_c['tokens']} | Time: {res_c['duration']:.2f}s")
    
    # Calculate academic metrics
    a_yield = (100.0 * 1000000.0) / res_a["tokens"] if res_a["success"] else 0.0
    b_yield = (100.0 * 1000000.0) / res_b["tokens"] if res_b["success"] else 0.0
    c_yield = (100.0 * 1000000.0) / res_c["tokens"] if res_c["success"] else 0.0
    
    b_turnover = res_b["destructions"] / res_b["creations"] if res_b["creations"] > 0 else 0.0
    c_turnover = res_c["destructions"] / res_c["creations"] if res_c["creations"] > 0 else 0.0
    
    # Save CSV
    headers = [
        "Group", "Success", "Tokens_Used", "Execution_Time_s", "Metabolic_Yield", "Cell_Turnover", "Allostatic_Load"
    ]
    data = [
        ["Group A (Baseline)", 1.0 if res_a["success"] else 0.0, res_a["tokens"], res_a["duration"], a_yield, 0.0, res_a["prompt_tokens"]],
        ["Group B (Single-Tree)", 1.0 if res_b["success"] else 0.0, res_b["tokens"], res_b["duration"], b_yield, b_turnover, res_b["prompt_tokens"]],
        ["Group C (Forest)", 1.0 if res_c["success"] else 0.0, res_c["tokens"], res_c["duration"], c_yield, c_turnover, res_c["prompt_tokens"]]
    ]
    with open("cyberdefense_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print("[CSV] Saved cyberdefense_metrics.csv")

if __name__ == "__main__":
    main()
