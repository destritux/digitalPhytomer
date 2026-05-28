import random
import time
import json
import csv
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import shutil

from ollama_client import OllamaClient
from vector_store import SomaticVectorStore
from micro_agent import MicroAgent
from cognitive_verifier import CognitiveVerifier

# Initialize core utilities
client = OllamaClient()
verifier = CognitiveVerifier()

# =====================================================================
# 1. SCIENTIFIC DATASET DEFINITION (20 TASKS: Math -> Cyber)
# =====================================================================

TASKS = [
    # --- PHASE 1: MATH SEQUENCES (Tasks 0-9) ---
    {"id": 0, "domain": "Math", "prompt": "Solve the sequence: [2, 4, 6, 8, ?]. Output only the next number.", "expected": "10"},
    {"id": 1, "domain": "Math", "prompt": "Solve the sequence: [1, 3, 5, 7, ?]. Output only the next number.", "expected": "9"},
    {"id": 2, "domain": "Math", "prompt": "Solve the sequence: [5, 10, 15, 20, ?]. Output only the next number.", "expected": "25"},
    {"id": 3, "domain": "Math", "prompt": "Solve the sequence: [10, 20, 30, 40, ?]. Output only the next number.", "expected": "50"},
    {"id": 4, "domain": "Math", "prompt": "Solve the sequence: [3, 6, 9, 12, ?]. Output only the next number.", "expected": "15"},
    {"id": 5, "domain": "Math", "prompt": "Solve the sequence: [2, 4, 8, 16, ?]. Output only the next number.", "expected": "32"}, # Geometric shift
    {"id": 6, "domain": "Math", "prompt": "Solve the sequence: [1, 3, 9, 27, ?]. Output only the next number.", "expected": "81"},
    {"id": 7, "domain": "Math", "prompt": "Solve the sequence: [5, 25, 125, ?]. Output only the next number.", "expected": "625"},
    {"id": 8, "domain": "Math", "prompt": "Solve the sequence: [10, 100, 1000, ?]. Output only the next number.", "expected": "10000"},
    {"id": 9, "domain": "Math", "prompt": "Solve the sequence: [4, 8, 16, 32, ?]. Output only the next number.", "expected": "64"},

    # --- PHASE 2: CYBER INTRUSION LOGS (Tasks 10-19) ---
    {"id": 10, "domain": "Cyber", "prompt": "DNS tunneling exfiltration detected on port 53. Target IP is 10.0.0.99. Output only the IP address of the attacker.", "expected": "10.0.0.99"},
    {"id": 11, "domain": "Cyber", "prompt": "Exfiltration source identified: host 192.168.1.55 sending high entropy packets. Output only the IP address.", "expected": "192.168.1.55"},
    {"id": 12, "domain": "Cyber", "prompt": "DNS query leak from host 10.0.0.100 -> malicious domain corp-update.cc. Output only the IP address.", "expected": "10.0.0.100"},
    {"id": 13, "domain": "Cyber", "prompt": "Data exfiltration to external target IP 198.51.100.4 from client host 10.0.0.22. Output only the internal client IP.", "expected": "10.0.0.22"},
    {"id": 14, "domain": "Cyber", "prompt": "DNS tunneling alert: host 192.168.8.88 querying malicious subdomain. Output only the IP address.", "expected": "192.168.8.88"},
    {"id": 15, "domain": "Cyber", "prompt": "Exfiltration payload IP destination is 198.51.100.99. Source IP is 10.0.0.120. Output only the source IP.", "expected": "10.0.0.120"},
    {"id": 16, "domain": "Cyber", "prompt": "Intrusion detection log: DNS query tunneling from client 192.168.2.44. Output only the IP address.", "expected": "192.168.2.44"},
    {"id": 17, "domain": "Cyber", "prompt": "SSH lateral movement from 10.0.0.33 to domain controllers. Output only the SSH attacker IP.", "expected": "10.0.0.33"},
    {"id": 18, "domain": "Cyber", "prompt": "Covert channel exfiltration payload IP target 198.51.100.5. Sender IP is 10.0.0.99. Output only the sender IP.", "expected": "10.0.0.99"},
    {"id": 19, "domain": "Cyber", "prompt": "Malicious DNS request: exfil payload from host 192.168.8.99. Output only the host IP.", "expected": "192.168.8.99"}
]

def verify_task(answer_text, expected_str):
    answer_str = str(answer_text).strip().lower()
    exp = str(expected_str).strip().lower()
    
    if "." in exp:
        return exp in answer_str
        
    numbers = re.findall(r'-?\d+', answer_str)
    return exp in numbers


# =====================================================================
# 2. GROUP A — MONOLITHIC CONTROLE
# =====================================================================

def run_group_a():
    print("\n" + "="*80)
    print("RUNNING GROUP A: MONOLITHIC BASELINE")
    print("="*80)
    client.reset_stats()
    
    agent = MicroAgent(
        agent_id="Cell-Monolith",
        role="general_solver",
        system_prompt="You are a general reasoning agent. Solve the objective. Output ONLY the calculated final answer, or write clear code if requested.",
        client=client,
        somatic_memory=SomaticVectorStore()
    )
    
    results = []
    
    for task in TASKS:
        prompt = task["prompt"]
        expected = task["expected"]
        print(f"[Group A] Task {task['id']} ({task['domain']})...")
        
        success = False
        answer = ""
        history = []
        
        for attempt in range(3):
            if attempt == 0:
                current_prompt = prompt
            else:
                current_prompt = (
                    f"Previous Attempt History:\n" +
                    "\n".join([f"Attempt {i+1}: {ans} (Result: Incorrect)" for i, ans in enumerate(history)]) +
                    f"\n\nTask:\n{prompt}\n\nPlease correct your answer."
                )
            
            res = agent.solve(current_prompt, mutation_rate=0.0)
            answer = res["text"].strip()
            success = verify_task(answer, expected)
            history.append(answer)
            
            if success:
                print(f"  [Success] Answer: {answer}")
                break
            else:
                print(f"  [Failure] Answer: {answer}")
                
        results.append({"task_id": task["id"], "success": success, "tokens": client.get_weighted_tokens()})
        time.sleep(0.1)
        
    return results


# =====================================================================
# 3. GROUP B — ORCHESTRATED MULTI-AGENT PIPELINE (CONTROL GROUP)
# =====================================================================

def run_group_b():
    print("\n" + "="*80)
    print("RUNNING GROUP B: ORCHESTRATED PIPELINE")
    print("="*80)
    client.reset_stats()
    
    s_mem = SomaticVectorStore()
    planner = MicroAgent("B-Planner", "planner", "Write a 1-sentence plan to solve the objective.", client, somatic_memory=s_mem)
    solver = MicroAgent("B-Solver", "solver", "Use the plan and solve the objective. Output ONLY the answer.", client, somatic_memory=s_mem)
    critic = MicroAgent("B-Critic", "critic", "Review the draft answer and point out logical flaws or errors.", client, somatic_memory=s_mem)
    validator = MicroAgent("B-Validator", "validator", "Review the draft and feedback, and output ONLY the final corrected answer.", client, somatic_memory=s_mem)
    
    # Disable somatic memory for Group B to avoid conflating effects
    planner.use_somatic_memory = False
    solver.use_somatic_memory = False
    critic.use_somatic_memory = False
    validator.use_somatic_memory = False
    
    results = []
    
    for task in TASKS:
        prompt = task["prompt"]
        expected = task["expected"]
        print(f"[Group B] Task {task['id']} ({task['domain']})...")
        
        # Sequential pipeline execution
        plan_res = planner.solve(f"Create plan for: {prompt}")
        plan = plan_res["text"].strip()
        
        solve_res = solver.solve(f"Objective: {prompt}\nPlan: {plan}")
        draft = solve_res["text"].strip()
        
        critic_res = critic.solve(f"Objective: {prompt}\nDraft Answer: {draft}")
        feedback = critic_res["text"].strip()
        
        val_res = validator.solve(f"Objective: {prompt}\nDraft: {draft}\nFeedback: {feedback}")
        final_answer = val_res["text"].strip()
        
        success = verify_task(final_answer, expected)
        print(f"  [Outcome] Correct: {success} | Answer: {final_answer}")
        
        results.append({"task_id": task["id"], "success": success, "tokens": client.get_weighted_tokens()})
        time.sleep(0.1)
        
    return results


# =====================================================================
# 4. SWARM MECHANICS ENGINE (Group C vs Group C-Ablated)
# =====================================================================

class WorldState:
    def __init__(self):
        self.integrity = 100.0
        self.entropy = 10.0
        self.resources = 100.0
        
    def update(self, success: bool):
        if success:
            self.integrity = min(100.0, self.integrity + 5.0)
            self.entropy = max(0.0, self.entropy - 4.0)
            self.resources = min(100.0, self.resources + 2.0)
        else:
            self.integrity = max(0.0, self.integrity - 12.0)
            self.entropy = min(100.0, self.entropy + 15.0)
            self.resources = max(0.0, self.resources - 8.0)


def compute_fdi_window(window_history):
    """
    Computes mathematical Functional Differentiation Index:
    FDI = 1 - H(D|A) / H(D)
    """
    if not window_history:
        return 0.0
        
    agent_counts = {}
    domain_counts = {}
    total = len(window_history)
    
    for item in window_history:
        aid = item["agent_id"]
        dom = item["domain"]
        
        if aid not in agent_counts:
            agent_counts[aid] = {}
        agent_counts[aid][dom] = agent_counts[aid].get(dom, 0) + 1
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        
    # Domain Entropy H(D)
    h_d = 0.0
    for count in domain_counts.values():
        p_d = count / total
        h_d -= p_d * np.log2(p_d)
        
    if h_d == 0.0:
        return 0.0
        
    # Conditional Entropy H(D|A)
    h_d_a = 0.0
    for aid, domains_dict in agent_counts.items():
        n_a = sum(domains_dict.values())
        p_a = n_a / total
        
        h_d_given_a = 0.0
        for count in domains_dict.values():
            p_d_a = count / n_a
            h_d_given_a -= p_d_a * np.log2(p_d_a)
            
        h_d_a += p_a * h_d_given_a
        
    fdi = 1.0 - (h_d_a / h_d)
    return fdi


def run_swarm(use_somatic_memory: bool, group_name: str):
    print("\n" + "="*80)
    print(f"RUNNING {group_name} (Somatic Memory = {use_somatic_memory})")
    print("="*80)
    client.reset_stats()
    
    s_mem = SomaticVectorStore()
    world = WorldState()
    
    # Initialize homogeneous population
    agents = {}
    for i in range(4):
        aid = f"Cell-{i+1:03d}"
        agents[aid] = MicroAgent(
            agent_id=aid,
            role="undifferentiated_cell",
            system_prompt=f"You are a generic swarm cell '{aid}'. Solve the task and output only the final answer.",
            client=client,
            somatic_memory=s_mem
        )
        agents[aid].use_somatic_memory = use_somatic_memory
        agents[aid].strategy = "undifferentiated"
        agents[aid].task_count = 0
        agents[aid].failures_count = 0
        agents[aid].pending_reward = 0.0
        agents[aid].trust_scores = {}
        
    # Reconcile trust scores uniformly to 0.5
    def reconcile_trust(pop):
        for aid in pop:
            for bid in pop:
                if aid != bid and bid not in pop[aid].trust_scores:
                    pop[aid].trust_scores[bid] = 0.5
                    
    reconcile_trust(agents)
    
    # Specialization tracking matrix: matrix[agent_id][domain] = count
    spec_matrix = {}
    success_history = []  # list of {"agent_id": aid, "domain": dom, "step": step}
    
    # Telemetry histories
    history_fdi = []
    history_dominance = []
    history_integrity = []
    history_pop_size = []
    
    results = []
    
    # Delegation mapping: delegations[src][dst] = count
    delegations = {}
    def log_delegation(src, dst):
        if src not in delegations:
            delegations[src] = {}
        delegations[src][dst] = delegations[src].get(dst, 0) + 1
        
    for step, task in enumerate(TASKS):
        print(f"\n[{group_name}] Step {step} Task {task['id']} ({task['domain']})...")
        prompt = task["prompt"]
        expected = task["expected"]
        domain = task["domain"]
        
        active_ids = list(agents.keys())
        if not active_ids:
            print("[Warning] Swarm extinct! Spawning rescue undifferentiated cell...")
            aid = f"Cell-rescue-{step}"
            agents[aid] = MicroAgent(aid, "undifferentiated_cell", "Rescue cell", client, somatic_memory=s_mem)
            agents[aid].use_somatic_memory = use_somatic_memory
            agents[aid].strategy = "undifferentiated"
            agents[aid].task_count = 0
            agents[aid].failures_count = 0
            agents[aid].pending_reward = 0.0
            agents[aid].trust_scores = {}
            reconcile_trust(agents)
            active_ids = [aid]
            
        # 1. Decentralized Bidding with Monopoly Tax
        bids = {}
        for aid in active_ids:
            agent = agents[aid]
            monopoly_tax = min(0.8, agent.task_count * 0.05)
            # Homogeneous bidding metric (energy * capacity, penalizing monopoly)
            bid_val = agent.energy * (1.0 - monopoly_tax) + random.uniform(-0.5, 0.5)
            bids[aid] = bid_val
            
        primary_id = max(bids, key=bids.get)
        primary = agents[primary_id]
        print(f"  [Bid Won] Cell {primary_id} claims task (Energy: {primary.energy:.1f}, Monopoly Tax: {min(0.8, primary.task_count * 0.05)*100:.1f}%)")
        
        # Execute Task
        res = primary.solve(prompt)
        answer = res["text"].strip()
        success = verify_task(answer, expected)
        primary.record_attempt(answer, "verified", success, prompt=prompt)
        
        final_success = success
        final_solver_id = primary_id
        
        if success:
            print(f"  [Primary Success] Cell {primary_id} resolved the task.")
            primary.pending_reward += 25.0
            primary.task_count += 1
            world.update(True)
        else:
            print(f"  [Primary Failure] Cell {primary_id} failed. Emitting delegation request...")
            primary.adjust_energy(-20)
            
            # 2. Decentralized Peer-to-Peer Delegation based on local trust scores
            backup_bids = {}
            for aid in active_ids:
                if aid != primary_id:
                    agent = agents[aid]
                    trust = primary.trust_scores.get(aid, 0.5)
                    monopoly_tax = min(0.8, agent.task_count * 0.05)
                    backup_bid_val = agent.energy * trust * (1.0 - monopoly_tax) + random.uniform(-0.25, 0.25)
                    backup_bids[aid] = backup_bid_val
                    
            if backup_bids:
                helper_id = max(backup_bids, key=backup_bids.get)
                helper = agents[helper_id]
                log_delegation(primary_id, helper_id)
                print(f"  [Delegation claimed] Helper {helper_id} accepted delegation (Trust: {primary.trust_scores.get(helper_id, 0.5):.2f})")
                
                # Helper attempt
                helper_res = helper.solve(f"Previous agent failed. Objective:\n{prompt}\nSolve and output ONLY answer.")
                helper_answer = helper_res["text"].strip()
                final_success = verify_task(helper_answer, expected)
                helper.record_attempt(helper_answer, "verified_backup", final_success, prompt=prompt)
                
                # Trust network updates (local & symmetric)
                if final_success:
                    print(f"    [Helper Success] Helper {helper_id} resolved delegated task.")
                    helper.pending_reward += 25.0
                    helper.task_count += 1
                    final_solver_id = helper_id
                    
                    # Local & Symmetric reinforcement
                    primary.trust_scores[helper_id] = min(1.0, primary.trust_scores.get(helper_id, 0.5) + 0.1)
                    helper.trust_scores[primary_id] = min(1.0, helper.trust_scores.get(primary_id, 0.5) + 0.05) # reciprocal bonus
                    world.update(True)
                else:
                    print(f"    [Helper Failure] Helper {helper_id} failed.")
                    helper.adjust_energy(-20)
                    
                    # Penalize trust locally
                    primary.trust_scores[helper_id] = max(0.0, primary.trust_scores.get(helper_id, 0.5) - 0.05)
                    world.update(False)
            else:
                world.update(False)
                
        # Update Specialization Matrix on success
        if final_success:
            if final_solver_id not in spec_matrix:
                spec_matrix[final_solver_id] = {"Math": 0, "Cyber": 0}
            spec_matrix[final_solver_id][domain] = spec_matrix[final_solver_id].get(domain, 0) + 1
            success_history.append({"agent_id": final_solver_id, "domain": domain, "step": step})
            
        # 3. Delayed Reward Distribution (Steps 9, 19)
        phase_end = (step % 10 == 9)
        if phase_end:
            total_pending = sum(a.pending_reward for a in agents.values())
            print(f"  [Delayed Reward Phase End] Distributing pending energy pool: {total_pending:.1f}")
            if total_pending > 0:
                for aid in list(agents.keys()):
                    agent = agents[aid]
                    share_80 = 0.8 * agent.pending_reward
                    agent.adjust_energy(share_80)
                    
                    # 20% shared with highest affinity peer
                    share_20 = 0.2 * agent.pending_reward
                    if agent.trust_scores:
                        active_peers = [p for p in agent.trust_scores if p in agents]
                        if active_peers:
                            highest_peer = max(active_peers, key=agent.trust_scores.get)
                            agents[highest_peer].adjust_energy(share_20)
                            print(f"    Energy Share: {aid} shared {share_20:.1f} with peer {highest_peer}")
                            
                    agent.pending_reward = 0.0
                    
        # 4. Metabolic decay and evolutionary cycle
        for aid in list(agents.keys()):
            agent = agents[aid]
            # Monopoly decay tax
            decay = 6 + (agent.task_count * 0.5)
            agent.adjust_energy(-decay)
            
            # Prune dead cells
            if agent.is_dead():
                print(f"  [GC] Cell {aid} died (depleted energy). Pruning from swarm.")
                del agents[aid]
                continue
                
            # Division (reproduction) with trust crossover inheritance
            if agent.energy >= 120 and len(agents) < 8:
                child_id = f"Cell-replica-{random.randint(100, 999)}"
                print(f"  [Swarm Division] Cell {aid} splits. Spawned child {child_id}.")
                agents[child_id] = MicroAgent(child_id, "undifferentiated_cell", agent.system_prompt, client, somatic_memory=s_mem)
                agents[child_id].use_somatic_memory = use_somatic_memory
                agents[child_id].strategy = agent.strategy
                agents[child_id].task_count = 0
                agents[child_id].failures_count = 0
                agents[child_id].pending_reward = 0.0
                agents[child_id].trust_scores = {}
                
                # Division cost
                agent.adjust_energy(-45)
                reconcile_trust(agents)
                
                # Crossover trust score inheritance
                parent_peers = [p for p in agent.trust_scores.keys() if p in agents]
                if parent_peers:
                    best_peer = max(parent_peers, key=lambda p: agent.trust_scores[p])
                    for peer in agents:
                        if peer != child_id:
                            p_trust = agent.trust_scores.get(peer, 0.5)
                            b_trust = agents[best_peer].trust_scores.get(peer, 0.5)
                            agents[child_id].trust_scores[peer] = max(0.0, min(1.0, 0.5 * p_trust + 0.5 * b_trust + random.uniform(-0.05, 0.05)))
                            
        reconcile_trust(agents)
        
        # 5. Window-Based Metrics Calculation (Window K = 10 steps)
        # Calculate FDI and Hub Dominance
        window_start = max(0, step - 9)
        window_history = [item for item in success_history if window_start <= item["step"] <= step]
        
        fdi_val = compute_fdi_window(window_history)
        
        # Hub Dominance
        active_ids = list(agents.keys())
        avg_trusts = []
        for aid in active_ids:
            links = [agents[aid].trust_scores[bid] for bid in active_ids if bid != aid]
            avg_trusts.append(np.mean(links) if links else 0.5)
        hub_dom = np.std(avg_trusts) if len(avg_trusts) > 1 else 0.0
        
        history_fdi.append(fdi_val)
        history_dominance.append(hub_dom)
        history_integrity.append(world.integrity)
        history_pop_size.append(len(active_ids))
        
        results.append({
            "task_id": task["id"],
            "success": final_success,
            "tokens": client.get_weighted_tokens(),
            "fdi": fdi_val,
            "dominance": hub_dom,
            "integrity": world.integrity,
            "pop_size": len(active_ids)
        })
        time.sleep(0.1)
        
    return results, spec_matrix, {
        "fdi": history_fdi,
        "dominance": history_dominance,
        "integrity": history_integrity,
        "pop_size": history_pop_size
    }


# =====================================================================
# 5. EXPERIMENTAL COMPARATIVE RUN
# =====================================================================

def main():
    print("[Warm-up] Initializing qwen2.5:0.5b in GPU memory...")
    client.pull_model("qwen2.5:0.5b")
    client.generate(prompt="Hello", model_name="qwen2.5:0.5b")
    print("[Warm-up] Done.")
    
    # 1. Run all conditions
    res_a = run_group_a()
    res_b = run_group_b()
    res_c, spec_c, metrics_c = run_swarm(use_somatic_memory=True, group_name="Group C (Emergent Swarm)")
    res_c_abl, spec_c_abl, metrics_c_abl = run_swarm(use_somatic_memory=False, group_name="Group C-Ablated (No Memory)")
    
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)
    
    # 2. Save comparative metrics CSV
    print("\n[Save] Saving results/scientific_mvp_metrics.csv...")
    with open("results/scientific_mvp_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Task_ID", "Domain", 
            "GroupA_Success", "GroupA_Tokens", 
            "GroupB_Success", "GroupB_Tokens", 
            "GroupC_Success", "GroupC_Tokens", "GroupC_FDI", "GroupC_Dominance", "GroupC_Integrity", "GroupC_PopSize",
            "GroupCAbl_Success", "GroupCAbl_Tokens", "GroupCAbl_FDI", "GroupCAbl_Dominance", "GroupCAbl_Integrity", "GroupCAbl_PopSize"
        ])
        for i in range(len(TASKS)):
            task = TASKS[i]
            writer.writerow([
                task["id"], task["domain"],
                1 if res_a[i]["success"] else 0, res_a[i]["tokens"],
                1 if res_b[i]["success"] else 0, res_b[i]["tokens"],
                1 if res_c[i]["success"] else 0, res_c[i]["tokens"], metrics_c["fdi"][i], metrics_c["dominance"][i], metrics_c["integrity"][i], metrics_c["pop_size"][i],
                1 if res_c_abl[i]["success"] else 0, res_c_abl[i]["tokens"], metrics_c_abl["fdi"][i], metrics_c_abl["dominance"][i], metrics_c_abl["integrity"][i], metrics_c_abl["pop_size"][i]
            ])
            
    # 3. Save Specialization Matrices
    print("[Save] Saving results/specialization_matrix.csv...")
    with open("results/specialization_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "Math", "Cyber"])
        for aid, counts in spec_c.items():
            writer.writerow([aid, counts.get("Math", 0), counts.get("Cyber", 0)])
            
    with open("results/specialization_matrix_ablated.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "Math", "Cyber"])
        for aid, counts in spec_c_abl.items():
            writer.writerow([aid, counts.get("Math", 0), counts.get("Cyber", 0)])
            
    # Copy to brain folder for user visibility
    brain_dir = "/home/destritux/.gemini/antigravity-cli/brain/1f9283d8-d5ea-4cf5-b98d-5d0494e22db2"
    os.makedirs(brain_dir, exist_ok=True)
    shutil.copy("results/scientific_mvp_metrics.csv", os.path.join(brain_dir, "scientific_mvp_metrics.csv"))
    shutil.copy("results/specialization_matrix.csv", os.path.join(brain_dir, "specialization_matrix.csv"))
    shutil.copy("results/specialization_matrix_ablated.csv", os.path.join(brain_dir, "specialization_matrix_ablated.csv"))
    print("[Save] CSV logs saved successfully.")
    
    # 4. Calculate Phase-wise Success Rates and Learning Deltas
    # Math: tasks 0-4 (first 5), 5-9 (last 5). Cyber: tasks 10-14 (first 5), 15-19 (last 5)
    def calc_stats(results_list):
        math_first = sum(1 for r in results_list[0:5] if r["success"]) / 5.0
        math_last = sum(1 for r in results_list[5:10] if r["success"]) / 5.0
        cyber_first = sum(1 for r in results_list[10:15] if r["success"]) / 5.0
        cyber_last = sum(1 for r in results_list[15:20] if r["success"]) / 5.0
        
        math_overall = sum(1 for r in results_list[0:10] if r["success"]) / 10.0
        cyber_overall = sum(1 for r in results_list[10:20] if r["success"]) / 10.0
        
        return {
            "math_overall": math_overall,
            "cyber_overall": cyber_overall,
            "math_delta": math_last - math_first,
            "cyber_delta": cyber_last - cyber_first
        }
        
    stats_a = calc_stats(res_a)
    stats_b = calc_stats(res_b)
    stats_c = calc_stats(res_c)
    stats_c_abl = calc_stats(res_c_abl)
    
    print("\n" + "="*80)
    print("COMPARATIVE STUDY METRICS SUMMARY")
    print("="*80)
    print("Math Sequences Phase (Overall Accuracy | Learning Delta):")
    print(f"  Group A (Monolithic):  {stats_a['math_overall']*100:.1f}% | Delta: {stats_a['math_delta']*100:+.1f}%")
    print(f"  Group B (Orchestrated): {stats_b['math_overall']*100:.1f}% | Delta: {stats_b['math_delta']*100:+.1f}%")
    print(f"  Group C (Emergent):    {stats_c['math_overall']*100:.1f}% | Delta: {stats_c['math_delta']*100:+.1f}%")
    print(f"  Group C-Ablated:       {stats_c_abl['math_overall']*100:.1f}% | Delta: {stats_c_abl['math_delta']*100:+.1f}%")
    
    print("\nCyberdefense Logs Phase (Overall Accuracy | Learning Delta):")
    print(f"  Group A (Monolithic):  {stats_a['cyber_overall']*100:.1f}% | Delta: {stats_a['cyber_delta']*100:+.1f}%")
    print(f"  Group B (Orchestrated): {stats_b['cyber_overall']*100:.1f}% | Delta: {stats_b['cyber_delta']*100:+.1f}%")
    print(f"  Group C (Emergent):    {stats_c['cyber_overall']*100:.1f}% | Delta: {stats_c['cyber_delta']*100:+.1f}%")
    print(f"  Group C-Ablated:       {stats_c_abl['cyber_overall']*100:.1f}% | Delta: {stats_c_abl['cyber_delta']*100:+.1f}%")
    
    # Write summary stats file
    with open("results/scientific_mvp_summary.txt", "w") as f:
        f.write("=== COMPARATIVE STUDY METRICS SUMMARY ===\n")
        f.write(f"Group A (Monolithic): Math={stats_a['math_overall']*100:.1f}% (Delta={stats_a['math_delta']*100:+.1f}%), Cyber={stats_a['cyber_overall']*100:.1f}% (Delta={stats_a['cyber_delta']*100:+.1f}%)\n")
        f.write(f"Group B (Orchestrated): Math={stats_b['math_overall']*100:.1f}% (Delta={stats_b['math_delta']*100:+.1f}%), Cyber={stats_b['cyber_overall']*100:.1f}% (Delta={stats_b['cyber_delta']*100:+.1f}%)\n")
        f.write(f"Group C (Emergent): Math={stats_c['math_overall']*100:.1f}% (Delta={stats_c['math_delta']*100:+.1f}%), Cyber={stats_c['cyber_overall']*100:.1f}% (Delta={stats_c['cyber_delta']*100:+.1f}%)\n")
        f.write(f"Group C-Ablated: Math={stats_c_abl['math_overall']*100:.1f}% (Delta={stats_c_abl['math_delta']*100:+.1f}%), Cyber={stats_c_abl['cyber_overall']*100:.1f}% (Delta={stats_c_abl['cyber_delta']*100:+.1f}%)\n")
    shutil.copy("results/scientific_mvp_summary.txt", os.path.join(brain_dir, "scientific_mvp_summary.txt"))
    
    # 5. Plot Curves
    print("\n[Plot] Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    steps = range(1, len(TASKS) + 1)
    
    def rolling_avg(arr, window=3):
        clean = []
        for i in range(len(arr)):
            sub = arr[max(0, i-window+1):i+1]
            clean.append(np.mean(sub))
        return clean
        
    rolling_a = rolling_avg([1 if r["success"] else 0 for r in res_a])
    rolling_b = rolling_avg([1 if r["success"] else 0 for r in res_b])
    rolling_c = rolling_avg([1 if r["success"] else 0 for r in res_c])
    rolling_c_abl = rolling_avg([1 if r["success"] else 0 for r in res_c_abl])
    
    # Plot 1: Paradigm Shift Recovery Curves
    ax1.plot(steps, rolling_a, label="Group A (Monolithic)", color="#1f77b4", linewidth=2, marker="o", markevery=2)
    ax1.plot(steps, rolling_b, label="Group B (Orchestrated)", color="#aec7e8", linewidth=2, marker="s", markevery=2)
    ax1.plot(steps, rolling_c, label="Group C (Emergent Swarm)", color="#2ca02c", linewidth=2.5, marker="^", markevery=2)
    ax1.plot(steps, rolling_c_abl, label="Group C-Ablated (No Memory)", color="#d62728", linewidth=2, linestyle="--", marker="d", markevery=2)
    
    ax1.axvline(x=10, color="gray", linestyle="-.", alpha=0.7)
    ax1.text(5, 0.95, "Math Sequences", ha="center", fontsize=10, fontweight="bold")
    ax1.text(15, 0.95, "Cyber Intrusion Logs", ha="center", fontsize=10, fontweight="bold")
    
    ax1.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rolling Accuracy (Window=3)", fontsize=11, fontweight="bold")
    ax1.set_title("Paradigm Shift Recovery Curve", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: FDI and Hub Dominance Curves (C vs C-Ablated)
    ax2.plot(steps, metrics_c["fdi"], label="Group C: FDI", color="#2ca02c", linewidth=2.5)
    ax2.plot(steps, metrics_c_abl["fdi"], label="Group C-Ablated: FDI", color="#d62728", linewidth=2, linestyle="--")
    
    ax2.plot(steps, metrics_c["dominance"], label="Group C: Hub Dominance", color="#ff7f0e", linewidth=2)
    ax2.plot(steps, metrics_c_abl["dominance"], label="Group C-Ablated: Hub Dominance", color="#bcbd22", linewidth=1.5, linestyle="--")
    
    ax2.axvline(x=10, color="gray", linestyle="-.", alpha=0.5)
    
    ax2.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value (Index 0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax2.set_title("Emergent Specialization vs Centralization", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/scientific_mvp_curves.png", dpi=300)
    shutil.copy("results/scientific_mvp_curves.png", os.path.join(brain_dir, "scientific_mvp_curves.png"))
    plt.close()
    print("[Plot] Saved results/scientific_mvp_curves.png successfully.")

if __name__ == "__main__":
    main()
