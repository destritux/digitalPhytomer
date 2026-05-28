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


def compute_persistence_score(spec_matrix):
    """
    Computes the Persistence Score:
    Persistence = dominant_domain_tasks / total_tasks_solved for each agent.
    Returns the average across all agents that solved at least one task.
    """
    persistences = []
    for aid, counts in spec_matrix.items():
        math_count = counts.get("Math", 0)
        cyber_count = counts.get("Cyber", 0)
        total = math_count + cyber_count
        if total > 0:
            dominant = max(math_count, cyber_count)
            persistences.append(dominant / total)
    if persistences:
        return np.mean(persistences)
    else:
        return 1.0  # Default to 1.0 if no tasks have been solved yet


def save_spec_matrix_csv(spec_matrix, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "Math", "Cyber"])
        for aid in sorted(spec_matrix.keys()):
            writer.writerow([aid, spec_matrix[aid].get("Math", 0), spec_matrix[aid].get("Cyber", 0)])


def plot_specialization_heatmaps(matrix_c, matrix_c_abl, phase_name, filepath):
    agents_list = [f"Cell-{i+1:03d}" for i in range(8)]
    domains = ["Math", "Cyber"]
    
    # Convert to 2D numpy arrays
    data_c = np.zeros((8, 2))
    data_abl = np.zeros((8, 2))
    
    for idx, aid in enumerate(agents_list):
        data_c[idx, 0] = matrix_c.get(aid, {}).get("Math", 0)
        data_c[idx, 1] = matrix_c.get(aid, {}).get("Cyber", 0)
        
        data_abl[idx, 0] = matrix_c_abl.get(aid, {}).get("Math", 0)
        data_abl[idx, 1] = matrix_c_abl.get(aid, {}).get("Cyber", 0)
        
    # Shared colorbar scale
    vmin = 0
    vmax = max(data_c.max(), data_abl.max(), 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Group C heatmap
    im1 = ax1.imshow(data_c, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title("Group C (Emergent Swarm)", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(2))
    ax1.set_xticklabels(domains, fontsize=10, fontweight="bold")
    ax1.set_yticks(range(8))
    ax1.set_yticklabels(agents_list)
    ax1.set_ylabel("Agents", fontsize=11, fontweight="bold")
    
    # Annotate Group C
    for i in range(8):
        for j in range(2):
            ax1.text(j, i, f"{int(data_c[i, j])}", ha="center", va="center", 
                     color="black" if data_c[i, j] < vmax/2 else "white", fontweight="bold")
                     
    # Group C-Ablated heatmap
    im2 = ax2.imshow(data_abl, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title("Group C-Ablated (No Somatic Memory)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(domains, fontsize=10, fontweight="bold")
    ax2.set_yticks(range(8))
    ax2.set_yticklabels([]) # Hide yticklabels for the second plot since they are the same
    
    # Annotate Group C-Ablated
    for i in range(8):
        for j in range(2):
            ax2.text(j, i, f"{int(data_abl[i, j])}", ha="center", va="center", 
                     color="black" if data_abl[i, j] < vmax/2 else "white", fontweight="bold")
                     
    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label="Successful Tasks Solved")
    
    plt.suptitle(f"Domain Specialization Heatmap - {phase_name}", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def run_swarm(use_somatic_memory: bool, group_name: str):
    print("\n" + "="*80)
    print(f"RUNNING {group_name} (Somatic Memory = {use_somatic_memory})")
    print("="*80)
    client.reset_stats()
    
    # Initialize homogeneous population of N=8 agents
    agents = {}
    for i in range(8):
        aid = f"Cell-{i+1:03d}"
        agents[aid] = MicroAgent(
            agent_id=aid,
            role="undifferentiated_cell",
            system_prompt=f"You are a generic swarm cell '{aid}'. Solve the task and output only the final answer.",
            client=client,
            somatic_memory=SomaticVectorStore() # Isolated local memory store
        )
        agents[aid].use_somatic_memory = use_somatic_memory
        agents[aid].strategy = "undifferentiated"
        agents[aid].solved_count = 0
        agents[aid].trust_scores = {}
        
    # Reconcile trust scores uniformly to 0.5
    def reconcile_trust(pop):
        for aid in pop:
            for bid in pop:
                if aid != bid and bid not in pop[aid].trust_scores:
                    pop[aid].trust_scores[bid] = 0.5
                    
    reconcile_trust(agents)
    
    # Specialization tracking matrix: matrix[agent_id][domain] = count
    spec_matrix = {aid: {"Math": 0, "Cyber": 0} for aid in agents.keys()}
    success_history = []  # list of {"agent_id": aid, "domain": dom, "step": step}
    
    # Telemetry histories
    history_fdi = []
    history_dominance = []
    history_mean_trust = []
    history_max_trust = []
    history_coordination_entropy = []
    history_persistence = []
    
    results = []
    
    # Delegation mapping: delegations[src][dst] = count
    delegations = {}
    def log_delegation(src, dst):
        if src not in delegations:
            delegations[src] = {}
        delegations[src][dst] = delegations[src].get(dst, 0) + 1
        
    spec_matrix_p1 = None
    
    for step, task in enumerate(TASKS):
        print(f"\n[{group_name}] Step {step} Task {task['id']} ({task['domain']})...")
        prompt = task["prompt"]
        expected = task["expected"]
        domain = task["domain"]
        
        active_ids = sorted(list(agents.keys()))
        
        # 1. Bidding based on competence (somatic memory similarity) and monopoly tax
        bids = {}
        for aid in active_ids:
            agent = agents[aid]
            monopoly_tax = min(0.4, agent.solved_count * 0.015)
            
            competence = 0.0
            if use_somatic_memory:
                q_emb = client.get_embeddings(prompt)
                matches = agent.memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                if matches:
                    doc_domain = matches[0]["metadata"].get("domain")
                    if doc_domain == domain:
                        competence = matches[0]["score"]
                    else:
                        competence = matches[0]["score"] * 0.1
                    
            bid_val = competence * 0.6 + (1.0 - monopoly_tax) * 0.4 + random.uniform(-0.05, 0.05)
            bids[aid] = bid_val
            
        primary_id = max(bids, key=bids.get)
        primary = agents[primary_id]
        primary_tax = min(0.4, primary.solved_count * 0.015)
        print(f"  [Bid Won] Cell {primary_id} claims task (Monopoly Tax: {primary_tax*100:.1f}%)")
        
        # Execute Task
        res = primary.solve(prompt)
        answer = res["text"].strip()
        success = verify_task(answer, expected)
        primary.record_attempt(answer, "verified", success, prompt=prompt)
        
        final_success = success
        final_solver_id = primary_id
        
        if success:
            print(f"  [Primary Success] Cell {primary_id} resolved the task.")
            # Record success pattern in somatic memory
            if use_somatic_memory:
                emb = client.get_embeddings(prompt)
                primary.memory.vector_store.add_document(
                    text=f"Domain: {domain}. Task: {prompt}. Solution: {expected}",
                    embedding=emb,
                    metadata={"domain": domain, "type": "success_pattern"}
                )
        else:
            print(f"  [Primary Failure] Cell {primary_id} failed. Emitting delegation request...")
            
            # 2. Decentralized Peer-to-Peer Delegation
            backup_bids = {}
            for aid in active_ids:
                if aid != primary_id:
                    agent = agents[aid]
                    trust = primary.trust_scores.get(aid, 0.5)
                    monopoly_tax = min(0.4, agent.solved_count * 0.015)
                    
                    competence = 0.0
                    if use_somatic_memory:
                        q_emb = client.get_embeddings(prompt)
                        matches = agent.memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                        if matches:
                            doc_domain = matches[0]["metadata"].get("domain")
                            if doc_domain == domain:
                                competence = matches[0]["score"]
                            else:
                                competence = matches[0]["score"] * 0.1
                            
                    backup_bid_val = trust * 0.4 + competence * 0.3 + (1.0 - monopoly_tax) * 0.3 + random.uniform(-0.05, 0.05)
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
                
                if final_success:
                    print(f"    [Helper Success] Helper {helper_id} resolved delegated task.")
                    final_solver_id = helper_id
                    
                    # Record success pattern in helper's somatic memory
                    if use_somatic_memory:
                        emb = client.get_embeddings(prompt)
                        helper.memory.vector_store.add_document(
                            text=f"Domain: {domain}. Task: {prompt}. Solution: {expected}",
                            embedding=emb,
                            metadata={"domain": domain, "type": "success_pattern"}
                        )
                        
                    # Trust network updates (local & symmetric, moderate deltas)
                    primary.trust_scores[helper_id] = min(1.0, primary.trust_scores.get(helper_id, 0.5) + 0.03)
                    helper.trust_scores[primary_id] = min(1.0, helper.trust_scores.get(primary_id, 0.5) + 0.01)
                else:
                    print(f"    [Helper Failure] Helper {helper_id} failed.")
                    # Penalize trust locally (delegator trust in helper)
                    primary.trust_scores[helper_id] = max(0.0, primary.trust_scores.get(helper_id, 0.5) - 0.02)
                    
        # Update Specialization Matrix and Solved Count on success
        if final_success:
            spec_matrix[final_solver_id][domain] += 1
            agents[final_solver_id].solved_count += 1
            success_history.append({"agent_id": final_solver_id, "domain": domain, "step": step})
            
        # 3. Calculate sliding-window metrics
        window_start = max(0, step - 9)
        window_history = [item for item in success_history if window_start <= item["step"] <= step]
        
        # FDI (NaN for first 9 tasks)
        if step < 9:
            fdi_val = np.nan
        else:
            fdi_val = compute_fdi_window(window_history)
            
        # Hub Dominance
        avg_trusts = []
        for aid in active_ids:
            links = [agents[aid].trust_scores.get(bid, 0.5) for bid in active_ids if bid != aid]
            avg_trusts.append(np.mean(links) if links else 0.5)
        hub_dom = np.std(avg_trusts) if len(avg_trusts) > 1 else 0.0
        
        # Mean & Max Trust
        trust_vals = []
        for aid in active_ids:
            for bid in active_ids:
                if aid != bid:
                    trust_vals.append(agents[aid].trust_scores.get(bid, 0.5))
        trust_vals = np.array(trust_vals)
        mean_tr = np.mean(trust_vals) if len(trust_vals) > 0 else 0.5
        max_tr = np.max(trust_vals) if len(trust_vals) > 0 else 0.5
        
        # Coordination Entropy
        if len(trust_vals) > 0 and np.sum(trust_vals) > 0:
            p_trusts = trust_vals / np.sum(trust_vals)
            coord_entropy = -np.sum(p_trusts * np.log2(p_trusts + 1e-9))
        else:
            coord_entropy = 0.0
            
        # Persistence Score
        persistence = compute_persistence_score(spec_matrix)
        
        # Log to histories
        history_fdi.append(fdi_val)
        history_dominance.append(hub_dom)
        history_mean_trust.append(mean_tr)
        history_max_trust.append(max_tr)
        history_coordination_entropy.append(coord_entropy)
        history_persistence.append(persistence)
        
        results.append({
            "task_id": task["id"],
            "success": final_success,
            "tokens": client.get_weighted_tokens(),
            "fdi": fdi_val,
            "dominance": hub_dom,
            "mean_trust": mean_tr,
            "max_trust": max_tr,
            "coordination_entropy": coord_entropy,
            "persistence": persistence
        })
        
        # Phase-wise matrix capture and saving
        if step == 9:
            spec_matrix_p1 = {aid: dict(counts) for aid, counts in spec_matrix.items()}
            suffix = "" if use_somatic_memory else "_ablated"
            save_spec_matrix_csv(spec_matrix_p1, f"results/specialization_matrix_phase1{suffix}.csv")
        elif step == 19:
            suffix = "" if use_somatic_memory else "_ablated"
            save_spec_matrix_csv(spec_matrix, f"results/specialization_matrix_phase2{suffix}.csv")
            
        time.sleep(0.1)
        
    return results, spec_matrix_p1, spec_matrix, {
        "fdi": history_fdi,
        "dominance": history_dominance,
        "mean_trust": history_mean_trust,
        "max_trust": history_max_trust,
        "coordination_entropy": history_coordination_entropy,
        "persistence": history_persistence
    }


def main():
    print("[Warm-up] Initializing qwen2.5:0.5b in GPU memory...")
    client.pull_model("qwen2.5:0.5b")
    client.generate(prompt="Hello", model_name="qwen2.5:0.5b")
    print("[Warm-up] Done.")
    
    # 1. Run all conditions
    res_a = run_group_a()
    res_b = run_group_b()
    res_c, spec_c_p1, spec_c_p2, metrics_c = run_swarm(use_somatic_memory=True, group_name="Group C (Emergent Swarm)")
    res_c_abl, spec_c_abl_p1, spec_c_abl_p2, metrics_c_abl = run_swarm(use_somatic_memory=False, group_name="Group C-Ablated (No Memory)")
    
    # Ensure results folder exists
    os.makedirs("results", exist_ok=True)
    
    # 2. Save comparative metrics CSV (tidy long-form)
    print("\n[Save] Saving results/scientific_mvp_metrics.csv...")
    with open("results/scientific_mvp_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "group", "accuracy_window", "fdi_window", 
            "mean_trust", "max_trust", "coordination_entropy", "persistence_score"
        ])
        
        # Helper to compute rolling accuracy
        def get_rolling_accs(res_list):
            accs = []
            for t in range(20):
                window = [1 if r["success"] else 0 for r in res_list[max(0, t-9):t+1]]
                accs.append(np.mean(window))
            return accs
            
        rolling_a = get_rolling_accs(res_a)
        rolling_b = get_rolling_accs(res_b)
        
        # Write Group A
        for t in range(20):
            writer.writerow([t, "A", rolling_a[t], "NaN", "NaN", "NaN", "NaN", "NaN"])
            
        # Write Group B
        for t in range(20):
            writer.writerow([t, "B", rolling_b[t], "NaN", "NaN", "NaN", "NaN", "NaN"])
            
        # Write Group C
        for t in range(20):
            writer.writerow([
                t, "C", 
                np.mean([1 if r["success"] else 0 for r in res_c[max(0, t-9):t+1]]),
                metrics_c["fdi"][t] if not np.isnan(metrics_c["fdi"][t]) else "NaN",
                metrics_c["mean_trust"][t],
                metrics_c["max_trust"][t],
                metrics_c["coordination_entropy"][t],
                metrics_c["persistence"][t]
            ])
            
        # Write Group C-Ablated
        for t in range(20):
            writer.writerow([
                t, "C-Ablated",
                np.mean([1 if r["success"] else 0 for r in res_c_abl[max(0, t-9):t+1]]),
                metrics_c_abl["fdi"][t] if not np.isnan(metrics_c_abl["fdi"][t]) else "NaN",
                metrics_c_abl["mean_trust"][t],
                metrics_c_abl["max_trust"][t],
                metrics_c_abl["coordination_entropy"][t],
                metrics_c_abl["persistence"][t]
            ])
            
    # Copy to brain folder for user visibility
    brain_dir = "/home/destritux/.gemini/antigravity-cli/brain/1f9283d8-d5ea-4cf5-b98d-5d0494e22db2"
    os.makedirs(brain_dir, exist_ok=True)
    shutil.copy("results/scientific_mvp_metrics.csv", os.path.join(brain_dir, "scientific_mvp_metrics.csv"))
    shutil.copy("results/specialization_matrix_phase1.csv", os.path.join(brain_dir, "specialization_matrix_phase1.csv"))
    shutil.copy("results/specialization_matrix_phase1_ablated.csv", os.path.join(brain_dir, "specialization_matrix_phase1_ablated.csv"))
    shutil.copy("results/specialization_matrix_phase2.csv", os.path.join(brain_dir, "specialization_matrix_phase2.csv"))
    shutil.copy("results/specialization_matrix_phase2_ablated.csv", os.path.join(brain_dir, "specialization_matrix_phase2_ablated.csv"))
    
    # Save the legacy final filenames for backwards compatibility
    shutil.copy("results/specialization_matrix_phase2.csv", "results/specialization_matrix.csv")
    shutil.copy("results/specialization_matrix_phase2_ablated.csv", "results/specialization_matrix_ablated.csv")
    shutil.copy("results/specialization_matrix.csv", os.path.join(brain_dir, "specialization_matrix.csv"))
    shutil.copy("results/specialization_matrix_ablated.csv", os.path.join(brain_dir, "specialization_matrix_ablated.csv"))
    print("[Save] CSV logs saved successfully.")
    
    # 3. Plot Specialization Heatmaps
    print("\n[Plot] Generating heatmaps...")
    plot_specialization_heatmaps(spec_c_p1, spec_c_abl_p1, "Phase 1 (Task 9)", "results/specialization_heatmap_phase1.png")
    plot_specialization_heatmaps(spec_c_p2, spec_c_abl_p2, "Phase 2 (Task 19)", "results/specialization_heatmap_phase2.png")
    shutil.copy("results/specialization_heatmap_phase1.png", os.path.join(brain_dir, "specialization_heatmap_phase1.png"))
    shutil.copy("results/specialization_heatmap_phase2.png", os.path.join(brain_dir, "specialization_heatmap_phase2.png"))
    print("[Plot] Saved specialization heatmaps successfully.")
    
    # 4. Calculate Phase-wise Success Rates and Learning Deltas
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
    print("COMPARATIVE STUDY SUMMARY")
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
    print("\n[Plot] Generating curves plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    steps = range(1, 21)
    
    # Compute rolling accuracies with window=3 for visualization curve
    def get_rolling_vis(res_list):
        accs = []
        for t in range(20):
            window = [1 if r["success"] else 0 for r in res_list[max(0, t-2):t+1]]
            accs.append(np.mean(window))
        return accs
        
    vis_a = get_rolling_vis(res_a)
    vis_b = get_rolling_vis(res_b)
    vis_c = get_rolling_vis(res_c)
    vis_c_abl = get_rolling_vis(res_c_abl)
    
    # Subplot 1: Paradigm Shift Recovery Curves (Rolling Accuracy)
    ax1.plot(steps, vis_a, label="Group A (Monolithic)", color="#1f77b4", linewidth=2, marker="o", markevery=2)
    ax1.plot(steps, vis_b, label="Group B (Orchestrated)", color="#aec7e8", linewidth=2, marker="s", markevery=2)
    ax1.plot(steps, vis_c, label="Group C (Emergent Swarm)", color="#2ca02c", linewidth=2.5, marker="^", markevery=2)
    ax1.plot(steps, vis_c_abl, label="Group C-Ablated (No Memory)", color="#d62728", linewidth=2, linestyle="--", marker="d", markevery=2)
    
    ax1.axvline(x=10, color="gray", linestyle="-.", alpha=0.7)
    ax1.text(5, 0.95, "Math Sequences", ha="center", fontsize=10, fontweight="bold")
    ax1.text(15, 0.95, "Cyber Intrusion Logs", ha="center", fontsize=10, fontweight="bold")
    
    ax1.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rolling Accuracy (Window=3)", fontsize=11, fontweight="bold")
    ax1.set_title("Paradigm Shift Recovery Curve", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 2: FDI and Persistence Curves (C vs C-Ablated)
    # Note: FDI has NaNs for the first 9 steps, matplotlib handles them gracefully
    ax2.plot(steps, metrics_c["fdi"], label="Group C: FDI", color="#2ca02c", linewidth=2.5)
    ax2.plot(steps, metrics_c_abl["fdi"], label="Group C-Ablated: FDI", color="#d62728", linewidth=2, linestyle="--")
    
    ax2.plot(steps, metrics_c["persistence"], label="Group C: Persistence", color="#1f77b4", linewidth=2)
    ax2.plot(steps, metrics_c_abl["persistence"], label="Group C-Ablated: Persistence", color="#ff7f0e", linewidth=1.5, linestyle="--")
    
    ax2.axvline(x=10, color="gray", linestyle="-.", alpha=0.5)
    ax2.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value (Index 0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax2.set_title("Emergent Specialization & Persistence", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 3: Trust Graph Metrics (Hub Dominance and Coordination Entropy)
    ax3.plot(steps, metrics_c["dominance"], label="Group C: Hub Dominance", color="#9467bd", linewidth=2)
    ax3.plot(steps, metrics_c_abl["dominance"], label="Group C-Ablated: Hub Dominance", color="#c5b0d5", linewidth=1.5, linestyle="--")
    
    # Normalize entropy to [0, 1] for visual plotting (absolute max entropy is log2(56) = 5.8)
    norm_entropy_c = [e / 5.8 for e in metrics_c["coordination_entropy"]]
    norm_entropy_c_abl = [e / 5.8 for e in metrics_c_abl["coordination_entropy"]]
    
    ax3.plot(steps, norm_entropy_c, label="Group C: Coord. Entropy", color="#e377c2", linewidth=2)
    ax3.plot(steps, norm_entropy_c_abl, label="Group C-Ablated: Coord. Entropy", color="#f7b6d2", linewidth=1.5, linestyle="--")
    
    ax3.axvline(x=10, color="gray", linestyle="-.", alpha=0.5)
    ax3.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Metric Value (Normalized 0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax3.set_title("Trust Network Dynamics", fontsize=13, fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/scientific_mvp_curves.png", dpi=300)
    shutil.copy("results/scientific_mvp_curves.png", os.path.join(brain_dir, "scientific_mvp_curves.png"))
    plt.close()
    print("[Plot] Saved results/scientific_mvp_curves.png successfully.")


if __name__ == "__main__":
    main()
