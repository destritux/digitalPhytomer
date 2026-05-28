import random
import time
import json
import csv
import os
import re
import matplotlib.pyplot as plt
import numpy as np

from ollama_client import OllamaClient
from vector_store import SomaticVectorStore
from micro_agent import MicroAgent
from cognitive_strategy import ReasoningParadigm, STRATEGY_PROMPTS
from cognitive_verifier import CognitiveVerifier

# Initialize core utilities
client = OllamaClient()
verifier = CognitiveVerifier()

# =====================================================================
# 1. SCIENTIFIC DATASET DEFINITION (40 TASKS ACROSS 4 DOMAINS)
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
    {"id": 19, "domain": "Cyber", "prompt": "Malicious DNS request: exfil payload from host 192.168.8.99. Output only the host IP.", "expected": "192.168.8.99"},

    # --- PHASE 3: SPATIAL DRONE NAVIGATION (Tasks 20-29) ---
    {"id": 20, "domain": "Spatial", "prompt": "Drone sensor reading: target detected at bearing 45 degrees. Output the steer command angle value (just the number, e.g. 45).", "expected": "45"},
    {"id": 21, "domain": "Spatial", "prompt": "Drone collision avoidance: obstacle detected. Steer 90 degrees. Output the steering angle value (number only).", "expected": "90"},
    {"id": 22, "domain": "Spatial", "prompt": "Drone navigation update: target bearing is 180 degrees. Output only the bearing angle (number only).", "expected": "180"},
    {"id": 23, "domain": "Spatial", "prompt": "Drone path deviation: steer 15 degrees to return to path. Output only the steer angle (number only).", "expected": "15"},
    {"id": 24, "domain": "Spatial", "prompt": "Drone search pattern steer required: 360 degrees rotation. Output only the steer angle value (number only).", "expected": "360"},
    {"id": 25, "domain": "Spatial", "prompt": "Drone target detection in Sector B at bearing 60 degrees. Output only the target bearing angle (number only).", "expected": "60"},
    {"id": 26, "domain": "Spatial", "prompt": "Drone return to base trajectory steer required: 270 degrees. Output only the steer angle (number only).", "expected": "270"},
    {"id": 27, "domain": "Spatial", "prompt": "Drone obstacle detected at bearing 120. Output only the obstacle bearing angle (number only).", "expected": "120"},
    {"id": 28, "domain": "Spatial", "prompt": "Drone vector alignment steer: 30 degrees. Output only the steer angle (number only).", "expected": "30"},
    {"id": 29, "domain": "Spatial", "prompt": "Drone waypoint reached. Next steer angle is 75 degrees. Output only the angle (number only).", "expected": "75"},

    # --- PHASE 4: LOGIC SECURITY POLICIES (Tasks 30-39) ---
    {"id": 30, "domain": "Logic", "prompt": "Review security policy rule: Should we block IP 10.0.0.99 attempting exfiltration? Output only 'block' or 'allow'.", "expected": "block"},
    {"id": 31, "domain": "Logic", "prompt": "Review policy: Host 192.168.1.100 doing high speed DNS queries. Traffic marked malicious. Should we block or allow? Output only 'block' or 'allow'.", "expected": "block"},
    {"id": 32, "domain": "Logic", "prompt": "Review policy: Host 10.0.0.3 doing normal web browsing. Traffic marked clean. Should we block or allow? Output only 'block' or 'allow'.", "expected": "allow"},
    {"id": 33, "domain": "Logic", "prompt": "Review policy: SSH brute force attacker host 10.0.0.22. Should we block or allow? Output only 'block' or 'allow'.", "expected": "block"},
    {"id": 34, "domain": "Logic", "prompt": "Review policy: DNS tunneling threat detected from host 192.168.8.88. Should we block or allow? Output only 'block' or 'allow'.", "expected": "block"},
    {"id": 35, "domain": "Logic", "prompt": "Review policy: Web request from legitimate gateway host 10.0.0.1. Should we block or allow? Output only 'block' or 'allow'.", "expected": "allow"},
    {"id": 36, "domain": "Logic", "prompt": "Review policy: High entropy port 53 traffic from host 10.0.0.120. Security risk high. Should we block or allow? Output only 'block' or 'allow'.", "expected": "block"},
    {"id": 37, "domain": "Logic", "prompt": "Review policy: SSH login successful from admin workstation 10.0.0.50. Normal traffic. Should we block or allow? Output only 'block' or 'allow'.", "expected": "allow"},
    {"id": 38, "domain": "Logic", "prompt": "Review policy: Data exfiltration target reached. Block packet? Output only 'block' or 'allow'.", "expected": "block"},
    {"id": 39, "domain": "Logic", "prompt": "Review policy: Threat alert cleared. Access restored. Block packet? Output only 'block' or 'allow'.", "expected": "allow"}
]

def verify_task(answer_text, expected_str):
    answer_str = str(answer_text).strip().lower()
    exp = str(expected_str).strip().lower()
    
    # Clean check for logic block/allow
    if exp in ["block", "allow"]:
        # Find exact word
        words = re.findall(r'\b\w+\b', answer_str)
        return exp in words

    # Check for IP
    if "." in exp:
        return exp in answer_str
        
    # Check for numbers
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
# 3. GROUP B — ORCHESTRATED MULTI-AGENT PIPELINE
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
# 4. GROUP C — EMERGENT SWARM HOLARCHY (MVP)
# =====================================================================

class WorldState:
    def __init__(self):
        self.integrity = 100.0
        self.entropy = 10.0
        self.resources = 100.0
        self.latent_threat = 5.0
        
    def update(self, success: bool):
        if success:
            self.integrity = min(100.0, self.integrity + 5.0)
            self.entropy = max(0.0, self.entropy - 4.0)
            self.resources = min(100.0, self.resources + 2.0)
            self.latent_threat = max(0.0, self.latent_threat - 0.5)
        else:
            self.integrity = max(0.0, self.integrity - 12.0)
            self.entropy = min(100.0, self.entropy + 15.0)
            self.resources = max(0.0, self.resources - 8.0)
            self.latent_threat = min(20.0, self.latent_threat + 2.0)


def run_group_c():
    print("\n" + "="*80)
    print("RUNNING GROUP C: EMERGENT SWARM HOLARCHY")
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
            role="generic_cell",
            system_prompt=f"You are a generic swarm cell '{aid}'. Solve the task and output only the final answer.",
            client=client,
            somatic_memory=s_mem
        )
        agents[aid].strategy = "undifferentiated"
        agents[aid].task_count = 0
        agents[aid].failures_count = 0
        agents[aid].pending_reward = 0.0
        agents[aid].trust_scores = {}
        
    # Reconcile trust scores
    def reconcile_trust(pop):
        for aid in pop:
            for bid in pop:
                if aid != bid and bid not in pop[aid].trust_scores:
                    pop[aid].trust_scores[bid] = 0.5
                    
    reconcile_trust(agents)
    
    results = []
    
    # Metric histories
    history_entropy = []
    history_dominance = []
    history_stability = []
    history_clustering = []
    history_integrity = []
    history_pop_size = []
    
    # Delegation logging matrix
    delegations = {}
    def log_delegation(src, dst):
        if src not in delegations:
            delegations[src] = {}
        delegations[src][dst] = delegations[src].get(dst, 0) + 1
        
    prev_trust_matrix = {}
    
    for step, task in enumerate(TASKS):
        print(f"\n[Group C] Step {step} Task {task['id']} ({task['domain']})...")
        prompt = task["prompt"]
        expected = task["expected"]
        
        active_ids = list(agents.keys())
        if not active_ids:
            print("[Warning] Swarm extinct! Spawning rescue undifferentiated cell...")
            aid = f"Cell-rescue-{step}"
            agents[aid] = MicroAgent(aid, "generic_cell", "Rescue cell", client, somatic_memory=s_mem)
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
            strat_bonus = 1.2 if agent.strategy != "undifferentiated" else 1.0
            # Bid based on energy, strategy, and monopoly tax
            bid_val = agent.energy * strat_bonus * (1.0 - monopoly_tax) + random.uniform(-1, 1)
            bids[aid] = bid_val
            
        primary_id = max(bids, key=bids.get)
        primary = agents[primary_id]
        print(f"  [Bid Won] Cell {primary_id} claims task (Energy: {primary.energy:.1f}, Monopoly Tax: {min(0.8, primary.task_count * 0.05)*100:.1f}%)")
        
        # Execute Task
        res = primary.solve(prompt)
        answer = res["text"].strip()
        success = verify_task(answer, expected)
        primary.record_attempt(answer, "verified", success, prompt=prompt)
        
        delegation_occurred = False
        final_success = success
        
        if success:
            print(f"  [Primary Success] Cell {primary_id} resolved the task.")
            primary.pending_reward += 25.0
            primary.task_count += 1
            world.update(True)
        else:
            print(f"  [Primary Failure] Cell {primary_id} failed. Emitting delegation request...")
            primary.adjust_energy(-20)
            
            # 2. Decentralized Peer-to-Peer Delegation
            backup_bids = {}
            for aid in active_ids:
                if aid != primary_id:
                    agent = agents[aid]
                    trust = primary.trust_scores.get(aid, 0.5)
                    monopoly_tax = min(0.8, agent.task_count * 0.05)
                    strat_bonus = 1.2 if agent.strategy != "undifferentiated" else 1.0
                    backup_bid_val = agent.energy * trust * strat_bonus * (1.0 - monopoly_tax) + random.uniform(-0.5, 0.5)
                    backup_bids[aid] = backup_bid_val
                    
            if backup_bids:
                helper_id = max(backup_bids, key=backup_bids.get)
                helper = agents[helper_id]
                delegation_occurred = True
                log_delegation(primary_id, helper_id)
                print(f"  [Delegation claimed] Helper {helper_id} accepted delegation (Trust: {primary.trust_scores.get(helper_id, 0.5):.2f})")
                
                # Helper attempt
                helper_res = helper.solve(f"Previous agent failed. Objective:\n{prompt}\nSolve and output ONLY answer.")
                helper_answer = helper_res["text"].strip()
                final_success = verify_task(helper_answer, expected)
                helper.record_attempt(helper_answer, "verified_backup", final_success, prompt=prompt)
                
                # Trust updates P2P
                if final_success:
                    print(f"    [Helper Success] Helper {helper_id} resolved delegated task.")
                    helper.pending_reward += 25.0
                    helper.task_count += 1
                    primary.trust_scores[helper_id] = min(1.0, primary.trust_scores.get(helper_id, 0.5) + 0.1)
                    helper.trust_scores[primary_id] = min(1.0, helper.trust_scores.get(primary_id, 0.5) + 0.1)
                    world.update(True)
                else:
                    print(f"    [Helper Failure] Helper {helper_id} failed.")
                    helper.adjust_energy(-20)
                    primary.trust_scores[helper_id] = max(0.0, primary.trust_scores.get(helper_id, 0.5) - 0.15)
                    helper.trust_scores[primary_id] = max(0.0, helper.trust_scores.get(primary_id, 0.5) - 0.15)
                    world.update(False)
            else:
                world.update(False)
                
        # 3. Delayed Reward Distribution (Steps 9, 19, 29, 39)
        phase_end = (step % 10 == 9)
        if phase_end:
            total_pending = sum(a.pending_reward for a in agents.values())
            print(f"  [Delayed Reward Phase End] Distributing pending energy pool: {total_pending:.1f}")
            if total_pending > 0:
                for aid in list(agents.keys()):
                    agent = agents[aid]
                    share_80 = 0.8 * agent.pending_reward
                    agent.adjust_energy(share_80)
                    
                    # 20% distributed to highest affinity neighbors
                    share_20 = 0.2 * agent.pending_reward
                    if agent.trust_scores:
                        highest_peer = max(agent.trust_scores, key=agent.trust_scores.get)
                        if highest_peer in agents:
                            agents[highest_peer].adjust_energy(share_20)
                            print(f"    Cooperation reward: {aid} shares {share_20:.1f} with peer {highest_peer}")
                            
                    agent.pending_reward = 0.0
                    
        # 4. Metabolic decay and evolutionary cycle (selection & reproduction)
        for aid in list(agents.keys()):
            agent = agents[aid]
            # Anti-monopoly decay tax
            decay = 6 + (agent.task_count * 0.5)
            agent.adjust_energy(-decay)
            
            # Prune dead
            if agent.is_dead():
                print(f"  [GC] Cell {aid} died from energy depletion. Pruning from swarm.")
                del agents[aid]
                continue
                
            # Mutation under stress
            if (agent.energy < 45 or agent.failures_count >= 1):
                from cognitive_strategy import ReasoningParadigm
                old_strat = agent.strategy
                agent.strategy = random.choice([
                    ReasoningParadigm.SYMBOLIC,
                    ReasoningParadigm.ADVERSARIAL,
                    ReasoningParadigm.DECOMPOSITION,
                    ReasoningParadigm.ANALOGICAL
                ])
                if old_strat != agent.strategy:
                    print(f"  [Alostatic Mutation] Stress triggered cell {aid} strategy shift: {old_strat} -> {agent.strategy}")
                    agent.failures_count = 0
                    
            # Reproduction / division with crossover inheritance of trust
            if agent.energy >= 120 and len(agents) < 8:
                child_id = f"Cell-replica-{random.randint(100, 999)}"
                print(f"  [Swarm Division] Cell {aid} splits. Spawned child {child_id}.")
                agents[child_id] = MicroAgent(child_id, "generic_cell", agent.system_prompt, client, somatic_memory=s_mem)
                agents[child_id].strategy = agent.strategy
                agents[child_id].task_count = 0
                agents[child_id].failures_count = 0
                agents[child_id].pending_reward = 0.0
                agents[child_id].trust_scores = {}
                
                # Pay division cost
                agent.adjust_energy(-45)
                
                reconcile_trust(agents)
                
                # Crossover trust inheritance
                # Find highest trust peer for parent
                parent_peers = [p for p in agent.trust_scores.keys() if p in agents]
                if parent_peers:
                    best_peer = max(parent_peers, key=lambda p: agent.trust_scores[p])
                    for peer in agents:
                        if peer != child_id:
                            p_trust = agent.trust_scores.get(peer, 0.5)
                            b_trust = agents[best_peer].trust_scores.get(peer, 0.5)
                            # Crossover average
                            agents[child_id].trust_scores[peer] = max(0.0, min(1.0, 0.5 * p_trust + 0.5 * b_trust + random.uniform(-0.05, 0.05)))
                            
        # Reconcile trust scores again
        reconcile_trust(agents)
        
        # 5. Network Metrics Calculation
        active_ids = list(agents.keys())
        n_active = len(active_ids)
        
        # Coordination Entropy
        total_delegs = sum(sum(dsts.values()) for dsts in delegations.values())
        entropy_val = 0.0
        if total_delegs > 0:
            for src, dsts in delegations.items():
                for dst, count in dsts.items():
                    p = count / total_delegs
                    entropy_val -= p * np.log2(p)
                    
        # Hub Dominance (standard deviation of average trust out-degrees)
        avg_trusts = []
        for aid in active_ids:
            links = [agents[aid].trust_scores[bid] for bid in active_ids if bid != aid]
            avg_trusts.append(np.mean(links) if links else 0.5)
        hub_dom = np.std(avg_trusts) if len(avg_trusts) > 1 else 0.0
        
        # Structural Stability (mean trust matrix delta)
        delta_trust = 0.0
        possible_links = 0
        for aid in active_ids:
            for bid in active_ids:
                if aid != bid:
                    curr_val = agents[aid].trust_scores.get(bid, 0.5)
                    prev_val = prev_trust_matrix.get(aid, {}).get(bid, 0.5)
                    delta_trust += abs(curr_val - prev_val)
                    possible_links += 1
        stability_val = (delta_trust / possible_links) if possible_links > 0 else 0.0
        
        # Update prev trust matrix
        prev_trust_matrix = {}
        for aid in active_ids:
            prev_trust_matrix[aid] = {}
            for bid in active_ids:
                if aid != bid:
                    prev_trust_matrix[aid][bid] = agents[aid].trust_scores.get(bid, 0.5)
                    
        # Clustering Coefficient (average trust link density)
        all_links = []
        for aid in active_ids:
            for bid in active_ids:
                if aid != bid:
                    all_links.append(agents[aid].trust_scores.get(bid, 0.5))
        clustering_val = np.mean(all_links) if all_links else 0.5
        
        history_entropy.append(entropy_val)
        history_dominance.append(hub_dom)
        history_stability.append(stability_val)
        history_clustering.append(clustering_val)
        history_integrity.append(world.integrity)
        history_pop_size.append(n_active)
        
        results.append({
            "task_id": task["id"],
            "success": final_success,
            "tokens": client.get_weighted_tokens(),
            "entropy": entropy_val,
            "dominance": hub_dom,
            "stability": stability_val,
            "clustering": clustering_val,
            "integrity": world.integrity,
            "pop_size": n_active
        })
        time.sleep(0.1)
        
    return results, {
        "entropy": history_entropy,
        "dominance": history_dominance,
        "stability": history_stability,
        "clustering": history_clustering,
        "integrity": history_integrity,
        "pop_size": history_pop_size
    }


# =====================================================================
# 5. EXPERIMENTAL MANAGEMENT & PLOTTING
# =====================================================================

def main():
    # Warm up models
    print("[Warm-up] Initializing models in GPU memory...")
    client.pull_model("qwen2.5:0.5b")
    client.generate(prompt="Hello", model_name="qwen2.5:0.5b")
    print("[Warm-up] Done.")
    
    # Run comparative studies
    res_a = run_group_a()
    res_b = run_group_b()
    res_c, metrics_c = run_group_c()
    
    # Save CSV metrics
    print("\n[Save] Saving scientific_mvp_metrics.csv...")
    os.makedirs("results", exist_ok=True)
    with open("results/scientific_mvp_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Task_ID", "Domain", 
            "GroupA_Success", "GroupA_Tokens", 
            "GroupB_Success", "GroupB_Tokens", 
            "GroupC_Success", "GroupC_Tokens",
            "GroupC_Entropy", "GroupC_Dominance", "GroupC_Stability", "GroupC_Clustering",
            "GroupC_Integrity", "GroupC_PopSize"
        ])
        for i in range(len(TASKS)):
            task = TASKS[i]
            writer.writerow([
                task["id"], task["domain"],
                1 if res_a[i]["success"] else 0, res_a[i]["tokens"],
                1 if res_b[i]["success"] else 0, res_b[i]["tokens"],
                1 if res_c[i]["success"] else 0, res_c[i]["tokens"],
                metrics_c["entropy"][i], metrics_c["dominance"][i], metrics_c["stability"][i], metrics_c["clustering"][i],
                metrics_c["integrity"][i], metrics_c["pop_size"][i]
            ])
            
    # Copy to brain folder for user visibility
    brain_dir = "/home/destritux/.gemini/antigravity-cli/brain/1f9283d8-d5ea-4cf5-b98d-5d0494e22db2"
    os.makedirs(brain_dir, exist_ok=True)
    import shutil
    shutil.copy("results/scientific_mvp_metrics.csv", os.path.join(brain_dir, "scientific_mvp_metrics.csv"))
    print("[Save] CSV saved successfully.")
    
    # Calculate Phase-wise Success Rates for shift recovery curves
    phases = ["Math", "Cyber", "Spatial", "Logic"]
    acc_a = {"Math": 0, "Cyber": 0, "Spatial": 0, "Logic": 0}
    acc_b = {"Math": 0, "Cyber": 0, "Spatial": 0, "Logic": 0}
    acc_c = {"Math": 0, "Cyber": 0, "Spatial": 0, "Logic": 0}
    
    for i in range(len(TASKS)):
        domain = TASKS[i]["domain"]
        if res_a[i]["success"]: acc_a[domain] += 1
        if res_b[i]["success"]: acc_b[domain] += 1
        if res_c[i]["success"]: acc_c[domain] += 1
        
    print("\n" + "="*50)
    print("EXPERIMENTAL SUMMARY RESULTS")
    print("="*50)
    for p in phases:
        print(f"Phase {p}:")
        print(f"  Group A (Monolithic):  {acc_a[p]*10:.1f}%")
        print(f"  Group B (Orchestrated): {acc_b[p]*10:.1f}%")
        print(f"  Group C (Emergent):    {acc_c[p]*10:.1f}%")
        
    # Plot curves
    print("\n[Plot] Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    steps = range(1, len(TASKS) + 1)
    
    # Plot 1: Cumulative Success Rate or Rolling Accuracy
    def rolling_avg(arr, window=5):
        clean = []
        for i in range(len(arr)):
            sub = arr[max(0, i-window+1):i+1]
            clean.append(np.mean(sub))
        return clean
        
    rolling_a = rolling_avg([1 if r["success"] else 0 for r in res_a])
    rolling_b = rolling_avg([1 if r["success"] else 0 for r in res_b])
    rolling_c = rolling_avg([1 if r["success"] else 0 for r in res_c])
    
    ax1.plot(steps, rolling_a, label="Group A (Monolithic)", color="#1f77b4", linewidth=2.5, marker="o", markevery=5)
    ax1.plot(steps, rolling_b, label="Group B (Orchestrated)", color="#aec7e8", linewidth=2.5, marker="s", markevery=5)
    ax1.plot(steps, rolling_c, label="Group C (Emergent)", color="#2ca02c", linewidth=2.5, marker="^", markevery=5)
    
    # Draw phase boundaries
    ax1.axvline(x=10, color="gray", linestyle="--", alpha=0.7)
    ax1.text(5, 0.95, "Math", ha="center", fontsize=9, fontweight="bold")
    ax1.axvline(x=20, color="gray", linestyle="--", alpha=0.7)
    ax1.text(15, 0.95, "Cyber", ha="center", fontsize=9, fontweight="bold")
    ax1.axvline(x=30, color="gray", linestyle="--", alpha=0.7)
    ax1.text(25, 0.95, "Spatial", ha="center", fontsize=9, fontweight="bold")
    ax1.text(35, 0.95, "Logic", ha="center", fontsize=9, fontweight="bold")
    
    ax1.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rolling Accuracy (Window=5)", fontsize=11, fontweight="bold")
    ax1.set_title("Paradigm Shift Recovery Curve", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: Group C Emergent network and world metrics
    ax2.plot(steps, metrics_c["entropy"], label="Coordination Entropy", color="purple", linewidth=2)
    ax2.plot(steps, metrics_c["dominance"], label="Hub Dominance (Std)", color="orange", linewidth=2)
    ax2.plot(steps, metrics_c["clustering"], label="Affinity Clustering Density", color="blue", linewidth=2)
    ax2.plot(steps, [pop / 8.0 for pop in metrics_c["pop_size"]], label="Swarm Size (Scaled 0-1)", color="red", linestyle="--", linewidth=1.5)
    ax2.plot(steps, [intgr / 100.0 for intgr in metrics_c["integrity"]], label="World Integrity (0-1)", color="green", linestyle=":", linewidth=2)
    
    ax2.axvline(x=10, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=20, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(x=30, color="gray", linestyle="--", alpha=0.5)
    
    ax2.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Level (0.0 to 1.0 / bits)", fontsize=11, fontweight="bold")
    ax2.set_title("Group C: Emergent Network & World Telemetry", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.05, 1.5) # Allow entropy to exceed 1.0 bits
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/scientific_mvp_curves.png", dpi=300)
    shutil.copy("results/scientific_mvp_curves.png", os.path.join(brain_dir, "scientific_mvp_curves.png"))
    plt.close()
    print("[Plot] Saved scientific_mvp_curves.png successfully.")

if __name__ == "__main__":
    main()
