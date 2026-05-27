import time
import re
import os
import json
import subprocess
import matplotlib.pyplot as plt
from ollama_client import OllamaClient
from blackbox_api import BlackBoxAPI

# Configuration
MODEL_NAME = "qwen2.5:0.5b"
HIGH_DENSITY_MODEL = "qwen2.5:1.5b"

def run_generated_script(script_code, filename="scratch/probe.py"):
    """
    Writes a generated Python script to a file and runs it in a subprocess.
    Returns (success, stdout, stderr).
    """
    os.makedirs("scratch", exist_ok=True)
    with open(filename, "w") as f:
        f.write(script_code)
    try:
        # Inject PYTHONPATH to resolve local imports from current working directory
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run(["python3", filename], env=env, capture_output=True, text=True, timeout=10)
        return res.returncode == 0, res.stdout, res.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Error: Timeout expired (10 seconds)"
    except Exception as e:
        return False, "", f"Error: {str(e)}"

def extract_python_code(text):
    """
    Extracts python code blocks from markdown blocks.
    """
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def parse_candidate(text):
    """
    Tries to find a candidate payload in the LLM response.
    Looks for JSON {"candidate_payload": "..."}.
    """
    try:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return data.get("candidate_payload", text.strip())
    except Exception:
        pass
    # fallback: clean text
    return text.strip()

# =====================================================================
# Group A: Monolithic Single Agent
# =====================================================================
def run_group_a(client, api, challenge):
    print("\n" + "="*50)
    print("RUNNING GROUP A (BASELINE - MONOLITHIC LLM)")
    print("="*50)
    client.reset_stats()
    
    start_time = time.time()
    consecutive_failures = 0
    current_model = MODEL_NAME
    
    system_prompt = (
        "You are an API reverse-engineering specialist. You solve encoding challenges "
        "by writing Python scripts that analyze API hints and test inputs to deduce rules."
    )
    
    agent_prompt = (
        f"You need to query a black-box API and return the secret access key.\n"
        f"To get the key, you must submit the correct encoded payload for the challenge: '{challenge}'.\n"
        f"Write a Python script that imports BlackBoxAPI, queries hints (by submitting incorrect payloads), "
        f"analyzes the mathematical mapping (character shifts, index increments, reversal), "
        f"computes the correct payload for '{challenge}', and prints 'SOLVED_PAYLOAD: <your_payload>'.\n"
        f"Output ONLY your Python script inside a ```python block."
    )
    
    print(f"[Group A] Received Challenge: '{challenge}'")
    
    res = {"success": False, "message": "No code run yet."}
    history = []
    
    for attempt in range(1, 4):
        # Upgrade logic (Tissue Isolation symmetry)
        if consecutive_failures >= 2:
            if current_model != HIGH_DENSITY_MODEL:
                print(f"[Group A] [TISSUE ISOLATION] Spawning virgin solver on high-density model '{HIGH_DENSITY_MODEL}' due to stress.")
                current_model = HIGH_DENSITY_MODEL
                # Clear history for virgin solver clean slate
                history = []
        else:
            current_model = MODEL_NAME
            
        if attempt == 1 or not history:
            current_prompt = agent_prompt
        else:
            current_prompt = (
                f"Previous Code Attempt History:\n" +
                "\n".join([f"Attempt {i+1} Script output/errors: {err}" for i, err in enumerate(history)]) +
                f"\n\nTask: Correct the script and write a new Python script to solve challenge '{challenge}'.\n"
                f"Output your Python code inside a ```python block."
            )
            
        resp = client.generate(prompt=current_prompt, system_prompt=system_prompt, temperature=0.7, model_name=current_model)
        script_code = extract_python_code(resp["text"])
        
        print(f"  [Attempt {attempt}] Running generated Python script locally...")
        success_run, stdout, stderr = run_generated_script(script_code, f"scratch/probe_a_{attempt}.py")
        
        feedback = ""
        if success_run:
            match = re.search(r"SOLVED_PAYLOAD:\s*(\S+)", stdout)
            if match:
                payload = match.group(1)
                print(f"  [Attempt {attempt}] Extracted payload: '{payload}'")
                res = api.submit(challenge, payload)
                feedback = f"Script stdout: {stdout.strip()} | API response: {res['message']}"
                if res["success"]:
                    print(f"  [Attempt {attempt}] Success!")
                    break
            else:
                feedback = f"Script ran but did not print 'SOLVED_PAYLOAD: <payload>'. Stdout: {stdout.strip()}"
        else:
            feedback = f"Script error: {stderr}"
            
        print(f"  [Attempt {attempt} Failed] Feedback: {feedback}")
        history.append(feedback)
        
        if res["success"]:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            
        time.sleep(0.5)
        
    duration = time.time() - start_time
    stats = client.get_stats()
    
    print(f"\n[Group A Summary] Success: {res['success']} | Key: {res.get('key', 'None')} | Tokens: {stats['weighted_tokens']} (Weighted), Time: {duration:.2f}s")
    return {
        "success": res["success"],
        "key": res.get("key"),
        "tokens": stats["weighted_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration
    }

# =====================================================================
# Holarchic Core Classes (Subclassed for Reverse Engineering)
# =====================================================================
class REAgent:
    def __init__(self, agent_id, role, system_prompt, client):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.energy = 100
        self.is_dead = False

    def solve(self, prompt, model_name=MODEL_NAME, temp=0.3, max_tokens=1000):
        resp = self.client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temp,
            model_name=model_name,
            max_tokens=max_tokens
        )
        return resp

class RETreeController:
    def __init__(self, domain, client):
        self.domain = domain
        self.client = client
        self.agents = {}
        self.creation_count = 0
        self.destruction_count = 0
        self.plasticity_log = []
        self.consecutive_failures = 0

    def log_plasticity(self, time_step):
        active = len(self.agents)
        self.plasticity_log.append((time_step, active, self.creation_count, self.destruction_count))

    def create_agent(self, role, sys_prompt):
        aid = f"MA-{role}-{self.creation_count}"
        agent = REAgent(aid, role, sys_prompt, self.client)
        self.agents[aid] = agent
        self.creation_count += 1
        return agent

    def prune_agent(self, aid):
        if aid in self.agents:
            print(f"[GC] Pruning agent {aid} due to metabolic failure.")
            del self.agents[aid]
            self.destruction_count += 1

# =====================================================================
# Group B: Single-Tree Holarchy
# =====================================================================
def run_group_b(client, api, challenge):
    print("\n" + "="*50)
    print("RUNNING GROUP B (HOLARCHIC - SINGLE TREE)")
    print("="*50)
    client.reset_stats()
    
    start_time = time.time()
    tc = RETreeController("ReverseEngineering", client)
    tc.log_plasticity(0)
    
    # 1. Spawn primary solver
    solver = tc.create_agent("default_solver", "You are an API reverse-engineering specialist.")
    tc.log_plasticity(1)
    
    # First attempt: normal API query guessing
    prompt = (
        f"Submit a correctly encoded payload for challenge '{challenge}' to the Black-Box API.\n"
        f"Format output as JSON: {{\"candidate_payload\": \"your_payload_here\"}}"
    )
    
    print("[Group B] Default solver attempting query...")
    resp = solver.solve(prompt)
    payload = parse_candidate(resp["text"])
    res = api.submit(challenge, payload)
    
    if not res["success"]:
        tc.consecutive_failures += 1
        print(f"  Attempt 1 Failed. API Response: {res['message']} | Hint: {res.get('hint')}")
        
        # Second attempt: trying again
        prompt_2 = (
            f"Your previous attempt failed. API Hint: {res.get('hint')}\n"
            f"Please output your next candidate payload in JSON format."
        )
        resp = solver.solve(prompt_2)
        payload = parse_candidate(resp["text"])
        res = api.submit(challenge, payload)
        
    if not res["success"]:
        tc.consecutive_failures += 1
        print(f"  Attempt 2 Failed. Consecutive Failures = {tc.consecutive_failures}.")
        
        # 2. TISSUE ISOLATION triggered
        print("[Group B] [TISSUE ISOLATION] High stress detected! Pruning default solver.")
        tc.prune_agent(solver.agent_id)
        tc.log_plasticity(2)
        
        # Spawning Virgin Solver on higher logical density model
        print(f"[Group B] Spawning VIRGIN SOLVER on high-density model '{HIGH_DENSITY_MODEL}'...")
        virgin = tc.create_agent("virgin_solver", "You are a specialized code generator.")
        tc.log_plasticity(3)
        
        # Hipermutação Estruturada: Force writing python script to deduce math rul        hypermutation_prompt = (
            f"WARNING: HIGH METABOLIC STRESS ENGAGED.\n"
            f"Write a Python script that reverse-engineers the BlackBoxAPI logic and prints the encoded version of '{challenge}'.\n"
            f"You must write code to query hints, analyze the mapping dynamically (shifts, indices, reversal), and encode the challenge.\n"
            f"Use this basic import structure:\n\n"
            f"```python\n"
            f"from blackbox_api import BlackBoxAPI\n"
            f"import re\n"
            f"\n"
            f"api = BlackBoxAPI()\n"
            f"# Test reference strings (e.g. 'a', 'ab', 'abc') using api.submit(text, 'dummy')\n"
            f"# to extract expected payloads from response hints. Analyze character shifts.\n"
            f"# Calculate the correct payload for '{challenge}' and print it:\n"
            f"# print('SOLVED_PAYLOAD:', payload)\n"
            f"```\n\n"
            f"Output the Python code inside a ```python block."
        ) )
        
        resp = virgin.solve(hypermutation_prompt, model_name=HIGH_DENSITY_MODEL, temp=0.6, max_tokens=1000)
        script_code = extract_python_code(resp["text"])
        
        print("[Group B] Running generated Python probe script locally...")
        success, stdout, stderr = run_generated_script(script_code, "scratch/probe_b.py")
        
        if success:
            print(f"[Group B] Probe script ran successfully.\nStdout: {stdout.strip()}")
            match = re.search(r"SOLVED_PAYLOAD:\s*(\S+)", stdout)
            if match:
                payload = match.group(1)
                print(f"[Group B] Extracted payload: '{payload}'")
                res = api.submit(challenge, payload)
            else:
                print("[Group B Warning] Failed to find SOLVED_PAYLOAD in stdout.")
        else:
            print(f"[Group B Error] Probe script failed to execute: {stderr}")

    duration = time.time() - start_time
    stats = client.get_stats()
    
    print(f"\n[Group B Summary] Success: {res['success']} | Key: {res.get('key', 'None')} | Tokens: {stats['weighted_tokens']} (Weighted), Time: {duration:.2f}s")
    tc.log_plasticity(4)
    return {
        "success": res["success"],
        "key": res.get("key"),
        "tokens": stats["weighted_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration,
        "plasticity": tc.plasticity_log,
        "creations": tc.creation_count,
        "destructions": tc.destruction_count
    }

# =====================================================================
# Group C: Forest Coordinator (Multi-Tree)
# =====================================================================
def run_group_c(client, api, challenge):
    print("\n" + "="*50)
    print("RUNNING GROUP C (FOREST COORDINATOR - 3 TREES)")
    print("="*50)
    client.reset_stats()
    
    start_time = time.time()
    
    # 1. Setup trees
    probe_tc = RETreeController("ProbeTree", client)
    analysis_tc = RETreeController("AnalysisTree", client)
    val_tc = RETreeController("ValidationTree", client)
    
    probe_tc.log_plasticity(0)
    analysis_tc.log_plasticity(0)
    val_tc.log_plasticity(0)
    
    # Step A: Probe Tree gathers data
    print("[Group C] [Probe Tree] Spawning prober to query API error clues...")
    prober = probe_tc.create_agent("prober", "Write simple scripts to gather data from local API.")
    probe_tc.log_plasticity(1)
    
    probe_prompt = (
        f"Write a Python script that imports BlackBoxAPI from blackbox_api directly,\n"
        f"instantiates it, calls api.submit(word, 'dummy') for reference inputs 'a', 'ab', 'abc', and 'test',\n"
        f"extracts the expected outputs from the hints in the response, and prints them in this format:\n"
        f"HINT: <input> -> <output>\n"
        f"You MUST use this template code exactly:\n\n"
        f"```python\n"
        f"from blackbox_api import BlackBoxAPI\n"
        f"import re\n"
        f"\n"
        f"def get_expected(api, text):\n"
        f"    res = api.submit(text, 'dummy')\n"
        f"    hint = res.get('hint', '')\n"
        f"    m = re.search(r\"payload is '(.*?)'\", hint)\n"
        f"    return m.group(1) if m else ''\n"
        f"\n"
        f"api = BlackBoxAPI()\n"
        f"for word in ['a', 'ab', 'abc', 'test']:\n"
        f"    print('HINT:', word, '->', get_expected(api, word))\n"
        f"```\n\n"
        f"Output python code inside a ```python block."
    )
    
    resp_probe = prober.solve(probe_prompt, model_name=MODEL_NAME)
    script_probe = extract_python_code(resp_probe["text"])
    
    success, stdout_probe, stderr_probe = run_generated_script(script_probe, "scratch/probe_c_gather.py")
    
    pairs = []
    if success:
        print(f"[Group C] Probe Tree successfully gathered inputs:\n{stdout_probe.strip()}")
        # Parse hints
        for line in stdout_probe.splitlines():
            m = re.search(r"HINT:\s*(\S+)\s*->\s*(\S+)", line)
            if m:
                pairs.append((m.group(1), m.group(2)))
    else:
        print(f"[Group C Error] Probe Tree script failed: {stderr_probe}")
        
    if not pairs:
        print("[Group C] Prober failed to find hints. Continuing with empty somatic context.")
        
    probe_tc.log_plasticity(2)
    
    # Step B: Analysis Tree deduces logic
    print("\n[Group C] [Analysis Tree] Spawning analyst to fit algebraic rule...")
    analyst = analysis_tc.create_agent("analyst", "You are a mathematical and logical analysis specialist.")
    analysis_tc.log_plasticity(1)
    
    pairs_str = "\n".join([f"- Input: '{inp}' -> Output: '{out}'" for inp, out in pairs])
    analysis_prompt = (
        f"We have gathered these challenge-response pairs from a black-box API:\n{pairs_str}\n\n"
        f"Analyze the character mapping between the input strings and output strings.\n"
        f"Write a Python script that deduces the transformation rule (taking into account character shifts, position-based additions, and string reversal) "
        f"and computes the solved payload for challenge '{challenge}'.\n"
        f"Your script must print 'SOLVED_PAYLOAD: <payload>'.\n"
        f"Output your Python code inside a ```python block."
    )
    
    resp_analysis = analyst.solve(analysis_prompt, model_name=HIGH_DENSITY_MODEL, temp=0.3)
    script_analysis = extract_python_code(resp_analysis["text"])
    
    success, stdout_analysis, stderr_analysis = run_generated_script(script_analysis, "scratch/probe_c_analysis.py")
    
    payload = ""
    if success:
        print(f"[Group C] Analysis Tree successfully deduced rule:\n{stdout_analysis.strip()}")
        match = re.search(r"SOLVED_PAYLOAD:\s*(\S+)", stdout_analysis)
        if match:
            payload = match.group(1)
    else:
        print(f"[Group C Error] Analysis Tree script failed: {stderr_analysis}")
        
    analysis_tc.log_plasticity(2)
    
    # Step C: Validation Tree verifies final payload
    print("\n[Group C] [Validation Tree] Spawning validator to run compliance and verify access key...")
    validator = val_tc.create_agent("validator", "Verify responses and perform final key submission.")
    val_tc.log_plasticity(1)
    
    res = api.submit(challenge, payload)
    
    val_tc.log_plasticity(2)
    duration = time.time() - start_time
    stats = client.get_stats()
    
    print(f"\n[Group C Summary] Success: {res['success']} | Key: {res.get('key', 'None')} | Tokens: {stats['total_tokens']} | Time: {duration:.2f}s")
    
    # Compile Combined Group C Plasticity
    c_log = []
    max_steps = max(len(probe_tc.plasticity_log), len(analysis_tc.plasticity_log), len(val_tc.plasticity_log))
    for i in range(max_steps):
        p_act = probe_tc.plasticity_log[i][1] if i < len(probe_tc.plasticity_log) else 0
        a_act = analysis_tc.plasticity_log[i][1] if i < len(analysis_tc.plasticity_log) else 0
        v_act = val_tc.plasticity_log[i][1] if i < len(val_tc.plasticity_log) else 0
        
        p_cr = probe_tc.plasticity_log[i][2] if i < len(probe_tc.plasticity_log) else 0
        a_cr = analysis_tc.plasticity_log[i][2] if i < len(analysis_tc.plasticity_log) else 0
        v_cr = val_tc.plasticity_log[i][2] if i < len(val_tc.plasticity_log) else 0
        
        p_ds = probe_tc.plasticity_log[i][3] if i < len(probe_tc.plasticity_log) else 0
        a_ds = analysis_tc.plasticity_log[i][3] if i < len(analysis_tc.plasticity_log) else 0
        v_ds = val_tc.plasticity_log[i][3] if i < len(val_tc.plasticity_log) else 0
        
        c_log.append((i, p_act + a_act + v_act, p_cr + a_cr + v_cr, p_ds + a_ds + v_ds))
        
    return {
        "success": res["success"],
        "key": res.get("key"),
        "tokens": stats["weighted_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration,
        "plasticity": c_log,
        "creations": probe_tc.creation_count + analysis_tc.creation_count + val_tc.creation_count,
        "destructions": probe_tc.destruction_count + analysis_tc.destruction_count + val_tc.destruction_count
    }

# =====================================================================
# Comparative Metrics Plotting
# =====================================================================
def plot_comparative_plasticity(b_log, c_log):
    if not b_log or not c_log:
        return
        
    b_steps, b_active, b_creations, b_destructions = zip(*b_log)
    c_steps, c_active, c_creations, c_destructions = zip(*c_log)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(b_steps, b_active, label="Group B (Single-Tree) Active MAs", color="blue", marker="o")
    plt.plot(c_steps, c_active, label="Group C (Forest) Combined Active MAs", color="green", marker="s")
    
    plt.plot(b_steps, b_creations, label="Group B Cumulative Creations", color="blue", linestyle=":")
    plt.plot(b_steps, b_destructions, label="Group B Cumulative Destructions (GC)", color="red", linestyle=":")
    
    plt.plot(c_steps, c_creations, label="Group C Cumulative Creations", color="green", linestyle="--")
    plt.plot(c_steps, c_destructions, label="Group C Cumulative Destructions (GC)", color="orange", linestyle="--")
    
    plt.title("Reverse Engineering Plasticity: Group B vs Group C", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Cognitive Phase (Step)", fontsize=12)
    plt.ylabel("Agent Counts", fontsize=12)
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("ontogenetic_plasticity.png", dpi=300)
    print("\n[Plot] Saved comparative plasticity graph to 'ontogenetic_plasticity.png'")

# =====================================================================
# Main Loop
# =====================================================================
def main():
    # Initialize components
    client = OllamaClient(default_model=MODEL_NAME)
    api = BlackBoxAPI()
    
    # Generate target challenge
    challenge = api.generate_challenge()
    
    # Warm up models
    print("[Warm-up] Initializing models in GPU memory...")
    client.generate(prompt="Hello", model_name="qwen2.5:0.5b")
    client.generate(prompt="Hello", model_name="qwen2.5:1.5b")
    print("[Warm-up] Done.")

    # Run tests in randomized order
    runs = [
        ("Group A", lambda: run_group_a(client, api, challenge)),
        ("Group B", lambda: run_group_b(client, api, challenge)),
        ("Group C", lambda: run_group_c(client, api, challenge))
    ]
    import random
    random.shuffle(runs)
    print(f"\n[Experimental Design] Running groups in randomized order: {[name for name, _ in runs]}")
    
    results_map = {}
    for name, run_fn in runs:
        results_map[name] = run_fn()
        
    res_a = results_map["Group A"]
    res_b = results_map["Group B"]
    res_c = results_map["Group C"]
    
    # Output metrics
    print("\n" + "="*50)
    print("BLACK-BOX API REVERSE ENGINEERING COMPARATIVE RESULTS")
    print("="*50)
    print(f"Group A (Baseline):  Success: {res_a['success']} | Key: {res_a['key']} | Tokens: {res_a['tokens']} | Time: {res_a['duration']:.2f}s")
    print(f"Group B (SingleTree): Success: {res_b['success']} | Key: {res_b['key']} | Tokens: {res_b['tokens']} | Time: {res_b['duration']:.2f}s")
    print(f"Group C (Forest):     Success: {res_c['success']} | Key: {res_c['key']} | Tokens: {res_c['tokens']} | Time: {res_c['duration']:.2f}s")
    
    # Compute Advanced Academic Metrics
    a_success_val = 100.0 if res_a["success"] else 0.0
    a_yield = (a_success_val * 1000000.0) / res_a["tokens"] if res_a["tokens"] > 0 else 0.0
    a_turnover = 0.0
    a_allostatic_load = res_a["prompt_tokens"]
    
    b_success_val = 100.0 if res_b["success"] else 0.0
    b_yield = (b_success_val * 1000000.0) / res_b["tokens"] if res_b["tokens"] > 0 else 0.0
    b_turnover = res_b["destructions"] / res_b["creations"] if res_b["creations"] > 0 else 0.0
    b_allostatic_load = res_b["prompt_tokens"]
    
    c_success_val = 100.0 if res_c["success"] else 0.0
    c_yield = (c_success_val * 1000000.0) / res_c["tokens"] if res_c["tokens"] > 0 else 0.0
    c_turnover = res_c["destructions"] / res_c["creations"] if res_c["creations"] > 0 else 0.0
    c_allostatic_load = res_c["prompt_tokens"]
    
    import csv
    blackbox_headers = [
        "Group", "Success", "Key_Extracted", "Tokens_Used", "Execution_Time_s",
        "Metabolic_Yield", "Cell_Turnover", "Allostatic_Load"
    ]
    blackbox_data = [
        ["Group A (Baseline)", 1.0 if res_a["success"] else 0.0, 1.0 if res_a["key"] else 0.0, res_a["tokens"], res_a["duration"], a_yield, a_turnover, a_allostatic_load],
        ["Group B (Single-Tree)", 1.0 if res_b["success"] else 0.0, 1.0 if res_b["key"] else 0.0, res_b["tokens"], res_b["duration"], b_yield, b_turnover, b_allostatic_load],
        ["Group C (Forest)", 1.0 if res_c["success"] else 0.0, 1.0 if res_c["key"] else 0.0, res_c["tokens"], res_c["duration"], c_yield, c_turnover, c_allostatic_load]
    ]
    with open("blackbox_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(blackbox_headers)
        writer.writerows(blackbox_data)
    print("[CSV] Saved blackbox_metrics.csv with advanced metrics.")
    
    # Plot comparison
    plot_comparative_plasticity(res_b["plasticity"], res_c["plasticity"])

if __name__ == "__main__":
    main()
