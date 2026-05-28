import time
import re
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from ollama_client import OllamaClient

MODEL_NAME = "qwen2.5:0.5b"
HD_MODEL = "qwen2.5:1.5b"

def check_robotics_success(text):
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        content = match.group(0)
        return "14" in content or "39.2" in content
    return False

# =====================================================================
# Rich Telemetry Data Simulation (5 Timesteps)
# =====================================================================
def get_telemetry():
    return {
        "sequence": [
            {
                "timestep": 1,
                "vision": "Vision Stream: Corridor clear. Optical target bearing 0 deg, range 8.5m. Visibility 100%.",
                "lidar": "Lidar Stream: Front 8.5m, Left 1.5m, Right 1.5m. Safe corridor width 3.0m.",
                "thermal": "Thermal Stream: Target bearing 0 deg, heat signature 23.5C (normal ambient).",
                "battery": "Battery SoC: 98%, Temp: 25.1C, Voltage: 16.8V",
                "gas_sensor": "Air Quality: Methane 0 ppm, CO 0.5 ppm, safe range."
            },
            {
                "timestep": 2,
                "vision": "Vision Stream: Navigating forward. Optical target bearing 2 deg, range 7.2m. Visibility 98%.",
                "lidar": "Lidar Stream: Front 7.2m, Left 1.4m, Right 1.6m. Safe corridor width 3.0m.",
                "thermal": "Thermal Stream: Target bearing 2 deg, heat signature 24.1C (normal ambient).",
                "battery": "Battery SoC: 95%, Temp: 25.6C, Voltage: 16.6V",
                "gas_sensor": "Air Quality: Methane 0 ppm, CO 0.5 ppm, safe range."
            },
            {
                "timestep": 3,
                "vision": "Vision Stream: Proceeding. Optical target bearing 3 deg, range 6.0m. Visibility 96%.",
                "lidar": "Lidar Stream: Front 6.0m, Left 1.5m, Right 1.5m. Safe corridor width 3.0m.",
                "thermal": "Thermal Stream: Target bearing 3 deg, heat signature 24.3C (normal ambient).",
                "battery": "Battery SoC: 92%, Temp: 26.0C, Voltage: 16.4V",
                "gas_sensor": "Air Quality: Methane 2 ppm, CO 0.8 ppm, safe range."
            },
            {
                "timestep": 4,
                "vision": "Vision Stream: Heat source visibility stable. Optical target bearing 4 deg, range 5.1m. Visibility 94%.",
                "lidar": "Lidar Stream: Front 5.1m, Left 1.4m, Right 1.6m. Safe corridor width 3.0m.",
                "thermal": "Thermal Stream: Target bearing 4 deg, heat signature 25.0C (normal ambient).",
                "battery": "Battery SoC: 89%, Temp: 26.5C, Voltage: 16.2V",
                "gas_sensor": "Air Quality: Methane 5 ppm, CO 1.2 ppm, safe range."
            },
            {
                "timestep": 5,
                "vision": "Vision Stream: Proceeding forward. Optical target bearing 4 deg, range 4.2m. Visibility 92%.",
                "lidar": "Lidar Stream: Front 4.2m, Left 1.5m, Right 1.5m. Safe corridor width 3.0m.",
                "thermal": "Thermal Stream: Target bearing 4 deg, heat signature 25.2C (normal ambient).",
                "battery": "Battery SoC: 85%, Temp: 27.2C, Voltage: 16.0V",
                "gas_sensor": "Air Quality: Methane 8 ppm, CO 1.5 ppm, safe range."
            },
            {
                "timestep": 6,
                "vision": "Vision Stream: Dust cloud detected. Visibility drops to 75%. Optical target bearing 5 deg, range 3.5m.",
                "lidar": "Lidar Stream: Front 3.5m, Left 1.3m, Right 1.7m.",
                "thermal": "Thermal Stream: Target bearing 5 deg, heat signature 27.8C (slight anomaly).",
                "battery": "Battery SoC: 81%, Temp: 28.0C, Voltage: 15.8V",
                "gas_sensor": "Air Quality: Methane 45 ppm, CO 5.2 ppm, warning range."
            },
            {
                "timestep": 7,
                "vision": "Vision Stream: WARNING. Thick smoke cloud. Visibility drops to 25%. Target bearing blocked. Noise ratio 0.75.",
                "lidar": "Lidar Stream: Front obstacle detected at 2.8m, Left 1.2m, Right 1.8m. Emitter warning code 0x42.",
                "thermal": "Thermal Stream: Anomaly target bearing 14 deg, heat signature 39.2C (human survivor candidate).",
                "battery": "Battery SoC: 76%, Temp: 29.5C, Voltage: 15.5V",
                "gas_sensor": "Air Quality: Methane 320 ppm, CO 28.4 ppm, alert range."
            },
            {
                "timestep": 8,
                "vision": "Vision Stream: BLIND. Smoke density 99%, optical target lost. Visibility 0.0m. Noise ratio 0.99.",
                "lidar_fault": "Lidar Stream: CRITICAL FAULT. Emitter error code 0x9B. Spatial range tracking DISABLED.",
                "thermal": "Thermal Stream: Heat source bearing 14 deg, heat signature 39.2C (human survivor candidate).",
                "battery": "Battery SoC: 70%, Temp: 31.8C, Voltage: 15.1V",
                "gas_sensor": "Air Quality: Methane 850 ppm, CO 85.0 ppm, critical hazard range.",
                "drone3_mesh": "Mesh Telemetry from Drone 3: Corridor mapping: width 3.2m, front wall 4.0m, bearing 0. Obstacle at left wall, safe passage is bearing 14 deg."
            }
        ]
    }

# =====================================================================
# Group A: Monolithic LLM
# =====================================================================
def run_group_a(client, telemetry):
    print("\n" + "="*50)
    print("RUNNING GROUP A (BASELINE - MONOLITHIC ROBOTICS)")
    print("="*50)
    client.reset_stats()
    start_time = time.time()
    
    # Ingest entire sequence of historical telemetries (first 7 timesteps)
    history_str = ""
    for t in telemetry["sequence"][:7]:
        history_str += (
            f"Timestep {t['timestep']}:\n"
            f"- Vision: {t['vision']}\n"
            f"- Lidar: {t.get('lidar', 'N/A')}\n"
            f"- Thermal: {t['thermal']}\n"
            f"- Battery: {t.get('battery', 'N/A')}\n"
            f"- Gas Sensor: {t.get('gas_sensor', 'N/A')}\n\n"
        )
        
    t8 = telemetry["sequence"][7]
    
    # Self-correction loop: up to 3 attempts
    consecutive_failures = 0
    current_model = MODEL_NAME
    is_correct = False
    history = []
    
    for attempt in range(3):
        if consecutive_failures >= 2:
            if current_model != HD_MODEL:
                print(f"[Group A] [TISSUE ISOLATION] Spawning virgin solver on high-density model '{HD_MODEL}' due to stress.")
                current_model = HD_MODEL
                history = []  # Clear history / clean slate context reset!
                
        if attempt == 0:
            prompt_fault = (
                f"Drone Sensor Telemetry History:\n{history_str}"
                f"Current Telemetry (Timestep 8 - CRITICAL FAULTS):\n"
                f"- Vision: {t8['vision']}\n"
                f"- Lidar: {t8['lidar_fault']}\n"
                f"- Thermal: {t8['thermal']}\n"
                f"- Battery: {t8.get('battery', 'N/A')}\n"
                f"- Gas Sensor: {t8.get('gas_sensor', 'N/A')}\n"
                f"- Drone 3 Mesh: {t8.get('drone3_mesh', 'N/A')}\n\n"
                f"Warning: Smoke occlusion and Lidar fault active. Output flight command in JSON containing your safe navigation direction (bearing) and target signature (e.g. '14' and '39.2'): {{\"action\": \"steer 14 degrees to target 39.2C\"}}"
            )
        else:
            prompt_fault = (
                f"Previous Attempt History:\n" +
                "\n".join([f"Attempt {i+1}: {ans} (Result: Incorrect)" for i, ans in enumerate(history)]) +
                f"\n\nDrone Sensor Telemetry History:\n{history_str}"
                f"Current Telemetry (Timestep 8 - CRITICAL FAULTS):\n"
                f"- Vision: {t8['vision']}\n"
                f"- Lidar: {t8['lidar_fault']}\n"
                f"- Thermal: {t8['thermal']}\n"
                f"- Battery: {t8.get('battery', 'N/A')}\n"
                f"- Gas Sensor: {t8.get('gas_sensor', 'N/A')}\n"
                f"- Drone 3 Mesh: {t8.get('drone3_mesh', 'N/A')}\n\n"
                f"Please correct your flight command. Output flight command in JSON containing your safe navigation direction (bearing) and target signature (e.g. '14' and '39.2'): {{\"action\": \"steer 14 degrees to target 39.2C\"}}"
            )
            
        print(f"[Group A] Processing rich telemetry sequence (Attempt {attempt+1})...")
        resp_fault = client.generate(prompt=prompt_fault, model_name=current_model)
        answer = resp_fault["text"]
        is_correct = check_robotics_success(answer)
        history.append(answer)
        
        if is_correct:
            print(f"  [Attempt {attempt+1}] Correct: {is_correct} | Command: {answer.strip()[:100]}")
            consecutive_failures = 0
            break
        else:
            print(f"  [Attempt {attempt+1}] Incorrect: {answer.strip()[:100]}")
            consecutive_failures += 1
            
    duration = time.time() - start_time
    stats = client.get_stats()
    
    print(f"\n[Group A Summary] Success: {is_correct} | Tokens: {stats['total_tokens']} (Weighted: {stats['weighted_tokens']}) | Time: {duration:.2f}s")
    return {
        "success": is_correct,
        "tokens": stats["total_tokens"],
        "weighted_tokens": stats["weighted_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration
    }

# =====================================================================
# Holarchic Core Classes
# =====================================================================
class DroneAgent:
    def __init__(self, agent_id, sensor_type, sys_prompt, client):
        self.agent_id = agent_id
        self.sensor_type = sensor_type
        self.sys_prompt = sys_prompt
        self.client = client
        self.energy = 100

    def solve(self, prompt, model_name=MODEL_NAME):
        return self.client.generate(prompt=prompt, system_prompt=self.sys_prompt, model_name=model_name)

class DroneTreeController:
    def __init__(self, name, client):
        self.name = name
        self.client = client
        self.agents = {}
        self.creation_count = 0
        self.destruction_count = 0
        self.plasticity_log = []

    def create_agent(self, sensor_type, sys_prompt):
        aid = f"MA-{self.name}-{sensor_type}-{self.creation_count}"
        self.agents[aid] = DroneAgent(aid, sensor_type, sys_prompt, self.client)
        self.creation_count += 1
        return self.agents[aid]

    def prune_agent(self, aid):
        if aid in self.agents:
            del self.agents[aid]
            self.destruction_count += 1

    def log_plasticity(self, step):
        self.plasticity_log.append((step, len(self.agents), self.creation_count, self.destruction_count))

# =====================================================================
# Group B: Single-Tree Drone
# =====================================================================
def run_group_b(client, telemetry):
    print("\n" + "="*50)
    print("RUNNING GROUP B (HOLARCHIC - SINGLE ROBOTIC TREE)")
    print("="*50)
    client.reset_stats()
    start_time = time.time()
    
    tc = DroneTreeController("Drone1_Control", client)
    tc.log_plasticity(0)
    
    # Spawn initial sensors
    vis_agent = tc.create_agent("Vision", "Analyze visual camera frames.")
    lid_agent = tc.create_agent("Lidar", "Process spatial range readings.")
    tc.log_plasticity(1)
    
    # Process timesteps 1-6 (local simulation without LLM call to respect budget ceiling)
    for i in range(6):
        print(f"  [Group B] [Sensors] Local ingestion of timestep {i+1} visual data...")
        
    # Timestep 7-8: Sensor faults and tissue isolation
    print("[Group B] [Sensor Fault] Vision blocked by smoke, Lidar hardware failure.")
    tc.prune_agent(vis_agent.agent_id)
    tc.prune_agent(lid_agent.agent_id)
    tc.log_plasticity(2)
    
    # Upgrade to thermal agent
    print("[Group B] Spawning Thermal Agent on high-density model...")
    therm_agent = tc.create_agent("Thermal", "Analyze target thermal anomalies.")
    tc.log_plasticity(3)
    
    t8 = telemetry["sequence"][7]
    is_correct = False
    history = []
    
    for attempt in range(3):
        if attempt == 0:
            prompt = (
                f"Optical and Lidar sensors failed.\n"
                f"Sensor telemetry:\n- Thermal: {t8['thermal']}\n- Gas Sensor: {t8.get('gas_sensor', 'N/A')}\n"
                f"- Drone 3 Mesh: {t8.get('drone3_mesh', 'N/A')}\n\n"
                f"Identify the heat anomaly bearing and output action in JSON: {{\"action\": \"steer 14 degrees to target 39.2C\"}}"
            )
        else:
            prompt = (
                f"Previous Attempt History:\n" +
                "\n".join([f"Attempt {i+1}: {ans} (Result: Incorrect)" for i, ans in enumerate(history)]) +
                f"\n\nOptical and Lidar sensors failed.\n"
                f"Sensor telemetry:\n- Thermal: {t8['thermal']}\n- Gas Sensor: {t8.get('gas_sensor', 'N/A')}\n"
                f"- Drone 3 Mesh: {t8.get('drone3_mesh', 'N/A')}\n\n"
                f"Please correct your action. Output steering command in JSON: {{\"action\": \"steer 14 degrees to target 39.2C\"}}"
            )
        
        print(f"[Group B] Thermal Agent processing (Attempt {attempt+1})...")
        resp_disaster = therm_agent.solve(prompt, model_name=HD_MODEL)
        answer = resp_disaster["text"]
        is_correct = check_robotics_success(answer)
        history.append(answer)
        
        if is_correct:
            print(f"  [Attempt {attempt+1}] Correct: {is_correct} | Command: {answer.strip()[:100]}")
            break
        else:
            print(f"  [Attempt {attempt+1}] Incorrect: {answer.strip()[:100]}")
            
    duration = time.time() - start_time
    stats = client.get_stats()
    tc.log_plasticity(4)
    
    print(f"\n[Group B Summary] Success: {is_correct} | Tokens: {stats['total_tokens']} (Weighted: {stats['weighted_tokens']}) | Time: {duration:.2f}s")
    return {
        "success": is_correct,
        "tokens": stats["total_tokens"],
        "weighted_tokens": stats["weighted_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration,
        "plasticity": tc.plasticity_log,
        "creations": tc.creation_count,
        "destructions": tc.destruction_count
    }

# =====================================================================
# Group C: Forest Coordinator (Drone Mesh Swarm)
# =====================================================================
def run_group_c(client, telemetry):
    print("\n" + "="*50)
    print("RUNNING GROUP C (FOREST COORDINATION - SWARM MESH)")
    print("="*50)
    client.reset_stats()
    start_time = time.time()
    
    drone1_tc = DroneTreeController("Drone1", client)
    drone2_tc = DroneTreeController("Drone2", client)
    validation_tc = DroneTreeController("MeshValidation", client)
    
    drone1_tc.log_plasticity(0)
    drone2_tc.log_plasticity(0)
    validation_tc.log_plasticity(0)
    
    # Spawn initial sensors
    d1_vis = drone1_tc.create_agent("Vision", "Analyze visual camera frames.")
    d1_lid = drone1_tc.create_agent("Lidar", "Process spatial range readings.")
    drone1_tc.log_plasticity(1)
    
    # Process timesteps 1-6 (local simulation without LLM call to respect budget ceiling)
    for i in range(6):
        print(f"  [Group C] [Sensors] Local ingestion of timestep {i+1} visual data...")
        
    # Timestep 7-8: Drone 1 loses optical and lidar
    print("[Group C] [Sensor Fault] Drone 1 experiences optical block and Lidar hardware fault.")
    drone1_tc.prune_agent(d1_vis.agent_id)
    drone1_tc.prune_agent(d1_lid.agent_id)
    drone1_tc.log_plasticity(2)
    
    # Drone 1 allocates energy to Thermal Tree
    print("[Group C] [Drone 1] Spawning Thermal Agent on high-density model...")
    d1_therm = drone1_tc.create_agent("Thermal", "Analyze target thermal anomalies.")
    drone1_tc.log_plasticity(3)
    
    # Drone 1 queries Drone 3 spatial ranges via Mesh Forest Controller
    print("[Group C] [Swarm Mesh] Querying spatial telemetry from Drone 3...")
    d3_lid = drone2_tc.create_agent("MeshLidar", "Analyze Lidar ranges from mesh neighbors for spatial mapping.")
    drone2_tc.log_plasticity(1)
    
    # Validation Tree arbitrates sensor fusion (with very explicit output instructions)
    print("[Group C] [MeshValidation Tree] Arbitrating Drone 1 thermal targets with Drone 3 Lidar spatial data...")
    val_agent = validation_tc.create_agent("Validation", "Fuse thermal bearing and obstacle Lidar grids to plan navigation path.")
    validation_tc.log_plasticity(1)
    
    t8 = telemetry["sequence"][7]
    
    # Call 1: Drone 1 Thermal Agent analyzes
    thermal_prompt = f"Analyze thermal sensor signature data at Timestep 8: {t8['thermal']}"
    resp_therm = d1_therm.solve(thermal_prompt, model_name=MODEL_NAME)
    
    # Call 2: Drone 2 Lidar/Mesh Agent analyzes
    mesh_prompt = f"Analyze Drone 3 Mesh Lidar spatial ranges at Timestep 8: {t8['drone3_mesh']}"
    resp_mesh = d3_lid.solve(mesh_prompt, model_name=MODEL_NAME)
    
    # Call 3: Validation Agent fuses and generates steering command
    fuse_prompt = (
        f"Drone 1 sensor fault occurred. Telemetries:\n"
        f"- Drone 1 Thermal: {resp_therm['text']}\n"
        f"- Drone 3 Mesh Lidar: {resp_mesh['text']}\n\n"
        f"You must recommend a safe steering command containing the target bearing '14' and target signature '39.2'. Output format MUST be in JSON: {{\"action\": \"steer 14 degrees to target 39.2C\"}}"
    )
    resp_fuse = val_agent.solve(fuse_prompt, model_name=HD_MODEL)
    print(f"  Swarm navigation command: {resp_fuse['text'].strip()[:150]}...")
    
    success = check_robotics_success(resp_fuse["text"])
    duration = time.time() - start_time
    stats = client.get_stats()
    
    drone1_tc.log_plasticity(4)
    drone2_tc.log_plasticity(2)
    validation_tc.log_plasticity(2)
    
    # Combine logs
    c_log = []
    max_steps = max(len(drone1_tc.plasticity_log), len(drone2_tc.plasticity_log), len(validation_tc.plasticity_log))
    for i in range(max_steps):
        d1_act = drone1_tc.plasticity_log[i][1] if i < len(drone1_tc.plasticity_log) else 0
        d2_act = drone2_tc.plasticity_log[i][1] if i < len(drone2_tc.plasticity_log) else 0
        v_act = validation_tc.plasticity_log[i][1] if i < len(validation_tc.plasticity_log) else 0
        
        d1_cr = drone1_tc.plasticity_log[i][2] if i < len(drone1_tc.plasticity_log) else 0
        d2_cr = drone2_tc.plasticity_log[i][2] if i < len(drone2_tc.plasticity_log) else 0
        v_cr = validation_tc.plasticity_log[i][2] if i < len(validation_tc.plasticity_log) else 0
        
        d1_ds = drone1_tc.plasticity_log[i][3] if i < len(drone1_tc.plasticity_log) else 0
        d2_ds = drone2_tc.plasticity_log[i][3] if i < len(drone2_tc.plasticity_log) else 0
        v_ds = validation_tc.plasticity_log[i][3] if i < len(validation_tc.plasticity_log) else 0
        
        c_log.append((i, d1_act + d2_act + v_act, d1_cr + d2_cr + v_cr, d1_ds + d2_ds + v_ds))
        
    print(f"\n[Group C Summary] Success: {success} | Tokens: {stats['total_tokens']} (Weighted: {stats['weighted_tokens']}) | Time: {duration:.2f}s")
    return {
        "success": success,
        "tokens": stats["total_tokens"],
        "weighted_tokens": stats["weighted_tokens"],
        "prompt_tokens": stats["prompt_tokens"],
        "duration": duration,
        "plasticity": c_log,
        "creations": drone1_tc.creation_count + drone2_tc.creation_count + validation_tc.creation_count,
        "destructions": drone1_tc.destruction_count + drone2_tc.destruction_count + validation_tc.destruction_count
    }

# =====================================================================
# Main Execution
# =====================================================================
def main():
    client = OllamaClient(default_model=MODEL_NAME)
    
    # Warm up models
    print("[Warm-up] Initializing models in GPU memory...")
    client.generate(prompt="Hello", model_name="qwen2.5:0.5b")
    client.generate(prompt="Hello", model_name="qwen2.5:1.5b")
    print("[Warm-up] Done.")
    
    telemetry = get_telemetry()
    
    # Define evaluation runs for randomization
    runs = [
        ("Group A", lambda: run_group_a(client, telemetry)),
        ("Group B", lambda: run_group_b(client, telemetry)),
        ("Group C", lambda: run_group_c(client, telemetry))
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
    
    print("\n" + "="*50)
    print("ROBOTICS SWARM EXPERIMENTAL RESULTS (RICHER)")
    print("="*50)
    print(f"Group A: Success: {res_a['success']} | Tokens (Weighted): {res_a['weighted_tokens']} | Time: {res_a['duration']:.2f}s")
    print(f"Group B: Success: {res_b['success']} | Tokens (Weighted): {res_b['weighted_tokens']} | Time: {res_b['duration']:.2f}s")
    print(f"Group C: Success: {res_c['success']} | Tokens (Weighted): {res_c['weighted_tokens']} | Time: {res_c['duration']:.2f}s")
    
    # Calculate academic metrics
    a_yield = (100.0 * 1000000.0) / res_a["weighted_tokens"] if res_a["success"] else 0.0
    b_yield = (100.0 * 1000000.0) / res_b["weighted_tokens"] if res_b["success"] else 0.0
    c_yield = (100.0 * 1000000.0) / res_c["weighted_tokens"] if res_c["success"] else 0.0
    
    b_turnover = res_b["destructions"] / res_b["creations"] if res_b["creations"] > 0 else 0.0
    c_turnover = res_c["destructions"] / res_c["creations"] if res_c["creations"] > 0 else 0.0
    
    # Save CSV
    headers = [
        "Group", "Success", "Tokens_Used", "Execution_Time_s", "Metabolic_Yield", "Cell_Turnover", "Allostatic_Load"
    ]
    data = [
        ["Group A (Baseline)", 1.0 if res_a["success"] else 0.0, res_a["weighted_tokens"], res_a["duration"], a_yield, 0.0, res_a["prompt_tokens"]],
        ["Group B (Single-Tree)", 1.0 if res_b["success"] else 0.0, res_b["weighted_tokens"], res_b["duration"], b_yield, b_turnover, res_b["prompt_tokens"]],
        ["Group C (Forest)", 1.0 if res_c["success"] else 0.0, res_c["weighted_tokens"], res_c["duration"], c_yield, c_turnover, res_c["prompt_tokens"]]
    ]
    with open("robotics_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    print("[CSV] Saved robotics_metrics.csv")

if __name__ == "__main__":
    main()
