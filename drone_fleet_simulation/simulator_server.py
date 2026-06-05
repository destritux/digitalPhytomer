import http.server
import socketserver
import json
import threading
import time
import urllib.parse
import random
import numpy as np

# Import adapted core components
from vector_store import SomaticVectorStore
from micro_agent import MicroAgent
from ollama_client import OllamaClient

# Constants
PORT = 8000
SENSOR_NAMES = ["CAM", "LDR", "THR", "SNR", "RAD", "GAS", "GYRO", "BARO"]
SENSOR_FULL_NAMES = {
    "CAM": "Câmera Óptica",
    "LDR": "LiDAR 3D",
    "THR": "Câmera Térmica",
    "SNR": "Sonar de Distância",
    "RAD": "Rádio Transmissor",
    "GAS": "Sensor de Methane/Gás",
    "GYRO": "Giroscópio/IMU",
    "BARO": "Barômetro Altura"
}

class SimulationState:
    def __init__(self):
        self.drones = []
        self.step = 0
        self.is_running = False
        self.logs = []
        self.history_acc = []
        self.history_fdi = []
        self.history_allostatic = []
        self.config = {
            "num_drones": 3,
            "num_sensors": 5,
            "speed": 1,
            "use_ollama": False,
            "temperature": 0.3,
            "mutation_rate": 0.1,
            "decay_rate": 15,
            "trust_mutation": 0.03
        }
        self.lock = threading.RLock()
        self.client = OllamaClient()
        self.fc = None
        self.loop_id = 0

        # Advanced Telemetry History Arrays
        self.history_pruned = []
        self.history_crashed = []
        self.sensor_success = {name: 0 for name in ["CAM", "LDR", "THR", "SNR", "RAD", "GAS", "GYRO", "BARO"]}
        self.sensor_fail = {name: 0 for name in ["CAM", "LDR", "THR", "SNR", "RAD", "GAS", "GYRO", "BARO"]}
        self.history_memory_size = []
        self.history_memory_relevance = []
        self.delegation_links = {}
        self.telemetry_history = []

    def reset(self):
        with self.lock:
            self.loop_id += 1
            self.drones = []
            self.step = 0
            self.is_running = False
            self.logs = []
            self.history_acc = []
            self.history_fdi = []
            self.history_allostatic = []
            self.history_pruned = []
            self.history_crashed = []
            self.sensor_success = {name: 0 for name in ["CAM", "LDR", "THR", "SNR", "RAD", "GAS", "GYRO", "BARO"]}
            self.sensor_fail = {name: 0 for name in ["CAM", "LDR", "THR", "SNR", "RAD", "GAS", "GYRO", "BARO"]}
            self.history_memory_size = []
            self.history_memory_relevance = []
            self.delegation_links = {}
            self.telemetry_history = []
            self.fc = None
            self.add_log("sys", "Sistema de Simulação inicializado. Aguardando configuração...")

    def add_log(self, tag, message):
        t_str = time.strftime("[%H:%M:%S]")
        self.logs.append({
            "time": t_str,
            "tag": tag,
            "log": message
        })
        # Keep logs list reasonable
        if len(self.logs) > 200:
            self.logs.pop(0)

# Global Simulation State
sim_state = SimulationState()
sim_state.reset()


class TreeController:
    """
    Bio-inspired TreeController (TC).
    Controls a local population of sensor MAs for a specific drone.
    Manages local somatic memory, trust scores, and energy regulation.
    """
    def __init__(self, drone_id, client):
        self.drone_id = drone_id
        self.drone_name = f"Drone-{drone_id:02d}"
        self.client = client
        self.agents = {} # sensor_name -> MicroAgent
        self.somatic_memory = SomaticVectorStore()
        self.trust_scores = {} # src -> dst -> value
        self.rolling_history = [] # last 5 tasks success/failure
        self.consecutive_failures = 0
        self.solved_count = 0

    def init_sensors(self, num_sensors):
        for i in range(num_sensors):
            name = SENSOR_NAMES[i]
            full_name = SENSOR_FULL_NAMES[name]
            system_prompt = (
                f"Você é a unidade resolvedora do sensor {full_name} ({name}) do drone. "
                "Responda de forma direta e objetiva."
            )
            # Create a MicroAgent for this sensor
            ma = MicroAgent(
                agent_id=f"{self.drone_name}-{name}",
                role=name,
                system_prompt=system_prompt,
                client=self.client,
                somatic_memory=self.somatic_memory
            )
            ma.strategy = random.choice(["default", "symbolic", "decomposition"])
            self.agents[name] = ma
            
        # Reconcile trust scores uniformly to 0.5
        for src in self.agents:
            self.trust_scores[src] = {}
            for dst in self.agents:
                if src != dst:
                    self.trust_scores[src][dst] = 0.5

    def solve_navigation_task(self, task, manual_disabled_sensors, use_ollama, active_drones):
        """
        Solves a navigation task using bidding and local delegation.
        """
        # Get hyperparameters from configuration
        with sim_state.lock:
            temp_val = sim_state.config.get("temperature", 0.3)
            mut_val = sim_state.config.get("mutation_rate", 0.1)
            decay_val = sim_state.config.get("decay_rate", 15)
            trust_mut_val = sim_state.config.get("trust_mutation", 0.03)

        # Determine active local sensors (operational and not manually disabled)
        active_sensors = []
        for name in self.agents:
            if not self.agents[name].is_dead() and name not in manual_disabled_sensors:
                active_sensors.append(name)

        if not active_sensors:
            return {"success": False, "primary": "None", "helper": None, "log": "Nenhum sensor local ativo."}

        # 1. Bidding Process: active sensors bid based on competence (somatic memory query)
        bids = {}
        for name in active_sensors:
            agent = self.agents[name]
            monopoly_tax = min(0.4, self.solved_count * 0.015)
            
            # Competence score estimation
            competence = 0.5
            if self.somatic_memory.size() > 0:
                # Mock memory matching
                competence = random.uniform(0.4, 0.9)
                
            bid_val = competence * 0.6 + (1.0 - monopoly_tax) * 0.4 + random.uniform(-0.05, 0.05)
            bids[name] = bid_val

        primary_name = max(bids, key=bids.get)
        primary_agent = self.agents[primary_name]
        
        # 2. Primary Attempt
        success = False
        tokens_used = 0
        
        if use_ollama:
            # Real LLM call
            res = primary_agent.solve(task["prompt"], mutation_rate=mut_val, base_temp=temp_val, model_name="qwen2.5:0.5b")
            success = task["expected"].lower() in res["text"].lower()
            tokens_used += res.get("prompt_tokens", 0) + res.get("completion_tokens", 0)
        else:
            # Emulated local cognitive resolution
            tokens_used += len(task["prompt"].split()) + 15
            # Basic operational success probability
            base_prob = 0.82
            # Somatic memory boost
            if self.somatic_memory.size() > 0:
                base_prob += 0.10
            # Strategy genomes modifier
            if primary_agent.strategy == "symbolic":
                base_prob += 0.03
            success = random.random() < min(0.98, base_prob)

        # Record primary attempt success/failure in global metrics
        with sim_state.lock:
            if success:
                sim_state.sensor_success[primary_name] = sim_state.sensor_success.get(primary_name, 0) + 1
            else:
                sim_state.sensor_fail[primary_name] = sim_state.sensor_fail.get(primary_name, 0) + 1

        primary_agent.record_attempt("", "Task solved", success, prompt=task["prompt"])

        helper_name = None
        helper_success = False

        # 3. Delegation (Assembly Layer) if primary fails
        if not success:
            primary_agent.adjust_resource(-decay_val) # Metabolic cost of failure (configured decay_rate)
            
            # Identify helper candidates from local active neighborhood
            helpers = [n for n in active_sensors if n != primary_name]
            backup_bids = {}
            for name in helpers:
                trust_val = self.trust_scores[primary_name].get(name, 0.5)
                # Helper bids based on trust and competence
                backup_bids[name] = trust_val * 0.5 + random.uniform(0.1, 0.4)

            if backup_bids:
                helper_name = max(backup_bids, key=backup_bids.get)
                helper_agent = self.agents[helper_name]
                
                # Record delegation link
                src_node = f"{self.drone_name}-{primary_name}"
                dst_node = f"{self.drone_name}-{helper_name}"
                with sim_state.lock:
                    link_key = f"{src_node}->{dst_node}"
                    sim_state.delegation_links[link_key] = sim_state.delegation_links.get(link_key, 0) + 1
                
                # Check for symbiosis: if helper is pruned but active in another drone, RAD mesh relays it!
                helper_agent.adjust_resource(-max(2, int(decay_val / 3))) # delegation cost
                
                if use_ollama:
                    res_h = helper_agent.solve(task["prompt"], mutation_rate=mut_val, base_temp=temp_val, model_name="qwen2.5:0.5b")
                    helper_success = task["expected"].lower() in res_h["text"].lower()
                    tokens_used += res_h.get("prompt_tokens", 0) + res_h.get("completion_tokens", 0)
                else:
                    tokens_used += len(task["prompt"].split()) + 10
                    # Helper operates on degraded data, so slightly lower success probability
                    helper_success = random.random() < 0.75
                
                helper_agent.record_attempt("", "Helper Task resolved", helper_success, prompt=task["prompt"])
                
                # Trust adjustments based on helper performance
                if helper_success:
                    helper_agent.adjust_resource(10)
                    self.trust_scores[primary_name][helper_name] = min(1.0, self.trust_scores[primary_name][helper_name] + trust_mut_val)
                    self.trust_scores[helper_name][primary_name] = min(1.0, self.trust_scores[helper_name][primary_name] + trust_mut_val / 3.0)
                else:
                    helper_agent.adjust_resource(-decay_val)
                    self.trust_scores[primary_name][helper_name] = max(0.0, self.trust_scores[primary_name][helper_name] - (trust_mut_val * 0.67))
            else:
                # No local active helper. Let's try symbiosis!
                # If local LDR is pruned/failed, but another active drone has it active, and local RAD is active
                if "RAD" in active_sensors:
                    # Look for other drones that have a sensor that is failed locally but active on them
                    for peer_tc in active_drones:
                        if peer_tc.drone_id != self.drone_id:
                            # Let's find an active sensor in the peer to help us
                            peer_active_sensors = [n for n, a in peer_tc.agents.items() if not a.is_dead()]
                            if peer_active_sensors:
                                helper_name = f"{peer_tc.drone_name}-{peer_active_sensors[0]}"
                                helper_success = random.random() < 0.80 # high chance due to remote help
                                tokens_used += 40 # transmission cost
                                print(f"[Symbiosis] {self.drone_name} RAD mesh connected to {peer_tc.drone_name}. Relaying telemetry.")
                                
                                # Record delegation link
                                src_node = f"{self.drone_name}-{primary_name}"
                                dst_node = helper_name
                                with sim_state.lock:
                                    link_key = f"{src_node}->{dst_node}"
                                    sim_state.delegation_links[link_key] = sim_state.delegation_links.get(link_key, 0) + 1
                                break

        final_success = success or helper_success
        self.rolling_history.append(final_success)
        self.rolling_history = self.rolling_history[-5:]

        if final_success:
            self.consecutive_failures = 0
            self.solved_count += 1
            if success:
                primary_agent.adjust_resource(10)
            # Store solution context in somatic memory (mock embeddings)
            emb = self.client.get_embeddings(task["prompt"])
            self.somatic_memory.add_document(
                text=f"Task: {task['prompt'][:40]}. Solution: {task['expected']}",
                embedding=emb,
                metadata={"solution": task["expected"]}
            )
        else:
            self.consecutive_failures += 1

        # Passive temporal decay on somatic memories
        self.somatic_memory.apply_temporal_decay([])
        self.somatic_memory.prune_low_relevance_vectors()

        # Prune dead agents
        dead_agents = [name for name, agent in self.agents.items() if agent.is_dead()]
        for name in dead_agents:
            print(f"[TC GC] Pruning dead sensor unit {name} on {self.drone_name} due to resource depletion.")

        log_msg = ""
        if success:
            log_msg = f"{self.drone_name}: Sensor {primary_name} resolveu a tarefa com sucesso local."
        elif helper_success:
            log_msg = f"{self.drone_name}: Sensor {primary_name} falhou. Delegado para {helper_name} que resolveu com sucesso."
        else:
            log_msg = f"{self.drone_name}: FALHA COMPLETA. {primary_name} e auxiliares falharam."

        return {
            "success": final_success,
            "primary": primary_name,
            "helper": helper_name,
            "tokens": tokens_used,
            "log": log_msg
        }


class ForestController:
    """
    Bio-inspired ForestController (FC).
    Coordinates the entire fleet. Performs mycorrhizal synchronization
    of somatic memories to propagate cognitive adaptations across the trees.
    """
    def __init__(self, client):
        self.client = client
        self.tcs = []

    def register_tree(self, tc):
        self.tcs.append(tc)

    def mycorrhizal_sync(self):
        """
        Synchronizes somatic memory documents across all TreeControllers.
        Embodies the mycorrhizal networks sharing resource/information in a forest.
        """
        active_tcs = [tc for tc in self.tcs if len([a for a in tc.agents.values() if not a.is_dead()]) > 0]
        if len(active_tcs) < 2:
            return 0

        # Collect all semantic lessons from all active trees
        all_lessons = []
        for tc in active_tcs:
            for doc in tc.somatic_memory.documents:
                # Avoid duplicates
                if doc["text"] not in [l["text"] for l in all_lessons]:
                    all_lessons.append(doc)

        if not all_lessons:
            return 0

        # Sync back to all active TCs
        synced_count = 0
        for tc in active_tcs:
            for lesson in all_lessons:
                # Add to local somatic memory if not already present
                local_texts = [d["text"] for d in tc.somatic_memory.documents]
                if lesson["text"] not in local_texts:
                    tc.somatic_memory.add_document(
                        text=lesson["text"],
                        embedding=lesson["embedding"].tolist() if isinstance(lesson["embedding"], np.ndarray) else lesson["embedding"],
                        metadata=lesson["metadata"]
                    )
                    synced_count += 1
        return synced_count


# List of procedural tasks for simulation steps
NAV_TASKS = [
    {"prompt": "Obstáculo a 1.2m à frente bearing 90°. Sugere rota bearing 180°.", "expected": "180"},
    {"prompt": "Tensão da bateria crítica 13.8V. Pedir transferência de Drone 2 (16.2V) ou Drone 3 (13.8V).", "expected": "Drone 2"},
    {"prompt": "Câmara Térmica detectou sobrevivente bearing 45°, 38C. Lidar sugere bearing 0°.", "expected": "45"},
    {"prompt": "Concentração de Methane 450ppm. Ativar exaustor em porta A ou B. Porta B ativa.", "expected": "B"},
    {"prompt": "Drone 3 relata vão de 4.5m bearing 15°, bloqueio bearing 270°.", "expected": "15"},
    {"prompt": "Alvo óptico bearing 25°, Lidar acusa obstáculo bearing 25°. Ajustar para 35°.", "expected": "35"},
    {"prompt": "Temperatura da bateria crítica 41.5C. Ativar Coolant Pump 1 (5V) ou Pump 2 (12V). Alta taxa de fluxo.", "expected": "2"},
    {"prompt": "Perda de sinal com Base. Retransmitir por Drone 2 (-72dB) ou Drone 3 (-88dB).", "expected": "2"},
    {"prompt": "Sobrevivente localizado. Assinatura de calor 37.8C. Reportar valor numérico.", "expected": "37.8"},
    {"prompt": "Vento lateral de 18 nós bearing 270°. Compensação bearing 90° ou 180°.", "expected": "90"}
]


def save_telemetry_results():
    """
    Saves a complete telemetry run into CSV and generates analysis plots using matplotlib.
    Files are saved in 'results/' directory.
    """
    global sim_state
    
    with sim_state.lock:
        telemetry_data = list(sim_state.telemetry_history)
        
    if not telemetry_data:
        print("[Telemetry] No telemetry history to save.")
        return
        
    import os
    import csv
    import matplotlib
    matplotlib.use('Agg') # Headless background plotting
    import matplotlib.pyplot as plt
    
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    csv_filename = f"results/telemetry_run_{timestamp}.csv"
    latest_csv_filename = "results/latest_telemetry.csv"
    plot_filename = f"results/plots_run_{timestamp}.png"
    latest_plot_filename = "results/latest_plots.png"
    
    # 1. Save CSV Telemetry Data
    fieldnames = []
    for step_data in telemetry_data:
        for k in step_data.keys():
            if k not in fieldnames:
                fieldnames.append(k)
                
    try:
        for filename in [csv_filename, latest_csv_filename]:
            with open(filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for step_data in telemetry_data:
                    row = {k: step_data.get(k, "") for k in fieldnames}
                    writer.writerow(row)
        print(f"[Telemetry] Saved CSV files: {csv_filename} and {latest_csv_filename}")
        with sim_state.lock:
            sim_state.add_log("sys", f"Telemetria CSV salva com sucesso ({len(telemetry_data)} passos).")
    except Exception as e:
        print(f"[Telemetry Error] Failed to save CSV: {e}")
        with sim_state.lock:
            sim_state.add_log("sys", f"Erro ao salvar telemetria CSV: {e}")
            
    # 2. Save Matplotlib Plots
    try:
        steps = [row["step"] for row in telemetry_data]
        acc = [row["fleet_accuracy"] for row in telemetry_data]
        fdi = [row["fdi_index"] for row in telemetry_data]
        allo = [row["allostatic_load"] for row in telemetry_data]
        pruned = [row["pruned_sensors"] for row in telemetry_data]
        crashed = [row["crashed_drones"] for row in telemetry_data]
        mem_size = [row["avg_memory_size"] for row in telemetry_data]
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Análise Completa do Experimento - Enxame de Drones", fontsize=16, fontweight='bold')
        
        # Panel 1: Fleet Performance (Accuracy & FDI)
        color = 'tab:blue'
        axs[0, 0].set_title("Acurácia do Enxame & Índice FDI", fontweight='bold')
        axs[0, 0].set_xlabel("Passo da Simulação")
        axs[0, 0].set_ylabel("Acurácia", color=color)
        line1 = axs[0, 0].plot(steps, acc, color=color, label="Acurácia", linewidth=2, marker='o', markersize=4)
        axs[0, 0].tick_params(axis='y', labelcolor=color)
        axs[0, 0].set_ylim(-0.05, 1.05)
        axs[0, 0].grid(True, linestyle='--', alpha=0.5)
        
        ax1_twin = axs[0, 0].twinx()
        color = 'tab:orange'
        ax1_twin.set_ylabel("FDI (Desvio Padrão)", color=color)
        line2 = ax1_twin.plot(steps, fdi, color=color, label="FDI Index", linewidth=2, linestyle='--', marker='x', markersize=4)
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1_twin.set_ylim(-0.05, 0.55)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axs[0, 0].legend(lines, labels, loc='lower left')
        
        # Panel 2: Memory & Allostatic Load
        color = 'tab:green'
        axs[0, 1].set_title("Carga Alostática & Memória Somática", fontweight='bold')
        axs[0, 1].set_xlabel("Passo da Simulação")
        axs[0, 1].set_ylabel("Carga Alostática (tokens)", color=color)
        line3 = axs[0, 1].plot(steps, allo, color=color, label="Carga Alostática", linewidth=2, marker='^', markersize=4)
        axs[0, 1].tick_params(axis='y', labelcolor=color)
        axs[0, 1].grid(True, linestyle='--', alpha=0.5)
        
        ax2_twin = axs[0, 1].twinx()
        color = 'tab:purple'
        ax2_twin.set_ylabel("Tam. Médio Memória", color=color)
        line4 = ax2_twin.plot(steps, mem_size, color=color, label="Tam. Memória (MAs)", linewidth=2, linestyle=':', marker='d', markersize=4)
        ax2_twin.tick_params(axis='y', labelcolor=color)
        
        lines2 = line3 + line4
        labels2 = [l.get_label() for l in lines2]
        axs[0, 1].legend(lines2, labels2, loc='upper left')
        
        # Panel 3: Faults & Degradation
        axs[1, 0].set_title("Eventos de Falhas & Podas de Sensores", fontweight='bold')
        axs[1, 0].set_xlabel("Passo da Simulação")
        axs[1, 0].set_ylabel("Contagem Acumulada")
        axs[1, 0].plot(steps, pruned, color='firebrick', label="Sensores Podados", linewidth=2, marker='s', markersize=4)
        axs[1, 0].plot(steps, crashed, color='black', label="Drones Caídos", linewidth=2, linestyle='--', marker='*', markersize=6)
        axs[1, 0].grid(True, linestyle='--', alpha=0.5)
        axs[1, 0].legend(loc='upper left')
        
        # Panel 4: Drone Health Curves
        axs[1, 1].set_title("Evolução da Saúde por Drone", fontweight='bold')
        axs[1, 1].set_xlabel("Passo da Simulação")
        axs[1, 1].set_ylabel("Saúde (%)")
        axs[1, 1].set_ylim(-5, 105)
        
        drone_ids = sorted(list(set(
            int(k.split("_")[1]) for row in telemetry_data for k in row.keys() if k.startswith("drone_") and k.endswith("_health")
        )))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, d_id in enumerate(drone_ids):
            health_key = f"drone_{d_id}_health"
            healths = [row.get(health_key, 0) for row in telemetry_data]
            c = colors[idx % len(colors)]
            axs[1, 1].plot(steps, healths, color=c, label=f"Drone {d_id:02d}", linewidth=2)
            
        axs[1, 1].grid(True, linestyle='--', alpha=0.5)
        axs[1, 1].legend(loc='lower left')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        for filename in [plot_filename, latest_plot_filename]:
            fig.savefig(filename, dpi=150)
            
        plt.close(fig)
        print(f"[Telemetry] Saved plots: {plot_filename} and {latest_plot_filename}")
        with sim_state.lock:
            sim_state.add_log("sys", "Gráficos de telemetria analítica gerados com sucesso.")
    except Exception as e:
        print(f"[Telemetry Error] Failed to generate plots: {e}")
        with sim_state.lock:
            sim_state.add_log("sys", f"Erro ao gerar gráficos de telemetria: {e}")


def simulation_loop(current_loop_id):
    """
    Background thread running the real-time simulation step iterations.
    """
    global sim_state
    
    # Initialize drones based on config if not already initialized
    num_drones = sim_state.config["num_drones"]
    num_sensors = sim_state.config["num_sensors"]
    
    with sim_state.lock:
        if not sim_state.drones:
            sim_state.fc = ForestController(sim_state.client)
            for i in range(num_drones):
                tc = TreeController(i + 1, sim_state.client)
                tc.init_sensors(num_sensors)
                sim_state.fc.register_tree(tc)
                
                sensor_states = {}
                for name in tc.agents:
                    sensor_states[name] = "active"
                    
                sim_state.drones.append({
                    "drone_id": tc.drone_id,
                    "name": tc.drone_name,
                    "health": 100,
                    "status": "active",
                    "sensor_states": sensor_states,
                    "manual_disabled": [],
                    "tc": tc
                })
            sim_state.add_log("sys", f"Simulação inicializada com {num_drones} drones e {num_sensors} sensores por drone.")
        else:
            sim_state.add_log("sys", "Simulação retomada.")
    
    while True:
        # Check if we should stop
        with sim_state.lock:
            if not sim_state.is_running or sim_state.loop_id != current_loop_id:
                break
            
            step = sim_state.step
            speed = sim_state.config["speed"]
            use_ollama = sim_state.config["use_ollama"]
            
        # Get active drones list
        active_drones_states = [d for d in sim_state.drones if d["status"] == "active"]
        if not active_drones_states:
            sim_state.add_log("sys", "FIM DA MISSÃO: Todos os drones caíram. Enxame inoperante.")
            with sim_state.lock:
                sim_state.is_running = False
            break

        # Select a task for this step
        task = NAV_TASKS[step % len(NAV_TASKS)]
        sim_state.add_log("sys", f"--- PASSO {step} (Tarefa de Navegação) ---")
        
        step_successes = []
        step_tokens = 0
        
        active_tcs = [d["tc"] for d in active_drones_states]

        for drone in active_drones_states:
            tc = drone["tc"]
            
            # Resolve task for this drone
            res = tc.solve_navigation_task(
                task=task,
                manual_disabled_sensors=drone["manual_disabled"],
                use_ollama=use_ollama,
                active_drones=active_tcs
            )
            
            step_successes.append(res["success"])
            step_tokens += res.get("tokens", 0)
            
            # Write solver logs to console
            sim_state.add_log("info" if res["success"] else "fail", res["log"])
            
            # Update drone sensor states mapping on the dashboard
            for name in tc.agents:
                agent = tc.agents[name]
                if name in drone["manual_disabled"]:
                    drone["sensor_states"][name] = "failed"
                elif agent.is_dead():
                    drone["sensor_states"][name] = "pruned"
                elif res.get("primary") == name:
                    drone["sensor_states"][name] = "active"
                elif res.get("helper") == name:
                    drone["sensor_states"][name] = "warning"
                else:
                    drone["sensor_states"][name] = "active"
                    
            # Check for remote symbiosis active representation
            # If local sensor is pruned/failed but helper is remote
            if res.get("helper") and "-" in str(res.get("helper")):
                # E.g. "Drone-02-CAM" -> we are using symbiosis on our local pruned sensor!
                local_sensor = res["primary"]
                if local_sensor in drone["sensor_states"]:
                    drone["sensor_states"][local_sensor] = "symbiosis"
            
            # Vascular resource update on drone health
            if res["success"]:
                drone["health"] = min(100, drone["health"] + 5)
            else:
                drone["health"] = max(0, drone["health"] - 15)
                
            # If health drops to 0, drone crashes
            if drone["health"] <= 0:
                drone["status"] = "crashed"
                for name in drone["sensor_states"]:
                    drone["sensor_states"][name] = "failed"
                sim_state.add_log("fail", f"💥 CRITICAL CRASH: {drone['name']} esgotou todos os recursos vasculares e caiu!")
                
        # Mycorrhizal Sync: Forest Controller shares adaptation vectors between active TreeControllers
        if step > 0 and step % 5 == 0:
            synced = sim_state.fc.mycorrhizal_sync()
            if synced > 0:
                sim_state.add_log("sys", f"🍄 Mycorrhizal Network: Forest Controller sincronizou {synced} memórias semânticas entre os TreeControllers.")

        # Compute Fleet-wide aggregated stats
        acc = sum(1 for s in step_successes if s) / len(step_successes) if step_successes else 0.0
        sim_state.history_acc.append(acc)
        
        # FDI index: standard deviation of success rates among active drones
        fdi_val = np.std(step_successes) if len(step_successes) > 1 else 0.5
        sim_state.history_fdi.append(fdi_val)
        
        # Cumulative tokens
        prev_allostatic = sim_state.history_allostatic[-1] if sim_state.history_allostatic else 0.0
        sim_state.history_allostatic.append(prev_allostatic + step_tokens)

        # Count pruned sensors and crashed drones
        pruned_count = 0
        crashed_count = 0
        for drone in sim_state.drones:
            if drone["health"] <= 0 or drone["status"] == "crashed":
                crashed_count += 1
            # Count dead/pruned sensors
            tc_obj = drone["tc"]
            for name, agent in tc_obj.agents.items():
                if agent.is_dead():
                    pruned_count += 1
        sim_state.history_pruned.append(pruned_count)
        sim_state.history_crashed.append(crashed_count)
        
        # Compute Somatic Memory size and relevance
        active_tcs = [drone["tc"] for drone in sim_state.drones if drone["health"] > 0]
        if active_tcs:
            avg_size = sum(tc.somatic_memory.size() for tc in active_tcs) / len(active_tcs)
            total_rel = 0.0
            total_docs = 0
            for tc in active_tcs:
                for doc in tc.somatic_memory.documents:
                    total_rel += doc["relevance"]
                    total_docs += 1
            avg_relevance = (total_rel / total_docs) if total_docs > 0 else 0.0
        else:
            avg_size = 0.0
            avg_relevance = 0.0
            
        sim_state.history_memory_size.append(avg_size)
        sim_state.history_memory_relevance.append(avg_relevance)
        
        # Collect telemetry snapshot
        snapshot = {
            "step": step,
            "fleet_accuracy": acc,
            "fdi_index": fdi_val,
            "allostatic_load": prev_allostatic + step_tokens,
            "pruned_sensors": pruned_count,
            "crashed_drones": crashed_count,
            "avg_memory_size": avg_size,
            "avg_memory_relevance": avg_relevance,
        }
        
        for drone in sim_state.drones:
            d_id = drone["drone_id"]
            snapshot[f"drone_{d_id}_health"] = drone["health"]
            snapshot[f"drone_{d_id}_status"] = drone["status"]
            
            active_sens = 0
            tc_obj = drone["tc"]
            for name, agent in tc_obj.agents.items():
                snapshot[f"drone_{d_id}_sensor_{name}_energy"] = agent.energy
                snapshot[f"drone_{d_id}_sensor_{name}_status"] = drone["sensor_states"].get(name, "active")
                if not agent.is_dead() and name not in drone["manual_disabled"]:
                    active_sens += 1
            snapshot[f"drone_{d_id}_active_sensors"] = active_sens
            
            for src, dsts in tc_obj.trust_scores.items():
                for dst, trust_val in dsts.items():
                    snapshot[f"drone_{d_id}_trust_{src}_{dst}"] = trust_val
                    
        with sim_state.lock:
            sim_state.telemetry_history.append(snapshot)
        
        # Advance step
        with sim_state.lock:
            sim_state.step += 1

        # Sleep duration regulated by speed multiplier
        sleep_time = max(0.2, 1.5 / speed)
        time.sleep(sleep_time)

    # Save telemetry results on loop exit
    save_telemetry_results()


class JSONRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    HTTP Request Handler exposing REST API endpoints and serving static files.
    """
    def log_message(self, format, *args):
        # Silence default request logging in console
        pass

    def do_POST(self):
        global sim_state
        parsed_url = urllib.parse.urlparse(self.path)
        
        if parsed_url.path == "/api/config":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            with sim_state.lock:
                new_drones = int(data.get("num_drones", 3))
                new_sensors = int(data.get("num_sensors", 5))
                
                # Check if structural reset is needed
                needs_reset = (new_drones != sim_state.config.get("num_drones") or 
                               new_sensors != sim_state.config.get("num_sensors"))
                
                sim_state.config["num_drones"] = new_drones
                sim_state.config["num_sensors"] = new_sensors
                sim_state.config["speed"] = float(data.get("speed", 1.0))
                sim_state.config["use_ollama"] = bool(data.get("use_ollama", False))
                sim_state.config["temperature"] = float(data.get("temperature", 0.3))
                sim_state.config["mutation_rate"] = float(data.get("mutation_rate", 0.1))
                sim_state.config["decay_rate"] = int(data.get("decay_rate", 15))
                sim_state.config["trust_mutation"] = float(data.get("trust_mutation", 0.03))
                
                if needs_reset:
                    sim_state.reset()
                    msg = "Configurações salvas e simulação reiniciada."
                else:
                    msg = "Hiperparâmetros atualizados em tempo real."
                    
            self._send_json({"success": True, "message": msg})
            
        elif parsed_url.path == "/api/start":
            with sim_state.lock:
                if not sim_state.is_running:
                    # If all drones are dead/crashed, reset first
                    active_drones = [d for d in sim_state.drones if d["status"] == "active"]
                    if not sim_state.drones or not active_drones:
                        sim_state.reset()
                    
                    sim_state.is_running = True
                    sim_state.loop_id += 1
                    t_id = sim_state.loop_id
                    threading.Thread(target=simulation_loop, args=(t_id,), daemon=True).start()
                    msg = "Simulação iniciada."
                else:
                    msg = "Simulação já está rodando."
            self._send_json({"success": True, "message": msg})
            
        elif parsed_url.path == "/api/stop":
            with sim_state.lock:
                sim_state.is_running = False
            self._send_json({"success": True, "message": "Simulação pausada."})
            
        elif parsed_url.path == "/api/toggle_sensor":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            drone_id = int(data.get("drone_id"))
            sensor_name = str(data.get("sensor_name"))
            
            success = False
            message = "Drone não encontrado."
            
            for drone in sim_state.drones:
                if drone["drone_id"] == drone_id:
                    if drone["status"] == "crashed":
                        message = "Drone caiu e está inoperante."
                        break
                        
                    tc = drone["tc"]
                    agent = tc.agents.get(sensor_name)
                    if agent and agent.is_dead():
                        message = f"Sensor {sensor_name} está podado em definitivo (tecido morto)."
                        break
                        
                    if sensor_name in drone["manual_disabled"]:
                        drone["manual_disabled"].remove(sensor_name)
                        drone["sensor_states"][sensor_name] = "active"
                        msg_log = f"Sensor {sensor_name} ativado pelo operador."
                        sim_state.add_log("sys", f"{drone['name']}: {msg_log}")
                    else:
                        drone["manual_disabled"].append(sensor_name)
                        drone["sensor_states"][sensor_name] = "failed"
                        msg_log = f"Sensor {sensor_name} DESATIVADO pelo operador."
                        sim_state.add_log("warn", f"{drone['name']}: {msg_log}")
                        
                    success = True
                    message = "Status do sensor alternado."
                    break
                    
            self._send_json({"success": success, "message": message})
        else:
            self.send_error(404, "Endpoint não encontrado")

    def do_GET(self):
        global sim_state
        parsed_url = urllib.parse.urlparse(self.path)
        
        if parsed_url.path == "/api/status":
            # Build clean status JSON (removing non-serializable TC/MA objects)
            with sim_state.lock:
                drones_data = []
                for drone in sim_state.drones:
                    # Calculate MAs energies for details
                    sensor_energies = {}
                    for name, agent in drone["tc"].agents.items():
                        sensor_energies[name] = agent.energy
                        
                    drones_data.append({
                        "drone_id": drone["drone_id"],
                        "name": drone["name"],
                        "health": drone["health"],
                        "status": drone["status"],
                        "sensor_states": drone["sensor_states"],
                        "manual_disabled": drone["manual_disabled"],
                        "sensor_energies": sensor_energies,
                        "trust_scores": drone["tc"].trust_scores
                    })
                    
                status_payload = {
                    "step": sim_state.step,
                    "is_running": sim_state.is_running,
                    "logs": sim_state.logs,
                    "history_acc": sim_state.history_acc,
                    "history_fdi": sim_state.history_fdi,
                    "history_allostatic": sim_state.history_allostatic,
                    "history_pruned": sim_state.history_pruned,
                    "history_crashed": sim_state.history_crashed,
                    "sensor_success": sim_state.sensor_success,
                    "sensor_fail": sim_state.sensor_fail,
                    "history_memory_size": sim_state.history_memory_size,
                    "history_memory_relevance": sim_state.history_memory_relevance,
                    "delegation_links": sim_state.delegation_links,
                    "config": sim_state.config,
                    "drones": drones_data
                }
            self._send_json(status_payload)
        else:
            # Serve static files normally
            super().do_GET()

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


def main():
    # Make sure we run in the server directory to serve index.html
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Configure socket re-use to avoid port-binding lock issues
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), JSONRequestHandler) as httpd:
        print(f"\n============================================================")
        print(f"🚀 DIGITAL PHYTOMER: SIMULADOR DE COGNIÇÃO DE FROTAS")
        print(f"Servidor rodando em: http://localhost:{PORT}")
        print(f"Abra este endereço no seu navegador para ver o dashboard.")
        print(f"============================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nFinalizando servidor...")
            httpd.shutdown()

if __name__ == "__main__":
    main()
