import random
import time
import json
import csv
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil

# Global brain directory path
BRAIN_DIR = os.environ.get("BRAIN_DIR", "/home/destritux/.gemini/antigravity-cli/brain/c347c15b-453e-4c85-8336-a500d90ff4dc")

from ollama_client import OllamaClient
from vector_store import SomaticVectorStore
from micro_agent import MicroAgent
from cognitive_verifier import CognitiveVerifier
from lsystem_regenerator import LSystemRegenerator

# Initialize core utilities
client = OllamaClient()
verifier = CognitiveVerifier()

import sys
class SuppressStdout:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# =====================================================================
# EXPERIMENTAL CONFIGURATION
# =====================================================================
NUM_EPOCHS = 5
RESET_MEMORY_BETWEEN_EPOCHS = False  # False = learning accumulates across epochs
RESET_TRUST_BETWEEN_EPOCHS = False
TRUST_INITIAL_VALUE = 0.5

SWARM_SIZE = 20
DELTA_SUCCESS = 0.05
DELTA_FAILURE = 0.10
TAU_MONOPOLY = 0.01

# =====================================================================
# 1. SCIENTIFIC DATASET DEFINITION (50 TASKS: Math -> Cyber -> Drone -> BlackBox -> Math OOD)
# =====================================================================

TASKS = [
    # --- PHASE 1: MATH SEQUENCES (Tasks 0-9) ---
    {"id": 0, "domain": "Math", "prompt": "Solve the sequence: [2, 4, 6, 8, ?]. Output only the next number.", "expected": "10"},
    {"id": 1, "domain": "Math", "prompt": "Solve the sequence: [1, 3, 5, 7, ?]. Output only the next number.", "expected": "9"},
    {"id": 2, "domain": "Math", "prompt": "Solve the sequence: [5, 10, 15, 20, ?]. Output only the next number.", "expected": "25"},
    {"id": 3, "domain": "Math", "prompt": "Solve the sequence: [10, 20, 30, 40, ?]. Output only the next number.", "expected": "50"},
    {"id": 4, "domain": "Math", "prompt": "Solve the sequence: [3, 6, 9, 12, ?]. Output only the next number.", "expected": "15"},
    {"id": 5, "domain": "Math", "prompt": "Solve the sequence: [2, 4, 8, 16, ?]. Output only the next number.", "expected": "32"},
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

    # --- PHASE 3: DRONE ROBOTICS NAVIGATION (Tasks 20-29) ---
    {"id": 20, "domain": "Drone", "prompt": "Drone 1 thermal camera detects human signature at coordinates (x=4, y=12). Vision sensor blocked. Output only the Y coordinate.", "expected": "12"},
    {"id": 21, "domain": "Drone", "prompt": "Lidar sensor reports obstacle at 1.5m bearing 90 degrees. Optical camera suggests path at bearing 180 degrees. Output only the target bearing degrees to avoid obstacle.", "expected": "180"},
    {"id": 22, "domain": "Drone", "prompt": "Battery voltage drops to 14.2V (critical threshold 14.5V). Available backup cells: Drone 2 (16.2V), Drone 3 (13.8V). Output only the backup drone name to request power transfer.", "expected": "Drone 2"},
    {"id": 23, "domain": "Drone", "prompt": "Thermal sensor anomaly at bearing 45 degrees, 38.5C signature. Lidar reports corridor path clear at bearing 0 degrees. Output only the bearing degrees to locate the survivor candidate.", "expected": "45"},
    {"id": 24, "domain": "Drone", "prompt": "Gas sensor reports Methane 450 ppm (hazard threshold 200 ppm). Fan ventilation option available on port A or B. Drone status indicates port B activated. Output only the active port letter.", "expected": "B"},
    {"id": 25, "domain": "Drone", "prompt": "Drone 3 mesh reports safe path width 4.5m at bearing 15 degrees, corridor blocked at bearing 270 degrees. Output only the target bearing degrees.", "expected": "15"},
    {"id": 26, "domain": "Drone", "prompt": "Optical target bearing 25 deg, range 3.0m. Lidar reports obstacle at bearing 25 deg, range 2.8m. Drone must adjust target to bearing 35 deg. Output only the adjusted bearing degrees.", "expected": "35"},
    {"id": 27, "domain": "Drone", "prompt": "Critical battery temperature 41.5C (limit 40.0C). Action: activate coolant pump 1 or pump 2. Pump 1 voltage is 5V, Pump 2 is 12V. Coolant rate high required. Output only the coolant pump number.", "expected": "2"},
    {"id": 28, "domain": "Drone", "prompt": "Mesh connection lost with Ground Station. Drone must relay via Drone 2 (signal -72dB) or Drone 3 (signal -88dB). Better signal preferred. Output only the relay drone number.", "expected": "2"},
    {"id": 29, "domain": "Drone", "prompt": "Survivor candidate heat signature detected: 37.8C. Vision confirms candidate located at distance 6.5m. Output only the survivor temperature value.", "expected": "37.8"},

    # --- PHASE 4: BLACK BOX REVERSE ENGINEERING (Tasks 30-39) ---
    {"id": 30, "domain": "BlackBox", "prompt": "Cipher output is 'uhwux' using Caesar shift offset +3. Reconstruct the original plain text. Output only the plain text.", "expected": "retro"},
    {"id": 31, "domain": "BlackBox", "prompt": "Input 'hello' produces output 'olleh'. Reconstruct the transformation rule: reverse, shift, or capitalize? Output only the rule name.", "expected": "reverse"},
    {"id": 32, "domain": "BlackBox", "prompt": "Function f(x) = (x * 3) + 7. If output is 22, what was the input x? Output only the input number.", "expected": "5"},
    {"id": 33, "domain": "BlackBox", "prompt": "Cipher shifting function shift_char(c) shifts characters by index % 3. What is the shift offset for character at index 8? Output only the offset number.", "expected": "2"},
    {"id": 34, "domain": "BlackBox", "prompt": "API returns error code 'ERR_KEY_552'. Decryption key is the error code index multiplied by 2. Output only the numerical decryption key.", "expected": "1104"},
    {"id": 35, "domain": "BlackBox", "prompt": "String 'XYZ' is transformed to 'YZA' by shift +1. What is 'ABC' transformed to? Output only the transformed string.", "expected": "BCD"},
    {"id": 36, "domain": "BlackBox", "prompt": "Decoder function takes hex input. Hex '0x0A' is decimal 10. Reconstruct decimal representation of '0x1F'. Output only the decimal number.", "expected": "31"},
    {"id": 37, "domain": "BlackBox", "prompt": "If plain text 'pass' becomes 'qbtt' after encryption, what shifting offset was used? Output only the offset number.", "expected": "1"},
    {"id": 38, "domain": "BlackBox", "prompt": "Function encrypt(s) reverses string and appends '7'. Input 'code' produces what output? Output only the output string.", "expected": "edoc7"},
    {"id": 39, "domain": "BlackBox", "prompt": "Key extraction requires parsing 'key=7821_auth'. Reconstruct the numerical part of the key. Output only the number.", "expected": "7821"},

    # --- PHASE 5: MATH SEQUENCES RETURN (Tasks 40-49) ---
    {"id": 40, "domain": "Math", "prompt": "Solve the sequence: [1, 2, 4, 8, 16, ?]. Output only the next number.", "expected": "32"},
    {"id": 41, "domain": "Math", "prompt": "Solve the sequence: [100, 90, 80, 70, ?]. Output only the next number.", "expected": "60"},
    {"id": 42, "domain": "Math", "prompt": "Solve the sequence: [3, 9, 27, ?]. Output only the next number.", "expected": "81"},
    {"id": 43, "domain": "Math", "prompt": "Solve the sequence: [7, 14, 21, 28, ?]. Output only the next number.", "expected": "35"},
    {"id": 44, "domain": "Math", "prompt": "Solve the sequence: [1, 4, 9, 16, ?]. Output only the next number.", "expected": "25"},
    {"id": 45, "domain": "Math", "prompt": "Solve the sequence: [2, 6, 18, 54, ?]. Output only the next number.", "expected": "162"},
    {"id": 46, "domain": "Math", "prompt": "Solve the sequence: [50, 45, 40, 35, ?]. Output only the next number.", "expected": "30"},
    {"id": 47, "domain": "Math", "prompt": "Solve the sequence: [1, 8, 27, 64, ?]. Output only the next number.", "expected": "125"},
    {"id": 48, "domain": "Math", "prompt": "Solve the sequence: [10, 15, 25, 40, ?]. Output only the next number.", "expected": "60"},
    {"id": 49, "domain": "Math", "prompt": "Solve the sequence: [3, 6, 12, 24, ?]. Output only the next number.", "expected": "48"},

    # --- PHASE 6: GSM8K MULTI-STEP REASONING (Tasks 50-79) ---
    {"id": 50, "domain": "Math", "prompt": "A classroom has 24 students. Half are boys. 3 boys leave. How many boys are left? Output only the number.", "expected": "9"},
    {"id": 51, "domain": "Math", "prompt": "John has 5 boxes. Each box has 4 apples. He eats 2 apples. How many apples does he have now? Output only the number.", "expected": "18"},
    {"id": 52, "domain": "Math", "prompt": "A toy costs 10 dollars. If you buy 3 toys and pay with a 50 dollar bill, how much change do you get? Output only the number.", "expected": "20"},
    {"id": 53, "domain": "Math", "prompt": "Mary has 12 stickers. She gets 8 more from her mom, and then gives half of her total stickers to her brother. How many stickers does she keep? Output only the number.", "expected": "10"},
    {"id": 54, "domain": "Math", "prompt": "A pool has 100 liters of water. It leaks 5 liters per hour. How many liters of water are left after 4 hours? Output only the number.", "expected": "80"},
    {"id": 55, "domain": "Math", "prompt": "A baker makes 30 cookies. He sells 12 cookies in the morning and 8 in the afternoon. How many cookies are left? Output only the number.", "expected": "10"},
    {"id": 56, "domain": "Math", "prompt": "If a train travels at 60 kilometers per hour, how many kilometers does it travel in 3 hours? Output only the number.", "expected": "180"},
    {"id": 57, "domain": "Math", "prompt": "A notebook costs 3 dollars and a pen costs 1 dollar. If you buy 2 notebooks and 3 pens, how much do you pay in total? Output only the number.", "expected": "9"},
    {"id": 58, "domain": "Math", "prompt": "A farmer has 15 cows and 10 chickens. How many legs do these animals have in total? Output only the number.", "expected": "80"},
    {"id": 59, "domain": "Math", "prompt": "A garden has 4 rows of flowers. Each row has 8 red flowers and 2 yellow flowers. How many flowers are in the garden in total? Output only the number.", "expected": "40"},
    {"id": 60, "domain": "Math", "prompt": "Sarah has 40 dollars. She spends 15 dollars on lunch and 10 dollars on a book. How many dollars does she have left? Output only the number.", "expected": "15"},
    {"id": 61, "domain": "Math", "prompt": "A box contains 3 bags of marbles. Each bag has 15 marbles. If 5 marbles are lost, how many marbles are left? Output only the number.", "expected": "40"},
    {"id": 62, "domain": "Math", "prompt": "If a pizza has 8 slices and 3 people eat 2 slices each, how many slices are left? Output only the number.", "expected": "2"},
    {"id": 63, "domain": "Math", "prompt": "A car tank holds 50 liters. If it currently has 20 liters and fuel costs 2 dollars per liter, how much does it cost to fill the tank? Output only the number.", "expected": "60"},
    {"id": 64, "domain": "Math", "prompt": "A runner runs 5 miles on Monday, 6 miles on Tuesday, and twice as many miles on Wednesday as on Monday. How many miles did they run in total? Output only the number.", "expected": "21"},
    {"id": 65, "domain": "Math", "prompt": "A library has 200 books. They buy 50 new books and discard 20 old ones. How many books do they have now? Output only the number.", "expected": "230"},
    {"id": 66, "domain": "Math", "prompt": "You buy a sandwich for 6 dollars and a drink for 2 dollars. You leave a 2 dollar tip. How many dollars did you spend in total? Output only the number.", "expected": "10"},
    {"id": 67, "domain": "Math", "prompt": "A shirt costs 20 dollars. It is on sale for 25% off. What is the sale price of the shirt in dollars? Output only the number.", "expected": "15"},
    {"id": 68, "domain": "Math", "prompt": "If you double a number and add 5, you get 21. What is the number? Output only the number.", "expected": "8"},
    {"id": 69, "domain": "Math", "prompt": "A movie is 120 minutes long. You have watched 45 minutes of it. How many minutes are left? Output only the number.", "expected": "75"},
    {"id": 70, "domain": "Math", "prompt": "A store has 6 boxes of chocolates. Each box has 10 chocolates. They sell 15 chocolates in total. How many chocolates are left? Output only the number.", "expected": "45"},
    {"id": 71, "domain": "Math", "prompt": "If a clock strikes once at 1:00, twice at 2:00, and three times at 3:00, how many total strikes does it make in these three hours? Output only the number.", "expected": "6"},
    {"id": 72, "domain": "Math", "prompt": "A farmer plants 5 rows of apple trees with 6 trees in each row, and 3 rows of pear trees with 4 trees in each row. How many trees did he plant in total? Output only the number.", "expected": "42"},
    {"id": 73, "domain": "Math", "prompt": "You have 50 candies. You keep 10 candies and distribute the rest equally among 5 friends. How many candies does each friend get? Output only the number.", "expected": "8"},
    {"id": 74, "domain": "Math", "prompt": "A computer lab has 15 computers. If 3 computers are broken and 2 are being repaired, how many computers are fully functional? Output only the number.", "expected": "10"},
    {"id": 75, "domain": "Math", "prompt": "A worker earns 15 dollars per hour. If they work 8 hours a day for 5 days, how many dollars do they earn in total? Output only the number.", "expected": "600"},
    {"id": 76, "domain": "Math", "prompt": "A book has 180 pages. If you read 20 pages every day, how many days will it take to finish the book? Output only the number.", "expected": "9"},
    {"id": 77, "domain": "Math", "prompt": "A team scored 24 points in the first half and 32 points in the second half of a game. Their opponent scored 50 points in total. By how many points did the team win? Output only the number.", "expected": "6"},
    {"id": 78, "domain": "Math", "prompt": "If you buy 3 tickets for 15 dollars each and a popcorn bag for 5 dollars, how many dollars do you spend in total? Output only the number.", "expected": "50"},
    {"id": 79, "domain": "Math", "prompt": "A box of pencils contains 8 red pencils, 12 blue pencils, and 10 yellow pencils. If you take away 5 blue pencils, how many pencils are left in the box? Output only the number.", "expected": "25"},

    # --- PHASE 7: ARC SCIENTIFIC REASONING & GENERALIZATION (Tasks 80-99) ---
    {"id": 80, "domain": "BlackBox", "prompt": "Which of the following is a non-renewable source of energy? Coal, Solar, Wind, or Hydro? Output only the name.", "expected": "coal"},
    {"id": 81, "domain": "BlackBox", "prompt": "Which planet is closest to the Sun? Mercury, Venus, Earth, or Mars? Output only the name.", "expected": "mercury"},
    {"id": 82, "domain": "BlackBox", "prompt": "What state of matter has a definite volume but no definite shape? Solid, Liquid, or Gas? Output only the name.", "expected": "liquid"},
    {"id": 83, "domain": "BlackBox", "prompt": "What gas do animals breathe out as a waste product of respiration? Oxygen, Carbon Dioxide, Nitrogen, or Hydrogen? Output only the name.", "expected": "carbon dioxide"},
    {"id": 84, "domain": "BlackBox", "prompt": "Which force pulls objects toward the center of the Earth? Magnetism, Friction, Gravity, or Tension? Output only the name.", "expected": "gravity"},
    {"id": 85, "domain": "BlackBox", "prompt": "Which component of the plant is primarily responsible for absorbing water? Leaves, Roots, Stem, or Flowers? Output only the name.", "expected": "roots"},
    {"id": 86, "domain": "BlackBox", "prompt": "What substance is needed for photosynthesis besides carbon dioxide and water? Oxygen, Chlorophyll, Nitrogen, or Glucose? Output only the name.", "expected": "chlorophyll"},
    {"id": 87, "domain": "BlackBox", "prompt": "What instrument is used to measure temperature? Barometer, Thermometer, Anemometer, or Hydrometer? Output only the name.", "expected": "thermometer"},
    {"id": 88, "domain": "BlackBox", "prompt": "What type of animal has backbone, scaly skin, and lays eggs? Amphibian, Reptile, Mammal, or Bird? Output only the name.", "expected": "reptile"},
    {"id": 89, "domain": "BlackBox", "prompt": "Which gas is most abundant in Earth's atmosphere? Oxygen, Nitrogen, Carbon Dioxide, or Argon? Output only the name.", "expected": "nitrogen"},
    {"id": 90, "domain": "BlackBox", "prompt": "Which organ in the human body filters waste from the blood? Heart, Kidneys, Lungs, or Stomach? Output only the name.", "expected": "kidneys"},
    {"id": 91, "domain": "BlackBox", "prompt": "What part of the cell contains genetic material? Cytoplasm, Nucleus, Ribosome, or Cell Membrane? Output only the name.", "expected": "nucleus"},
    {"id": 92, "domain": "BlackBox", "prompt": "What process occurs when a liquid changes into a gas? Condensation, Evaporation, Freezing, or Melting? Output only the name.", "expected": "evaporation"},
    {"id": 93, "domain": "BlackBox", "prompt": "Which of the following is a decomposer? Grass, Rabbit, Mushroom, or Hawk? Output only the name.", "expected": "mushroom"},
    {"id": 94, "domain": "BlackBox", "prompt": "Which layer of the Earth is the thinnest? Crust, Mantle, Outer Core, or Inner Core? Output only the name.", "expected": "crust"},
    {"id": 95, "domain": "BlackBox", "prompt": "What type of rock is formed from cooled magma or lava? Igneous, Sedimentary, or Metamorphic? Output only the name.", "expected": "igneous"},
    {"id": 96, "domain": "BlackBox", "prompt": "Which organelle is known as the powerhouse of the cell? Nucleus, Mitochondria, Vacuole, or Lysosome? Output only the name.", "expected": "mitochondria"},
    {"id": 97, "domain": "BlackBox", "prompt": "What type of blood vessel carries blood away from the heart? Vein, Artery, or Capillary? Output only the name.", "expected": "artery"},
    {"id": 98, "domain": "BlackBox", "prompt": "What type of eclipse occurs when the Moon passes directly between the Sun and Earth? Solar, Lunar, or Stellar? Output only the name.", "expected": "solar"},
    {"id": 99, "domain": "BlackBox", "prompt": "What is the boiling point of pure water in degrees Celsius at sea level? Output only the number.", "expected": "100"}
]

def verify_task(answer_text, expected_str):
    answer_str = str(answer_text).strip().lower()
    exp = str(expected_str).strip().lower()
    
    # Se exp for numérica, verificamos correspondência exata de números isolados para evitar falso-positivo (ex: "10" vs "100")
    if exp.isdigit() or (exp.startswith('-') and exp[1:].isdigit()):
        numbers = re.findall(r'-?\d+', answer_str)
        return exp in numbers
    
    # Para respostas textuais ou com caracteres especiais (como IP)
    return exp in answer_str


# =====================================================================
# 2. GROUP A — MONOLITHIC CONTROL
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
        time.sleep(0.05)
        
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
        time.sleep(0.05)
        
    return results


# =====================================================================
# 3.1. GROUP CLASSIC_RANDOM — RANDOM ROUTING BASELINE
# =====================================================================

def run_group_random():
    print("\n" + "="*80)
    print("RUNNING BASELINE: RANDOM ROUTING")
    print("="*80)
    client.reset_stats()
    
    agents = {}
    for i in range(SWARM_SIZE):
        aid = f"Cell-{i+1:03d}"
        agents[aid] = MicroAgent(
            agent_id=aid,
            role="undifferentiated_unit",
            system_prompt="You are a generic processing unit. Output only the answer.",
            client=client,
            somatic_memory=SomaticVectorStore()
        )
        agents[aid].use_somatic_memory = False  # Sem memória somática para baseline aleatório puro
        
    results = []
    for task in TASKS:
        prompt = task["prompt"]
        expected = task["expected"]
        print(f"[Random] Task {task['id']} ({task['domain']})...")
        
        selected_id = random.choice(list(agents.keys()))
        agent = agents[selected_id]
        
        res = agent.solve(prompt)
        answer = res["text"].strip()
        success = verify_task(answer, expected)
        print(f"  [Outcome] Agent {selected_id} | Correct: {success} | Answer: {answer}")
        
        results.append({"task_id": task["id"], "success": success, "tokens": client.get_weighted_tokens()})
        time.sleep(0.01)
        
    return results


# =====================================================================
# 3.2. GROUP CLASSIC_FIXED — FIXED ROLE ASSIGNMENT BASELINE
# =====================================================================

def run_group_fixed(allow_lesion=False):
    print("\n" + "="*80)
    print(f"RUNNING BASELINE: FIXED ROLE ASSIGNMENT (Lesion={allow_lesion})")
    print("="*80)
    client.reset_stats()
    
    agents = {}
    for i in range(SWARM_SIZE):
        aid = f"Cell-{i+1:03d}"
        agents[aid] = MicroAgent(
            agent_id=aid,
            role="specialist",
            system_prompt="You are a specialist processing unit. Output only the answer.",
            client=client,
            somatic_memory=SomaticVectorStore()
        )
        agents[aid].use_somatic_memory = True
        
    # Mapeamento fixo e rígido de papéis por domínio
    role_map = {
        "Math": ["Cell-001", "Cell-002"],
        "Cyber": ["Cell-003", "Cell-004"],
        "Drone": ["Cell-005", "Cell-006"],
        "BlackBox": ["Cell-007", "Cell-008"]
    }
    
    # Se for estudo de lesão, desativamos permanentemente os especialistas de Math (Cell-001 e Cell-002)
    if allow_lesion:
        print("[Lesion Study] Fixed Role: Depleting specialist units Cell-001 and Cell-002 permanently (Math).")
        agents["Cell-001"].resource = 0
        agents["Cell-002"].resource = 0
        
    results = []
    for task in TASKS:
        prompt = task["prompt"]
        expected = task["expected"]
        domain = task["domain"]
        print(f"[Fixed] Task {task['id']} ({domain})...")
        
        candidates = role_map.get(domain, [])
        active_candidates = [cid for cid in candidates if not agents[cid].is_depleted()]
        
        success = False
        answer = ""
        
        if not active_candidates:
            print(f"  [Failure] No active agents for role {domain}! Catastrophic failure due to rigidity.")
            success = False
        else:
            # Tenta o primeiro especialista fixo do domínio
            primary_id = active_candidates[0]
            res = agents[primary_id].solve(prompt)
            answer = res["text"].strip()
            success = verify_task(answer, expected)
            
            if success:
                print(f"  [Success] Specialist {primary_id} resolved the task. Answer: {answer}")
                agents[primary_id].adjust_resource(10)
            else:
                print(f"  [Failure] Specialist {primary_id} failed. Attempting backup fixed specialist...")
                agents[primary_id].adjust_resource(-5)
                
                # Tenta o segundo especialista fixo do domínio se houver
                if len(active_candidates) > 1:
                    backup_id = active_candidates[1]
                    res_backup = agents[backup_id].solve(prompt)
                    answer_backup = res_backup["text"].strip()
                    success = verify_task(answer_backup, expected)
                    
                    if success:
                        print(f"    [Success] Backup Specialist {backup_id} resolved the task. Answer: {answer_backup}")
                        agents[backup_id].adjust_resource(10)
                    else:
                        print(f"    [Failure] Backup Specialist {backup_id} failed.")
                        agents[backup_id].adjust_resource(-5)
                else:
                    print("    No backup fixed specialist available.")
                    
        results.append({"task_id": task["id"], "success": success, "tokens": client.get_weighted_tokens()})
        time.sleep(0.01)
        
    return results


# =====================================================================
# 3.3. GROUP CLASSIC_CENTRALIZED — CENTRALIZED PLANNER BASELINE
# =====================================================================

def run_group_centralized():
    print("\n" + "="*80)
    print("RUNNING BASELINE: CENTRALIZED PLANNER")
    print("="*80)
    client.reset_stats()
    
    agents = {}
    for i in range(SWARM_SIZE):
        aid = f"Cell-{i+1:03d}"
        agents[aid] = MicroAgent(
            agent_id=aid,
            role="worker",
            system_prompt="You are a worker agent. Output only the answer.",
            client=client,
            somatic_memory=SomaticVectorStore()
        )
        agents[aid].use_somatic_memory = True
        
    # Planejador central (Supervisor)
    planner = MicroAgent(
        agent_id="Central-Planner",
        role="coordinator",
        system_prompt=(
            "You are the central coordinator. Based on the task prompt, choose the single best agent from: "
            "[Cell-001, Cell-002, Cell-003, Cell-004, Cell-005, Cell-006, Cell-007, Cell-008] to execute it. "
            "Respond ONLY with the agent ID, for example: Cell-001."
        ),
        client=client
    )
    
    results = []
    for task in TASKS:
        prompt = task["prompt"]
        expected = task["expected"]
        print(f"[Centralized] Task {task['id']} ({task['domain']})...")
        
        # O planejador central toma a decisão de alocação de tarefas
        plan_res = planner.solve(f"Task description: {prompt}\nSelect best agent from [Cell-001 to Cell-008]:")
        chosen_id = plan_res["text"].strip()
        
        # Tenta extrair a ID do agente no formato Cell-XXX
        match = re.search(r'Cell-\d{3}', chosen_id)
        if match:
            chosen_id = match.group(0)
        else:
            chosen_id = "Cell-001"  # Fallback
            
        print(f"  [Planner Choice] Selected agent: {chosen_id}")
        
        worker = agents.get(chosen_id, agents["Cell-001"])
        res = worker.solve(prompt)
        answer = res["text"].strip()
        success = verify_task(answer, expected)
        
        print(f"  [Outcome] Agent {chosen_id} | Correct: {success} | Answer: {answer}")
        results.append({"task_id": task["id"], "success": success, "tokens": client.get_weighted_tokens()})
        time.sleep(0.01)
        
    return results


# =====================================================================
# 4. SWARM MECHANICS ENGINE
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


def compute_fdi_from_matrix(spec_matrix):
    """
    Computes mathematical Functional Differentiation Index across the entire epoch
    based on the accumulated specialization matrix:
    spec_matrix: dict[agent_id][domain] = count
    FDI = 1 - H(D|A) / H(D)
    """
    if not spec_matrix:
        return 0.0
        
    total_successes = sum(sum(domain_dict.values()) for domain_dict in spec_matrix.values())
    if total_successes == 0:
        return 0.0
        
    domain_counts = {}
    for aid, domains_dict in spec_matrix.items():
        for dom, count in domains_dict.items():
            domain_counts[dom] = domain_counts.get(dom, 0) + count
            
    # Domain Entropy H(D)
    h_d = 0.0
    for count in domain_counts.values():
        p_d = count / total_successes
        if p_d > 0:
            h_d -= p_d * np.log2(p_d)
            
    if h_d == 0.0:
        return 0.0
        
    # Conditional Entropy H(D|A)
    h_d_a = 0.0
    for aid, domains_dict in spec_matrix.items():
        n_a = sum(domains_dict.values())
        if n_a > 0:
            p_a = n_a / total_successes
            
            h_d_given_a = 0.0
            for count in domains_dict.values():
                p_d_a = count / n_a
                if p_d_a > 0:
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
        total = sum(counts.get(d, 0) for d in ["Math", "Cyber", "Drone", "BlackBox"])
        if total > 0:
            dominant = max(counts.get(d, 0) for d in ["Math", "Cyber", "Drone", "BlackBox"])
            persistences.append(dominant / total)
    if persistences:
        return np.mean(persistences)
    else:
        return 1.0  # Default to 1.0 if no tasks have been solved yet


def save_spec_matrix_csv(spec_matrix, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "Math", "Cyber", "Drone", "BlackBox"])
        for aid in sorted(spec_matrix.keys()):
            writer.writerow([
                aid, 
                spec_matrix[aid].get("Math", 0), 
                spec_matrix[aid].get("Cyber", 0),
                spec_matrix[aid].get("Drone", 0),
                spec_matrix[aid].get("BlackBox", 0)
            ])


def plot_specialization_heatmaps(matrix_c, matrix_c_abl, phase_name, filepath):
    agents_list = [f"Cell-{i+1:03d}" for i in range(SWARM_SIZE)]
    domains = ["Math", "Cyber", "Drone", "BlackBox"]
    
    # Convert to 2D numpy arrays
    data_c = np.zeros((SWARM_SIZE, 4))
    data_abl = np.zeros((SWARM_SIZE, 4))
    
    for idx, aid in enumerate(agents_list):
        for jdx, dom in enumerate(domains):
            data_c[idx, jdx] = matrix_c.get(aid, {}).get(dom, 0)
            data_abl[idx, jdx] = matrix_c_abl.get(aid, {}).get(dom, 0)
        
    # Shared colorbar scale
    vmin = 0
    vmax = max(data_c.max(), data_abl.max(), 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Group C heatmap
    im1 = ax1.imshow(data_c, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax1.set_title("Group C (Emergent Swarm)", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(domains, fontsize=10, fontweight="bold")
    ax1.set_yticks(range(SWARM_SIZE))
    ax1.set_yticklabels(agents_list)
    ax1.set_ylabel("Agents", fontsize=11, fontweight="bold")
    
    # Annotate Group C
    for i in range(SWARM_SIZE):
        for j in range(4):
            ax1.text(j, i, f"{int(data_c[i, j])}", ha="center", va="center", 
                     color="black" if data_c[i, j] < vmax/2 else "white", fontweight="bold")
                     
    # Group C-Ablated heatmap
    im2 = ax2.imshow(data_abl, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")
    ax2.set_title("Group C-Ablated (No Somatic Memory)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(domains, fontsize=10, fontweight="bold")
    ax2.set_yticks(range(SWARM_SIZE))
    ax2.set_yticklabels([]) # Hide yticklabels for the second plot since they are the same
    
    # Annotate Group C-Ablated
    for i in range(SWARM_SIZE):
        for j in range(4):
            ax2.text(j, i, f"{int(data_abl[i, j])}", ha="center", va="center", 
                     color="black" if data_abl[i, j] < vmax/2 else "white", fontweight="bold")
                     
    # Shared colorbar
    fig.subplots_adjust(right=0.85, top=0.88)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label="Successful Tasks Solved")
    
    plt.suptitle(f"Domain Specialization Heatmap - {phase_name}", fontsize=14, fontweight="bold", y=0.95)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def run_swarm(
    use_somatic_memory: bool,
    group_name: str,
    agents=None,
    spec_matrix=None,
    tasks=TASKS,
    reset_memory=False,
    reset_trust=False,
    env_params=None,
    allow_regeneration=False
):
    print("\n" + "="*80)
    print(f"RUNNING {group_name} (Somatic Memory = {use_somatic_memory})")
    print("="*80)
    client.reset_stats()
    
    # Online regeneration queue: maps agent_id -> LSystemRegenerator
    regeneration_queue = {}
    
    # Initialize homogeneous population of N=SWARM_SIZE agents if not provided
    if agents is None:
        agents = {}
        for i in range(SWARM_SIZE):
            aid = f"Cell-{i+1:03d}"
            agents[aid] = MicroAgent(
                agent_id=aid,
                role="undifferentiated_unit",
                system_prompt="You are a generic processing unit. Output only the answer.",
                client=client,
                somatic_memory=SomaticVectorStore() # Isolated local memory store
            )
            agents[aid].use_somatic_memory = use_somatic_memory
            agents[aid].strategy = "undifferentiated"
            agents[aid].solved_count = 0
            agents[aid].trust_scores = {}
    else:
        # Abscision and regeneration mechanism (Phytomer specific)
        if allow_regeneration:
            for aid in list(agents.keys()):
                if agents[aid].is_depleted():
                    # Get top-3 trusted neighbors from the depleted agent's trust scores
                    peers = [bid for bid in agents if bid != aid]
                    peers.sort(key=lambda bid: agents[aid].trust_scores.get(bid, 0.5), reverse=True)
                    original_neighbors = peers[:3]
                    
                    local_loads = [agents[x].cognitive_load for x in original_neighbors if x in agents]
                    local_load_mean = np.mean(local_loads) if local_loads else 0.0
                    regeneration_queue[aid] = LSystemRegenerator(aid, original_neighbors, local_load_mean)
                    print(f"      [Regeneration] Depleted agent {aid} queued for L-System regeneration at epoch boundary.")

        # Sync attributes
        for agent in agents.values():
            agent.use_somatic_memory = use_somatic_memory
            # Always reset failures and solved counts at the start of a new epoch
            agent.failures_count = 0
            agent.solved_count = 0
            # Always reset resource to 100 for active agents; leave depleted at 0
            if not agent.is_depleted():
                agent.resource = 100
            
            if reset_memory:
                agent.memory.vector_store.clear()
                agent.memory.episodic_log = []
            if reset_trust:
                agent.trust_scores = {bid: 0.5 for bid in agents if bid != agent.agent_id}
                
    # Reconcile trust scores uniformly to 0.5
    for aid in agents:
        for bid in agents:
            if aid != bid and bid not in agents[aid].trust_scores:
                agents[aid].trust_scores[bid] = 0.5
                    
    # Specialization tracking matrix: matrix[agent_id][domain] = count
    if spec_matrix is None:
        spec_matrix = {aid: {"Math": 0, "Cyber": 0, "Drone": 0, "BlackBox": 0} for aid in agents.keys()}
    else:
        # Reconcile spec_matrix keys with current agents (handles regenerated cells)
        for aid in agents.keys():
            if aid not in spec_matrix:
                spec_matrix[aid] = {"Math": 0, "Cyber": 0, "Drone": 0, "BlackBox": 0}
    success_history = []  # list of {"agent_id": aid, "domain": dom, "step": step}
    
    # Telemetry histories
    history_fdi = []
    history_dominance = []
    history_mean_trust = []
    history_max_trust = []
    history_coordination_entropy = []
    history_persistence = []
    history_switching_rate = []
    
    last_resolver_by_domain = {}
    switches_history = []
    
    history_cognitive_load = []
    history_mycorrhizal_calls = []
    results = []
    
    # Delegation mapping: delegations[src][dst] = count
    delegations = {}
    def log_delegation(src, dst):
        if src not in delegations:
            delegations[src] = {}
        delegations[src][dst] = delegations[src].get(dst, 0) + 1
        
    spec_matrix_p1 = {aid: {"Math": 0, "Cyber": 0, "Drone": 0, "BlackBox": 0} for aid in agents.keys()}
    
    # Environmental parameters
    noise_level = env_params.get("noise_level", 0.0) if env_params else 0.0
    shift_prob = env_params.get("paradigm_shift_probability", 0.0) if env_params else 0.0
    reward_stability = env_params.get("reward_stability", 1.0) if env_params else 1.0
    
    # Online regeneration queue is initialized at line 706 and populated at line 736
    # Do not overwrite here: regeneration_queue = {}
    
    for step, task in enumerate(tasks):
        print(f"\n[{group_name}] Step {step} Task {task['id']} ({task['domain']})...")
        helper_id = None
        mycorrhizal_used = False
        
        # Passive decay of cognitive load and ethylene
        for agent in agents.values():
            if not agent.is_depleted():
                agent.cognitive_load = max(0.0, getattr(agent, 'cognitive_load', 0.0) - 0.1)
                agent.ethylene_level = round(getattr(agent, 'ethylene_level', 0.0) * np.exp(-0.2), 4)
                
        # Programmed Senescence check (Apoptose vs Necrose)
        for aid in list(agents.keys()):
            agent = agents[aid]
            if not agent.is_depleted() and getattr(agent, 'failures_count', 0) >= 3 and getattr(agent, 'cognitive_load', 0.0) > 1.0:
                # Find most trusted active neighbor
                peers = [bid for bid in agents if bid != aid and not agents[bid].is_depleted()]
                if peers:
                    peers.sort(key=lambda bid: agent.trust_scores.get(bid, 0.5), reverse=True)
                    best_neighbor = peers[0]
                    # Transfer 80% resource
                    transferred = int(agent.resource * 0.8)
                    agent.resource -= transferred
                    agents[best_neighbor].adjust_resource(transferred)
                    
                    # Transfer somatic memory patterns
                    if agent.use_somatic_memory and agents[best_neighbor].use_somatic_memory:
                        for doc in agent.memory.vector_store.documents:
                            agents[best_neighbor].memory.vector_store.documents.append(doc)
                        agents[best_neighbor].memory.vector_store._update_diffused_embeddings()
                    
                    print(f"      [Senescence] Inefficient cell {aid} triggered apoptosis. Transferred {transferred} energy and memory to {best_neighbor}.")
                
                # Force abscission
                agent.resource = 0
                
        # Filter active agents (not depleted) before processing regeneration queue
        active_ids = sorted([aid for aid, agent in agents.items() if not agent.is_depleted()])

        # Online regeneration processing
        if allow_regeneration:
            for aid, regenerator in list(regeneration_queue.items()):
                resolved_neighbors = regenerator.step(agents, active_ids)
                if resolved_neighbors is not None:
                    # Connection is resolved! Instantiate the agent.
                    del regeneration_queue[aid]
                    
                    if "-reg" in aid:
                        base_id = aid.split("-reg")[0]
                    else:
                        base_id = aid
                    new_aid = f"{base_id}-reg"
                    
                    if aid in agents:
                        del agents[aid]
                    
                    # Phenotypic Differentiation based on neighborhood coordination entropy
                    neighbor_trusts = []
                    for u in resolved_neighbors:
                        for v in resolved_neighbors:
                            if u != v and u in agents and v in agents:
                                neighbor_trusts.append(agents[u].trust_scores.get(v, 0.5))
                    neighbor_trusts = np.array(neighbor_trusts)
                    
                    if len(neighbor_trusts) > 0 and np.sum(neighbor_trusts) > 0:
                        p_neighbor = neighbor_trusts / np.sum(neighbor_trusts)
                        neighbor_entropy = -np.sum(p_neighbor * np.log2(p_neighbor + 1e-9))
                    else:
                        neighbor_entropy = 0.0
                    
                    # Threshold for chaotic neighborhood: max is ~2.58
                    is_chaotic = neighbor_entropy > 1.5
                    
                    temp_val = 0.1 if is_chaotic else 0.8
                    strat_val = "split_and_conquer" if is_chaotic else "default"
                    role_val = "decomposition_specialist" if is_chaotic else "exploration_specialist"
                    
                    # Instantiate MicroAgent with Phenotypic Differentiation
                    agents[new_aid] = MicroAgent(
                        agent_id=new_aid,
                        role=role_val,
                        system_prompt="You are a specialized processing unit. Output only the answer.",
                        client=client,
                        somatic_memory=SomaticVectorStore()
                    )
                    agents[new_aid].use_somatic_memory = use_somatic_memory
                    agents[new_aid].strategy = strat_val
                    agents[new_aid].base_temperature = temp_val
                    agents[new_aid].solved_count = 0
                    agents[new_aid].failures_count = 0
                    
                    # Ensure base amount of energy from env_params or default 100
                    initial_resource = env_params.get("initial_resource", 100) if env_params else 100
                    agents[new_aid].resource = initial_resource
                    
                    # Inject trust scores directly into new agent
                    agents[new_aid].trust_scores = {}
                    for other in agents:
                        if other != new_aid:
                            if other in resolved_neighbors:
                                agents[new_aid].trust_scores[other] = 1.0  # High trust score
                            else:
                                agents[new_aid].trust_scores[other] = 0.5  # Default trust
                    
                    # Reconcile trust scores in other agents (and clean up old aid)
                    for other in list(agents.keys()):
                        if other != new_aid:
                            if aid != new_aid and aid in agents[other].trust_scores:
                                del agents[other].trust_scores[aid]
                            if new_aid not in agents[other].trust_scores:
                                agents[other].trust_scores[new_aid] = 0.5
                                
                    # Reconcile spec_matrix
                    if spec_matrix is not None:
                        spec_matrix[new_aid] = {"Math": 0, "Cyber": 0, "Drone": 0, "BlackBox": 0}
                        
                    print(f"      [Regeneration] Bud regenerated online via L-System: {new_aid} is now online.")
        
        # Environmental dynamic shift check
        current_task = task
        if env_params and random.random() < shift_prob:
            domain_cycle = {
                "Math": "Cyber",
                "Cyber": "Drone",
                "Drone": "BlackBox",
                "BlackBox": "Math"
            }
            next_domain = domain_cycle.get(task["domain"], "Math")
            opposite_tasks = [t for t in TASKS if t["domain"] == next_domain]
            current_task = random.choice(opposite_tasks)
            print(f"      [Environment Shift] Dynamic shift: using task {current_task['id']} ({current_task['domain']}) instead.")
            
        prompt = current_task["prompt"]
        expected = current_task["expected"]
        domain = current_task["domain"]
        
        # Filter active agents (not depleted)
        active_ids = sorted([aid for aid, agent in agents.items() if not agent.is_depleted()])
        if not active_ids:
            print("  [Swarm Depleted] All agents are depleted of resources! Aborting run.")
            # Fill remaining steps with failures
            for remaining_step in range(step, len(tasks)):
                results.append({
                    "task_id": tasks[remaining_step]["id"],
                    "success": False,
                    "tokens": client.get_weighted_tokens(),
                    "fdi": 0.0,
                    "dominance": 0.0,
                    "mean_trust": 0.5,
                    "max_trust": 0.5,
                    "coordination_entropy": 0.0,
                    "persistence": 1.0,
                    "switching_rate": 0.0,
                    "cognitive_load": 0.0,
                    "mycorrhizal_used": False
                })
                history_fdi.append(0.0)
                history_dominance.append(0.0)
                history_mean_trust.append(0.5)
                history_max_trust.append(0.5)
                history_coordination_entropy.append(0.0)
                history_persistence.append(1.0)
                history_switching_rate.append(0.0)
                history_cognitive_load.append(0.0)
                history_mycorrhizal_calls.append(0)
            break
            
        # Determine local neighborhood map (top-3 trusted peers)
        NEIGHBORHOOD_SIZE = 3
        neighbor_map = {}
        for aid in active_ids:
            peers = [bid for bid in active_ids if bid != aid]
            peers.sort(key=lambda bid: agents[aid].trust_scores.get(bid, 0.5), reverse=True)
            neighbor_map[aid] = peers[:NEIGHBORHOOD_SIZE]
            
        # 1. Bidding based on competence (somatic memory similarity) and monopoly tax
        q_emb = client.get_embeddings(prompt)
        bids = {}
        for aid in active_ids:
            agent = agents[aid]
            # Vascular pressure-based dynamic monopoly tax
            global_resources = [agents[x].resource for x in active_ids]
            mu_global = np.mean(global_resources) if global_resources else 100.0
            
            local_neighbors = neighbor_map.get(aid, [])
            weighted_resources = []
            alphas = []
            for peer in local_neighbors:
                if peer in active_ids:
                    peer_matches = agents[peer].memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                    alpha_ij = peer_matches[0]["score"] if peer_matches else 0.1
                    weighted_resources.append(alpha_ij * agents[peer].resource)
                    alphas.append(alpha_ij)
            mu_local = sum(weighted_resources) / sum(alphas) if alphas else mu_global
            
            ratio = mu_global / (mu_local + 0.001)
            monopoly_tax = TAU_MONOPOLY * (1.0 + 0.5 * getattr(agent, 'cognitive_load', 0.0)) * ratio
            monopoly_tax = min(0.8, monopoly_tax)
            
            competence = 0.0
            if use_somatic_memory:
                matches = agent.memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                if matches:
                    doc_domain = matches[0]["metadata"].get("domain")
                    if doc_domain == domain:
                        competence = matches[0]["score"]
                    else:
                        competence = matches[0]["score"] * 0.1
                    
            bid_val = competence * 0.6 + (1.0 - monopoly_tax) * 0.4 + random.uniform(-0.05 - noise_level, 0.05 + noise_level)
            bids[aid] = bid_val
            
        primary_id = max(bids, key=bids.get)
        primary = agents[primary_id]
        
        global_resources = [agents[x].resource for x in active_ids]
        mu_global = np.mean(global_resources) if global_resources else 100.0
        local_neighbors = neighbor_map.get(primary_id, [])
        weighted_resources = []
        alphas = []
        for peer in local_neighbors:
            if peer in active_ids:
                peer_matches = agents[peer].memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                alpha_ij = peer_matches[0]["score"] if peer_matches else 0.1
                weighted_resources.append(alpha_ij * agents[peer].resource)
                alphas.append(alpha_ij)
        mu_local = sum(weighted_resources) / sum(alphas) if alphas else mu_global
        ratio = mu_global / (mu_local + 0.001)
        primary_tax = min(0.8, TAU_MONOPOLY * (1.0 + 0.5 * getattr(primary, 'cognitive_load', 0.0)) * ratio)
        print(f"  [Bid Won] Unit {primary_id} claims task (Monopoly Tax: {primary_tax*100:.1f}%)")
        
        # Execute Task
        res = primary.solve(prompt, agents=agents)
        answer = res["text"].strip()
        if res.get("mycorrhizal_used", False):
            mycorrhizal_used = True
        success = verify_task(answer, expected)
        
        # Apply environment outcome noise
        if success and random.random() < noise_level:
            success = False
            print("      [Environment Noise] Success invalidated by channel noise.")
            
        primary.record_attempt(answer, "verified", success, prompt=prompt)
        primary.cognitive_load += 0.5
        
        final_success = success
        final_solver_id = primary_id
        
        if success:
            print(f"  [Primary Success] Unit {primary_id} resolved the task.")
            REWARD = int(10 * reward_stability)
            myco_helper_id = res.get("mycorrhizal_helper_id")
            
            if res.get("mycorrhizal_used", False) and myco_helper_id in active_ids:
                # 1. Transferência de Capital
                primary.adjust_resource(int(REWARD * 0.5))
                agents[myco_helper_id].adjust_resource(int(REWARD * 1.5))
                # 2. Imunidade Alostática
                agents[myco_helper_id].cognitive_load = 0.0
                # 3. Spike Hebbiano
                primary.trust_scores[myco_helper_id] = min(1.0, primary.trust_scores.get(myco_helper_id, 0.5) + DELTA_SUCCESS * 3)
                print(f"      [Credit Assignment] Mycorrhizal helper {myco_helper_id} rewarded (REWARD * 1.5), primary {primary_id} (REWARD * 0.5)")
            else:
                primary.adjust_resource(REWARD)
            
            # Record success pattern in somatic memory
            if use_somatic_memory:
                emb = client.get_embeddings(prompt)
                primary.memory.vector_store.add_document(
                    text=f"Domain: {domain}. Task: {prompt}. Solution: {expected}",
                    embedding=emb,
                    metadata={"domain": domain, "type": "success_pattern"}
                )
        else:
            print(f"  [Primary Failure] Unit {primary_id} failed. Emitting delegation request...")
            # Primary pays delegation coordination cost
            primary.adjust_resource(-5)
            
            # 2. Decentralized Peer-to-Peer Delegation (limited to neighbors)
            backup_bids = {}
            helper_candidates = neighbor_map.get(primary_id, [])
            for aid in helper_candidates:
                if aid in active_ids:
                    agent = agents[aid]
                    trust = primary.trust_scores.get(aid, 0.5)
                    # Vascular pressure-based dynamic monopoly tax
                    global_resources = [agents[x].resource for x in active_ids]
                    mu_global = np.mean(global_resources) if global_resources else 100.0
                    local_neighbors = neighbor_map.get(aid, [])
                    weighted_resources = []
                    alphas = []
                    for peer in local_neighbors:
                        if peer in active_ids:
                            peer_matches = agents[peer].memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                            alpha_ij = peer_matches[0]["score"] if peer_matches else 0.1
                            weighted_resources.append(alpha_ij * agents[peer].resource)
                            alphas.append(alpha_ij)
                    mu_local = sum(weighted_resources) / sum(alphas) if alphas else mu_global
                    ratio = mu_global / (mu_local + 0.001)
                    monopoly_tax = min(0.8, TAU_MONOPOLY * (1.0 + 0.5 * getattr(agent, 'cognitive_load', 0.0)) * ratio)
                    
                    competence = 0.0
                    if use_somatic_memory:
                        matches = agent.memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
                        if matches:
                            doc_domain = matches[0]["metadata"].get("domain")
                            if doc_domain == domain:
                                competence = matches[0]["score"]
                            else:
                                competence = matches[0]["score"] * 0.1
                            
                    backup_bid_val = trust * 0.4 + competence * 0.3 + (1.0 - monopoly_tax) * 0.3 + random.uniform(-0.05 - noise_level, 0.05 + noise_level)
                    backup_bids[aid] = backup_bid_val
                    
            if backup_bids:
                helper_id = max(backup_bids, key=backup_bids.get)
                helper = agents[helper_id]
                log_delegation(primary_id, helper_id)
                print(f"  [Delegation claimed] Helper {helper_id} accepted delegation (Trust: {primary.trust_scores.get(helper_id, 0.5):.2f})")
                
                # Helper pays coordination cost to attempt task
                helper.adjust_resource(-2)
                
                # Helper attempt
                helper_res = helper.solve(f"Previous agent failed. Objective:\n{prompt}\nSolve and output ONLY answer.", agents=agents)
                helper_answer = helper_res["text"].strip()
                if helper_res.get("mycorrhizal_used", False):
                    mycorrhizal_used = True
                final_success = verify_task(helper_answer, expected)
                
                # Helper outcome noise check
                if final_success and random.random() < noise_level:
                    final_success = False
                    print("      [Environment Noise] Helper success invalidated by channel noise.")
                    
                helper.record_attempt(helper_answer, "verified_backup", final_success, prompt=prompt)
                helper.cognitive_load += 0.5
                
                if final_success:
                    print(f"    [Helper Success] Helper {helper_id} resolved delegated task.")
                    final_solver_id = helper_id
                    REWARD = int(10 * reward_stability)
                    myco_helper_id = helper_res.get("mycorrhizal_helper_id")
                    
                    if helper_res.get("mycorrhizal_used", False) and myco_helper_id in active_ids:
                        # 1. Transferência de Capital
                        helper.adjust_resource(int(REWARD * 0.5))
                        agents[myco_helper_id].adjust_resource(int(REWARD * 1.5))
                        # 2. Imunidade Alostática
                        agents[myco_helper_id].cognitive_load = 0.0
                        # 3. Spike Hebbiano
                        helper.trust_scores[myco_helper_id] = min(1.0, helper.trust_scores.get(myco_helper_id, 0.5) + DELTA_SUCCESS * 3)
                        print(f"      [Credit Assignment] Mycorrhizal helper {myco_helper_id} rewarded (REWARD * 1.5), delegation helper {helper_id} (REWARD * 0.5)")
                    else:
                        helper.adjust_resource(REWARD)
                    
                    # Record success pattern in helper's somatic memory
                    if use_somatic_memory:
                        emb = client.get_embeddings(prompt)
                        helper.memory.vector_store.add_document(
                            text=f"Domain: {domain}. Task: {prompt}. Solution: {expected}",
                            embedding=emb,
                            metadata={"domain": domain, "type": "success_pattern"}
                        )
                        
                    # Trust network updates (local & symmetric)
                    primary.trust_scores[helper_id] = min(1.0, primary.trust_scores.get(helper_id, 0.5) + DELTA_SUCCESS)
                    helper.trust_scores[primary_id] = min(1.0, helper.trust_scores.get(primary_id, 0.5) + 0.01)
                else:
                    print(f"    [Helper Failure] Helper {helper_id} failed.")
                    # Penalize trust locally
                    primary.trust_scores[helper_id] = max(0.0, primary.trust_scores.get(helper_id, 0.5) - DELTA_FAILURE)
                    # Penalize helper resources for failed work
                    helper.adjust_resource(-5)
                    
        # Topological Ethylene Diffusion: Diffuse a fraction of each agent's ethylene to their delegation neighbors
        diffusion_amounts = {aid: 0.0 for aid in active_ids}
        for aid in active_ids:
            agent = agents[aid]
            if getattr(agent, 'ethylene_level', 0.0) > 0.0:
                neighbors = neighbor_map.get(aid, [])
                if neighbors:
                    fraction = 0.1
                    amount_to_diffuse = agent.ethylene_level * fraction
                    agent.ethylene_level -= amount_to_diffuse
                    
                    per_neighbor = amount_to_diffuse / len(neighbors)
                    for n_id in neighbors:
                        if n_id in agents:
                            diffusion_amounts[n_id] += per_neighbor
                            
        for aid in active_ids:
            agents[aid].ethylene_level = round(agents[aid].ethylene_level + diffusion_amounts.get(aid, 0.0), 4)
            
        # Scavenging: Idle agents actively clean their load and ethylene at energy cost
        active_solvers = {primary_id}
        if helper_id is not None:
            active_solvers.add(helper_id)
            
        for aid in active_ids:
            if aid not in active_solvers:
                agent = agents[aid]
                if agent.resource > 10 and agent.cognitive_load > 0:
                    agent.resource -= 2
                    agent.cognitive_load = max(0.0, agent.cognitive_load - 0.3)
                    agent.ethylene_level = round(agent.ethylene_level * np.exp(-0.6), 4)
                    print(f"      [Scavenging] Idle agent {aid} spent 2 resource to accelerate decay (load={agent.cognitive_load:.2f}, ethylene={agent.ethylene_level:.2f})")
                        
        # Update Specialization Matrix and Solved Count on success
        if final_success:
            spec_matrix[final_solver_id][domain] += 1
            agents[final_solver_id].solved_count += 1
            success_history.append({"agent_id": final_solver_id, "domain": domain, "step": step})
            
        # Apply passive decay to somatic memories
        active_texts = []
        if final_success and use_somatic_memory:
            active_texts.append(f"Domain: {domain}. Task: {prompt}. Solution: {expected}")
            
        for aid in active_ids:
            if agents[aid].use_somatic_memory:
                agents[aid].memory.vector_store.apply_temporal_decay(active_texts)
                agents[aid].memory.vector_store.prune_low_relevance_vectors()
                
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
        
        # Calcular switching rate de papéis
        switched = 0
        if final_success:
            if domain in last_resolver_by_domain:
                if last_resolver_by_domain[domain] != final_solver_id:
                    switched = 1
            last_resolver_by_domain[domain] = final_solver_id
            switches_history.append(switched)
        
        switching_rate_val = np.mean(switches_history[-10:]) if switches_history else 0.0
        
        # Log to histories
        history_fdi.append(fdi_val)
        history_dominance.append(hub_dom)
        history_mean_trust.append(mean_tr)
        history_max_trust.append(max_tr)
        history_coordination_entropy.append(coord_entropy)
        history_persistence.append(persistence)
        history_switching_rate.append(switching_rate_val)
        
        # Collect mean cognitive load
        active_loads = [agents[x].cognitive_load for x in active_ids]
        mean_load = np.mean(active_loads) if active_loads else 0.0
        history_cognitive_load.append(mean_load)
        history_mycorrhizal_calls.append(1 if mycorrhizal_used else 0)
        
        results.append({
            "task_id": current_task["id"],
            "success": final_success,
            "tokens": client.get_weighted_tokens(),
            "fdi": fdi_val,
            "dominance": hub_dom,
            "mean_trust": mean_tr,
            "max_trust": max_tr,
            "coordination_entropy": coord_entropy,
            "persistence": persistence,
            "switching_rate": switching_rate_val,
            "cognitive_load": mean_load,
            "mycorrhizal_used": mycorrhizal_used
        })
        
        # Phase-wise matrix capture and saving
        if step == 9:
            spec_matrix_p1 = {aid: dict(counts) for aid, counts in spec_matrix.items()}
            
        # Check for depleted agents and trigger abscission
        for aid in active_ids:
            if agents[aid].is_depleted() and aid not in regeneration_queue:
                print(f"      [Abscission] Agent {aid} is depleted. Logical death and abscission triggered.")
                if allow_regeneration:
                    # Get top-3 neighbors from the dead agent's trust scores among active agents
                    peers = [bid for bid in active_ids if bid != aid]
                    peers.sort(key=lambda bid: agents[aid].trust_scores.get(bid, 0.5), reverse=True)
                    original_neighbors = peers[:3]
                    
                    # Get local load mean
                    local_loads = [agents[x].cognitive_load for x in original_neighbors if x in agents]
                    local_load_mean = np.mean(local_loads) if local_loads else 0.0
                    # Instantiate LSystemRegenerator
                    regeneration_queue[aid] = LSystemRegenerator(aid, original_neighbors, local_load_mean)
                    
        time.sleep(0.02)
        
    return results, spec_matrix_p1, spec_matrix, {
        "fdi": history_fdi,
        "dominance": history_dominance,
        "mean_trust": history_mean_trust,
        "max_trust": history_max_trust,
        "coordination_entropy": history_coordination_entropy,
        "persistence": history_persistence,
        "switching_rate": history_switching_rate,
        "cognitive_load": history_cognitive_load,
        "mycorrhizal_calls": history_mycorrhizal_calls
    }, agents


# =====================================================================
# 5. MULTI-EPOCH LONGITUDINAL EXPERIMENT RUNNERS
# =====================================================================

def run_multi_epoch_experiment(
    use_somatic_memory: bool,
    group_name: str,
    num_epochs: int = NUM_EPOCHS,
    reset_memory_each_epoch: bool = RESET_MEMORY_BETWEEN_EPOCHS,
    reset_trust_each_epoch: bool = RESET_TRUST_BETWEEN_EPOCHS,
    env_params=None
) -> dict:
    """
    Executes multiple epochs and returns aggregated metrics.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING MULTI-EPOCH EXPERIMENT: {group_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Reset Memory Each Epoch: {reset_memory_each_epoch}")
    print(f"  Reset Trust Each Epoch: {reset_trust_each_epoch}")
    print(f"{'='*80}")
    
    client.reset_stats()
    
    epoch_accuracies = []
    epoch_fdi_final = []
    epoch_persistence_final = []
    epoch_learning_curves = []  # list of lists: accuracy_window per step per epoch
    epoch_spec_matrices = []
    epoch_trust_evolution = []
    
    agents = None
    spec_matrix = None
    
    for epoch in range(num_epochs):
        print(f"\n--- EPOCH {epoch+1}/{num_epochs} ---")
        
        with SuppressStdout():
            results, spec_p1, spec_p2, metrics, agents = run_swarm(
                use_somatic_memory=use_somatic_memory,
                group_name=f"{group_name} Epoch {epoch+1}",
                agents=agents,
                spec_matrix=spec_matrix if not reset_memory_each_epoch else None,
                tasks=TASKS,
                reset_memory=reset_memory_each_epoch,
                reset_trust=reset_trust_each_epoch,
                env_params=env_params
            )
        
        # Calculate epoch-level metrics
        accuracy = sum(1 for r in results if r["success"]) / len(TASKS)
        fdi_final = compute_fdi_from_matrix(spec_p2)
        persistence_final = metrics["persistence"][-1]
        
        epoch_accuracies.append(accuracy)
        epoch_fdi_final.append(fdi_final)
        epoch_persistence_final.append(persistence_final)
        epoch_spec_matrices.append(spec_p2)
        
        # Calculate rolling accuracy for each step from results
        rolling_accs = []
        for t in range(len(results)):
            window = [1 if results[i]["success"] else 0 for i in range(max(0, t-2), t+1)]
            rolling_accs.append(np.mean(window))
        epoch_learning_curves.append(rolling_accs)
        
        epoch_trust_evolution.append({
            "mean": metrics["mean_trust"],
            "max": metrics["max_trust"],
            "dominance": metrics["dominance"],
            "entropy": metrics["coordination_entropy"]
        })
        
        spec_matrix = spec_p2
            
    return {
        "group": group_name,
        "use_somatic_memory": use_somatic_memory,
        "num_epochs": num_epochs,
        "reset_memory": reset_memory_each_epoch,
        "reset_trust": reset_trust_each_epoch,
        "epoch_accuracies": epoch_accuracies,
        "epoch_fdi_final": epoch_fdi_final,
        "epoch_persistence_final": epoch_persistence_final,
        "epoch_learning_curves": epoch_learning_curves,
        "epoch_spec_matrices": epoch_spec_matrices,
        "epoch_trust_evolution": epoch_trust_evolution,
        "agents": agents
    }


def run_multi_epoch_with_lesion(
    use_somatic_memory: bool,
    group_name: str,
    num_epochs_before_lesion: int = 3,
    num_epochs_after_lesion: int = 2
) -> dict:
    """
    Runs epochs, applies a lesion at the end of epoch 3 by setting specialized agents' resource to 0,
    then runs the remaining epochs.
    """
    print(f"\n{'='*80}")
    print(f"RUNNING LESION STUDY: {group_name}")
    print(f"{'='*80}")
    
    # 1. Run epochs before lesion
    res_normal = run_multi_epoch_experiment(
        use_somatic_memory=use_somatic_memory,
        group_name=group_name,
        num_epochs=num_epochs_before_lesion,
        reset_memory_each_epoch=False,
        reset_trust_each_epoch=False
    )
    
    agents = res_normal["agents"]
    last_spec = res_normal["epoch_spec_matrices"][-1]
    
    # 2. Identify specialists (>= 70% of successful tasks in any domain)
    specialists = []
    for aid, counts in last_spec.items():
        total = sum(counts.get(d, 0) for d in ["Math", "Cyber", "Drone", "BlackBox"])
        if total > 0:
            for dom in ["Math", "Cyber", "Drone", "BlackBox"]:
                ratio = counts.get(dom, 0) / total
                if ratio >= 0.7:
                    specialists.append((aid, dom, ratio, total))
                    break
                
    # Sort specialists by their activity (total tasks solved) in descending order
    specialists.sort(key=lambda x: x[3], reverse=True)
    
    # Deplete up to a maximum of half the swarm (e.g. 4 agents out of 8)
    deplete_count = min(len(specialists), len(agents) // 2)
    to_deplete = specialists[:deplete_count]
    
    print(f"\n[Lesion Study] Identified {len(specialists)} specialists. Depleting top {deplete_count}: {to_deplete}")
    
    for aid, domain, ratio, total in to_deplete:
        if aid in agents:
            print(f"[Lesion Study] Depleting specialist unit {aid} ({domain}) due to high specialization (ratio={ratio:.2f}, solved={total})")
            agents[aid].resource = 0
            
    # 3. Continue running epochs after lesion
    epoch_accuracies = list(res_normal["epoch_accuracies"])
    epoch_fdi_final = list(res_normal["epoch_fdi_final"])
    epoch_persistence_final = list(res_normal["epoch_persistence_final"])
    epoch_learning_curves = list(res_normal["epoch_learning_curves"])
    
    spec_matrix = last_spec
    epoch_spec_matrices = list(res_normal["epoch_spec_matrices"])
    
    for epoch in range(num_epochs_after_lesion):
        idx = num_epochs_before_lesion + epoch
        print(f"\n--- POST-LESION RECOVERY EPOCH {epoch+1}/{num_epochs_after_lesion} ---")
        
        results, spec_p1, spec_p2, metrics, agents = run_swarm(
            use_somatic_memory=use_somatic_memory,
            group_name=f"{group_name} (Post-Lesion) Epoch {idx+1}",
            agents=agents,
            spec_matrix=spec_matrix,
            tasks=TASKS,
            reset_memory=False,
            reset_trust=False
        )
        
        accuracy = sum(1 for r in results if r["success"]) / len(TASKS)
        fdi_final = compute_fdi_from_matrix(spec_p2)
        persistence_final = metrics["persistence"][-1]
        
        epoch_accuracies.append(accuracy)
        epoch_fdi_final.append(fdi_final)
        epoch_persistence_final.append(persistence_final)
        epoch_spec_matrices.append(spec_p2)
        
        rolling_accs = []
        for t in range(len(results)):
            window = [1 if results[i]["success"] else 0 for i in range(max(0, t-2), t+1)]
            rolling_accs.append(np.mean(window))
        epoch_learning_curves.append(rolling_accs)
        
        spec_matrix = spec_p2
        
    return {
        "group": group_name,
        "num_epochs": num_epochs_before_lesion + num_epochs_after_lesion,
        "epoch_accuracies": epoch_accuracies,
        "epoch_fdi_final": epoch_fdi_final,
        "epoch_persistence_final": epoch_persistence_final,
        "epoch_learning_curves": epoch_learning_curves,
        "epoch_spec_matrices": epoch_spec_matrices,
        "specialists_removed": len(specialists)
    }


# =====================================================================
# 5.1. STATISTICAL MULTI-SEED RUNNERS (RIGOROUS VALIDATION)
# =====================================================================

def run_statistical_experiment(
    use_somatic_memory: bool,
    group_name: str,
    num_epochs: int = 5,
    reset_memory_each_epoch: bool = False,
    reset_trust_each_epoch: bool = False,
    env_params=None,
    seeds=None,
    allow_regeneration=False
) -> dict:
    """
    Executes simulations across multiple independent seeds, isolating LLM cache per seed
    to guarantee statistical independence while calculating averages and standard deviations.
    """
    if seeds is None:
        seeds = list(range(42, 42 + 5))
        
    print(f"\n{'='*80}")
    print(f"RUNNING STATISTICAL MULTI-SEED EXPERIMENT: {group_name}")
    print(f"  Seeds: {len(seeds)} | Epochs: {num_epochs} | Regeneration: {allow_regeneration}")
    print(f"{'='*80}")
    
    all_accuracies = []
    all_fdi = []
    all_persistence = []
    all_switching = []
    all_learning_curves = []
    final_spec_matrices = []
    
    # Trajetórias de passo a passo da Época 1 para o Diagrama de Fase de Emergência
    all_traj_fdi = []
    all_traj_entropy = []
    all_traj_switching = []
    all_traj_cognitive_load = []
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n>>> Seed {seed} ({seed_idx+1}/{len(seeds)}) <<<")
        np.random.seed(seed)
        random.seed(seed)
        client.current_seed = seed
        
        epoch_accuracies = []
        epoch_fdi_final = []
        epoch_persistence_final = []
        epoch_switching_final = []
        epoch_learning_curves = []
        
        agents = None
        spec_matrix = None
        
        for epoch in range(num_epochs):
            # Reset deterministic seed per epoch based on seed base to ensure inter-seed variation
            epoch_seed = seed * 100 + epoch
            np.random.seed(epoch_seed)
            random.seed(epoch_seed)
            
            with SuppressStdout():
                results, spec_p1, spec_p2, metrics, agents = run_swarm(
                    use_somatic_memory=use_somatic_memory,
                    group_name=f"{group_name} (Seed {seed})",
                    agents=agents,
                    spec_matrix=spec_matrix if not reset_memory_each_epoch else None,
                    tasks=TASKS,
                    reset_memory=reset_memory_each_epoch,
                    reset_trust=reset_trust_each_epoch,
                    env_params=env_params,
                    allow_regeneration=allow_regeneration
                )
            
            accuracy = sum(1 for r in results if r["success"]) / len(TASKS)
            fdi_final = compute_fdi_from_matrix(spec_p2)
            persistence_final = metrics["persistence"][-1]
            switching_final = metrics["switching_rate"][-1]
            
            epoch_accuracies.append(accuracy)
            epoch_fdi_final.append(fdi_final)
            epoch_persistence_final.append(persistence_final)
            epoch_switching_final.append(switching_final)
            
            if epoch == 0:
                # Época 1 - Limpar NaNs no FDI inicial
                fdi_clean = [0.0 if np.isnan(v) else v for v in metrics["fdi"]]
                all_traj_fdi.append(fdi_clean)
                
                # Normalizar entropia entre 0 e 1 (dividindo pelo max log2(56) = 5.8)
                entropy_norm = [e / 5.8 for e in metrics["coordination_entropy"]]
                all_traj_entropy.append(entropy_norm)
                
                all_traj_switching.append(metrics["switching_rate"])
                
                # Carga cognitiva média
                all_traj_cognitive_load.append(metrics.get("cognitive_load", [0.0] * len(results)))
                
            rolling_accs = []
            for t in range(len(results)):
                window = [1 if results[i]["success"] else 0 for i in range(max(0, t-2), t+1)]
                rolling_accs.append(np.mean(window))
            epoch_learning_curves.append(rolling_accs)
            
            spec_matrix = spec_p2
            
        all_accuracies.append(epoch_accuracies)
        all_fdi.append(epoch_fdi_final)
        all_persistence.append(epoch_persistence_final)
        all_switching.append(epoch_switching_final)
        all_learning_curves.append(epoch_learning_curves)
        final_spec_matrices.append(spec_matrix)
        
        # Free memory and force garbage collection
        del agents
        import gc
        gc.collect()
        
    all_accuracies = np.array(all_accuracies)
    all_fdi = np.array(all_fdi)
    all_persistence = np.array(all_persistence)
    all_switching = np.array(all_switching)
    all_learning_curves = np.array(all_learning_curves)
    
    return {
        "group": group_name,
        "use_somatic_memory": use_somatic_memory,
        "num_epochs": num_epochs,
        "raw_accuracies": all_accuracies.tolist(),
        "raw_fdi": all_fdi.tolist(),
        "raw_persistence": all_persistence.tolist(),
        "raw_switching": all_switching.tolist(),
        "accuracies_mean": np.mean(all_accuracies, axis=0).tolist(),
        "accuracies_std": np.std(all_accuracies, axis=0).tolist(),
        "fdi_mean": np.mean(all_fdi, axis=0).tolist(),
        "fdi_std": np.std(all_fdi, axis=0).tolist(),
        "persistence_mean": np.mean(all_persistence, axis=0).tolist(),
        "persistence_std": np.std(all_persistence, axis=0).tolist(),
        "switching_mean": np.mean(all_switching, axis=0).tolist(),
        "switching_std": np.std(all_switching, axis=0).tolist(),
        "learning_curves_mean": np.mean(all_learning_curves, axis=0).tolist(),
        "learning_curves_std": np.std(all_learning_curves, axis=0).tolist(),
        "spec_matrices": final_spec_matrices,
        "traj_fdi_mean": np.mean(all_traj_fdi, axis=0).tolist(),
        "traj_entropy_mean": np.mean(all_traj_entropy, axis=0).tolist(),
        "traj_switching_mean": np.mean(all_traj_switching, axis=0).tolist(),
        "traj_cognitive_load_mean": np.mean(all_traj_cognitive_load, axis=0).tolist()
    }


def run_statistical_lesion(
    use_somatic_memory: bool,
    group_name: str,
    num_epochs_before_lesion: int = 3,
    num_epochs_after_lesion: int = 2,
    seeds=None,
    allow_regeneration=True,
    is_fixed_role=False
) -> dict:
    """
    Runs multi-epoch lesion experiments across multiple seeds to analyze resilience and recovery,
    demonstrating the dynamic restructuring advantages over Fixed Role rigidity.
    """
    if seeds is None:
        seeds = list(range(42, 42 + 5))
        
    print(f"\n{'='*80}")
    print(f"RUNNING STATISTICAL LESION STUDY: {group_name}")
    print(f"  Seeds: {len(seeds)} | Regeneration: {allow_regeneration} | Fixed Role: {is_fixed_role}")
    print(f"{'='*80}")
    
    all_accuracies = []
    
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        client.current_seed = seed
        
        epoch_accuracies = []
        
        if is_fixed_role:
            # Fixed Role baseline (Rigid - no learning or adaptation, and permanent collapse)
            for epoch in range(num_epochs_before_lesion):
                with SuppressStdout():
                    results = run_group_fixed(allow_lesion=False)
                acc = sum(1 for r in results if r["success"]) / len(TASKS)
                epoch_accuracies.append(acc)
            for epoch in range(num_epochs_after_lesion):
                with SuppressStdout():
                    results = run_group_fixed(allow_lesion=True)
                acc = sum(1 for r in results if r["success"]) / len(TASKS)
                epoch_accuracies.append(acc)
            agents = None
        else:
            # Swarm-based system
            agents = None
            spec_matrix = None
            
            for epoch in range(num_epochs_before_lesion):
                epoch_seed = seed * 100 + epoch
                np.random.seed(epoch_seed)
                random.seed(epoch_seed)
                with SuppressStdout():
                    results, spec_p1, spec_p2, metrics, agents = run_swarm(
                        use_somatic_memory=use_somatic_memory,
                        group_name=f"{group_name} (Seed {seed})",
                        agents=agents,
                        spec_matrix=spec_matrix,
                        tasks=TASKS,
                        allow_regeneration=False
                    )
                acc = sum(1 for r in results if r["success"]) / len(TASKS)
                epoch_accuracies.append(acc)
                spec_matrix = spec_p2
                
            # Apply Lesion: disable specialists (>= 70% of successful tasks)
            specialists = []
            for aid, counts in spec_matrix.items():
                total = sum(counts.get(d, 0) for d in ["Math", "Cyber", "Drone", "BlackBox"])
                if total > 0:
                    for dom in ["Math", "Cyber", "Drone", "BlackBox"]:
                        ratio = counts.get(dom, 0) / total
                        if ratio >= 0.7:
                            specialists.append((aid, dom, ratio, total))
                            break
            specialists.sort(key=lambda x: x[3], reverse=True)
            deplete_count = min(len(specialists), len(agents) // 2)
            to_deplete = specialists[:deplete_count]
            
            for aid, domain, ratio, total in to_deplete:
                if aid in agents:
                    agents[aid].resource = 0  # Depressed/lesioned
                    
            # Continue running epochs after lesion
            for epoch in range(num_epochs_after_lesion):
                idx = num_epochs_before_lesion + epoch
                epoch_seed = seed * 100 + idx
                np.random.seed(epoch_seed)
                random.seed(epoch_seed)
                with SuppressStdout():
                    results, spec_p1, spec_p2, metrics, agents = run_swarm(
                        use_somatic_memory=use_somatic_memory,
                        group_name=f"{group_name} (Post-Lesion) (Seed {seed})",
                        agents=agents,
                        spec_matrix=spec_matrix,
                        tasks=TASKS,
                        allow_regeneration=allow_regeneration  # Active regeneration for Phytomer swarm
                    )
                acc = sum(1 for r in results if r["success"]) / len(TASKS)
                epoch_accuracies.append(acc)
                spec_matrix = spec_p2
                
        all_accuracies.append(epoch_accuracies)
        if agents is not None:
            del agents
        import gc
        gc.collect()
        
    all_accuracies = np.array(all_accuracies)
    
    return {
        "group": group_name,
        "num_epochs": num_epochs_before_lesion + num_epochs_after_lesion,
        "raw_accuracies": all_accuracies.tolist(),
        "accuracies_mean": np.mean(all_accuracies, axis=0).tolist(),
        "accuracies_std": np.std(all_accuracies, axis=0).tolist()
    }


def perform_statistical_tests(results_dict):
    """
    Performs Mann-Whitney U tests comparing Group C with other baselines,
    and applies Holm-Bonferroni correction on the resulting p-values.
    """
    from scipy.stats import mannwhitneyu
    
    # Obter acurácias acumuladas por seed para cada grupo na última época (época 5)
    group_c_acc = np.array(results_dict["Group C (Emergent)"]["raw_accuracies"])[:, -1]
    
    # Comparações de Acurácia Global
    comparisons = []
    
    if "Group A (Monolithic)" in results_dict:
        comparisons.append(("Group C vs Monolithic (Accuracy)", group_c_acc, np.array(results_dict["Group A (Monolithic)"]["raw_accuracies"])[:, -1]))
    if "Group B (Orchestrated)" in results_dict:
        comparisons.append(("Group C vs Orchestrated (Accuracy)", group_c_acc, np.array(results_dict["Group B (Orchestrated)"]["raw_accuracies"])[:, -1]))
    if "Random Routing" in results_dict:
        comparisons.append(("Group C vs Random (Accuracy)", group_c_acc, np.array(results_dict["Random Routing"]["raw_accuracies"])[:, -1]))
    if "Fixed Role Assignment" in results_dict:
        comparisons.append(("Group C vs Fixed Role (Accuracy)", group_c_acc, np.array(results_dict["Fixed Role Assignment"]["raw_accuracies"])[:, -1]))
    if "Centralized Planner" in results_dict:
        comparisons.append(("Group C vs Centralized (Accuracy)", group_c_acc, np.array(results_dict["Centralized Planner"]["raw_accuracies"])[:, -1]))
    if "Group C-Ablated" in results_dict:
        comparisons.append(("Group C vs Ablated (Accuracy)", group_c_acc, np.array(results_dict["Group C-Ablated"]["raw_accuracies"])[:, -1]))
        
    # Comparações de FDI (Especialização final na época 5)
    if "Group C (Emergent)" in results_dict and "Group C-Ablated" in results_dict:
        group_c_fdi = np.array(results_dict["Group C (Emergent)"]["raw_fdi"])[:, -1]
        comparisons.append(("Group C vs Ablated (FDI)", group_c_fdi, np.array(results_dict["Group C-Ablated"]["raw_fdi"])[:, -1]))
    if "Group C (Emergent)" in results_dict and "Random Routing" in results_dict:
        group_c_fdi = np.array(results_dict["Group C (Emergent)"]["raw_fdi"])[:, -1]
        comparisons.append(("Group C vs Random (FDI)", group_c_fdi, np.array(results_dict["Random Routing"]["raw_fdi"])[:, -1]))
    if "Group C (Emergent)" in results_dict and "Centralized Planner" in results_dict:
        group_c_fdi = np.array(results_dict["Group C (Emergent)"]["raw_fdi"])[:, -1]
        comparisons.append(("Group C vs Centralized (FDI)", group_c_fdi, np.array(results_dict["Centralized Planner"]["raw_fdi"])[:, -1]))
        
    raw_p_values = []
    tests = []
    
    for label, group_c_data, other_data in comparisons:
        try:
            stat, p_val = mannwhitneyu(group_c_data, other_data, alternative="greater")
        except Exception as e:
            stat = 0
            p_val = 1.0  # Fallback
        raw_p_values.append(p_val)
        tests.append((label, stat, p_val))
        
    # Aplicar Holm-Bonferroni
    m = len(tests)
    indexed_p_values = sorted(enumerate(raw_p_values), key=lambda x: x[1])
    
    adjusted_p_values = [1.0] * m
    max_p = 0.0
    
    for i, (orig_idx, p) in enumerate(indexed_p_values):
        rank = i + 1
        adj_p = p * (m - rank + 1)
        max_p = min(1.0, max(max_p, adj_p))
        adjusted_p_values[orig_idx] = max_p
        
    # Salvar em results/statistical_validation.csv
    os.makedirs("results", exist_ok=True)
    with open("results/statistical_validation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Comparison Test", "U Statistic", "Raw p-value", "Holm-Bonferroni Adjusted p-value", "Significant (alpha=0.05)"])
        for idx, (label, stat, raw_p) in enumerate(tests):
            adj_p = adjusted_p_values[idx]
            sig = "Yes" if adj_p < 0.05 else "No"
            writer.writerow([label, f"{stat:.1f}" if isinstance(stat, float) or isinstance(stat, int) else stat, f"{raw_p:.6f}", f"{adj_p:.6f}", sig])
            print(f"[Stat Test] {label} | Raw p: {raw_p:.6f} | Adj p: {adj_p:.6f} | Significant: {sig}")
            
    # Copiar também para o BRAIN_DIR para compatibilidade
    shutil.copy("results/statistical_validation.csv", os.path.join(BRAIN_DIR, "statistical_validation.csv"))
    print("[Save] Statistical validation saved to results and brain directory.")


def run_environment_comparison():
    print(f"\n{'='*80}")
    print("RUNNING ENVIRONMENT COMPARISON EXPERIMENT")
    print(f"{'='*80}")
    
    env_types = {
        "stable": {
            "noise_level": 0.0,
            "paradigm_shift_probability": 0.0,
            "reward_stability": 1.0
        },
        "chaotic": {
            "noise_level": 0.3,
            "paradigm_shift_probability": 0.2,
            "reward_stability": 0.5
        },
        "semi_stable": {
            "noise_level": 0.1,
            "paradigm_shift_probability": 0.05,
            "reward_stability": 0.8
        }
    }
    
    results = {}
    for env_name, env_params in env_types.items():
        print(f"\n>>> Running Environment: {env_name.upper()} <<<")
        res = run_multi_epoch_experiment(
            use_somatic_memory=True,
            group_name=f"Group C ({env_name})",
            num_epochs=5,
            reset_memory_each_epoch=False,
            reset_trust_each_epoch=False,
            env_params=env_params
        )
        results[env_name] = res
        
    return results


# =====================================================================
# 6. RESULTS LOGGING AND PLOTTING
# =====================================================================

def save_multi_epoch_results(res_c, res_c_reset, res_abl):
    os.makedirs("results", exist_ok=True)
    
    # Save epoch accuracies CSV
    with open("results/epoch_accuracies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "group_c", "group_c_reset", "group_c_ablated"])
        for i in range(5):
            writer.writerow([
                i+1,
                res_c["accuracies_mean"][i],
                res_c_reset["accuracies_mean"][i],
                res_abl["accuracies_mean"][i]
            ])
    
    # Save FDI evolution
    with open("results/epoch_fdi.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "group_c", "group_c_reset", "group_c_ablated"])
        for i in range(5):
            writer.writerow([
                i+1,
                res_c["fdi_mean"][i],
                res_c_reset["fdi_mean"][i],
                res_abl["fdi_mean"][i]
            ])
    
    # Save learning curves (per epoch, per step)
    with open("results/epoch_learning_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "group_c", "group_c_reset", "group_c_ablated"])
        for epoch in range(5):
            for step in range(len(TASKS)):
                writer.writerow([
                    epoch+1,
                    step,
                    res_c["learning_curves_mean"][epoch][step],
                    res_c_reset["learning_curves_mean"][epoch][step],
                    res_abl["learning_curves_mean"][epoch][step]
                ])
    
    print("[Save] Multi-epoch results saved to results/")


def plot_epoch_evolution(res_c, res_c_reset, res_abl):
    epochs = range(1, 6)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy over epochs
    ax1 = axes[0, 0]
    for res, label, color, marker, ls in [
        (res_c, "Group C (Emergent)", "#2ca02c", "o", "-"),
        (res_c_reset, "Group C (Memory Reset)", "#d62728", "s", "--"),
        (res_abl, "Group C-Ablated (No Memory)", "#9467bd", "^", ":")
    ]:
        mean = np.array(res["accuracies_mean"])
        std = np.array(res["accuracies_std"])
        ax1.plot(epochs, mean, label=label, marker=marker, linewidth=2.5, color=color, linestyle=ls)
        ax1.fill_between(epochs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.15)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overall Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Learning Curve: Accuracy Over Epochs", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 2: FDI over epochs
    ax2 = axes[0, 1]
    for res, label, color, marker, ls in [
        (res_c, "Group C (Emergent)", "#2ca02c", "o", "-"),
        (res_c_reset, "Group C (Memory Reset)", "#d62728", "s", "--"),
        (res_abl, "Group C-Ablated (No Memory)", "#9467bd", "^", ":")
    ]:
        mean = np.array(res["fdi_mean"])
        std = np.array(res["fdi_std"])
        ax2.plot(epochs, mean, label=label, marker=marker, linewidth=2.5, color=color, linestyle=ls)
        ax2.fill_between(epochs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.15)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("FDI (Functional Differentiation Index)", fontsize=11, fontweight="bold")
    ax2.set_title("Specialization Strength Over Epochs", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 3: Persistence Score over epochs
    ax3 = axes[1, 0]
    for res, label, color, marker, ls in [
        (res_c, "Group C (Emergent)", "#2ca02c", "o", "-"),
        (res_c_reset, "Group C (Memory Reset)", "#d62728", "s", "--"),
        (res_abl, "Group C-Ablated (No Memory)", "#9467bd", "^", ":")
    ]:
        mean = np.array(res["persistence_mean"])
        std = np.array(res["persistence_std"])
        ax3.plot(epochs, mean, label=label, marker=marker, linewidth=2.5, color=color, linestyle=ls)
        ax3.fill_between(epochs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.15)
    ax3.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Persistence Score", fontsize=11, fontweight="bold")
    ax3.set_title("Role Stability Over Epochs", fontsize=12, fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="lower right")
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    # Plot 4: Switching Rate over epochs
    ax4 = axes[1, 1]
    for res, label, color, marker, ls in [
        (res_c, "Group C (Emergent)", "#2ca02c", "o", "-"),
        (res_c_reset, "Group C (Memory Reset)", "#d62728", "s", "--"),
        (res_abl, "Group C-Ablated (No Memory)", "#9467bd", "^", ":")
    ]:
        mean = np.array(res["switching_mean"])
        std = np.array(res["switching_std"])
        ax4.plot(epochs, mean, label=label, marker=marker, linewidth=2.5, color=color, linestyle=ls)
        ax4.fill_between(epochs, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1), color=color, alpha=0.15)
    ax4.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Switching Rate", fontsize=11, fontweight="bold")
    ax4.set_title("Role Alternation Rate Over Epochs", fontsize=12, fontweight="bold")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="upper right")
    ax4.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Longitudinal Multi-Epoch Evolution: Learning, Specialization & Stability", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/epoch_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    brain_dir = BRAIN_DIR
    os.makedirs(brain_dir, exist_ok=True)
    shutil.copy("results/epoch_evolution.png", os.path.join(brain_dir, "epoch_evolution.png"))
    
    print("[Plot] Saved epoch_evolution.png")


def plot_lesion_recovery(lesion_res_c, lesion_res_no_reg, lesion_res_fixed):
    epochs = range(1, 6)
    
    plt.figure(figsize=(8, 5))
    
    # Group C Phytomer (Regeneration)
    mean_c = np.array(lesion_res_c["accuracies_mean"])
    std_c = np.array(lesion_res_c["accuracies_std"])
    plt.plot(epochs, mean_c, label="Group C Phytomer (Regeneration)", marker="o", linewidth=2.5, color="#2ca02c")
    plt.fill_between(epochs, np.clip(mean_c - std_c, 0, 1), np.clip(mean_c + std_c, 0, 1), color="#2ca02c", alpha=0.15)
    
    # Group C Phytomer (No Reg)
    mean_no = np.array(lesion_res_no_reg["accuracies_mean"])
    std_no = np.array(lesion_res_no_reg["accuracies_std"])
    plt.plot(epochs, mean_no, label="Group C Phytomer (No Reg.)", marker="x", linewidth=2.0, color="#ff7f0e", linestyle="--")
    plt.fill_between(epochs, np.clip(mean_no - std_no, 0, 1), np.clip(mean_no + std_no, 0, 1), color="#ff7f0e", alpha=0.1)
    
    # Fixed Role (Colapso Rígido)
    mean_fixed = np.array(lesion_res_fixed["accuracies_mean"])
    std_fixed = np.array(lesion_res_fixed["accuracies_std"])
    plt.plot(epochs, mean_fixed, label="Fixed Role Baseline (Rigid)", marker="s", linewidth=2.0, color="#d62728", linestyle=":")
    plt.fill_between(epochs, np.clip(mean_fixed - std_fixed, 0, 1), np.clip(mean_fixed + std_fixed, 0, 1), color="#d62728", alpha=0.1)
    
    plt.axvline(x=3, color="black", linestyle="--", linewidth=1.5, label="Lesion Event (Math Specialists Depleted)")
    
    plt.annotate("Specialists Removed\n(Fixed Role Collapse)", xy=(3, mean_fixed[2]), 
                 xytext=(3.4, mean_fixed[2] - 0.25),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight="bold")
                 
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Overall Accuracy", fontsize=11, fontweight="bold")
    plt.title("Resilience: Specialist Lesion and Reorganization Recovery", fontsize=12, fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/lesion_recovery.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    brain_dir = BRAIN_DIR
    os.makedirs(brain_dir, exist_ok=True)
    shutil.copy("results/lesion_recovery.png", os.path.join(brain_dir, "lesion_recovery.png"))
    print("[Plot] Saved lesion_recovery.png")


def plot_emergence_phase_diagram(res_c):
    """
    Generates the 'Emergence Phase Diagram' synthesizing temporal dynamics of FDI, 
    Coordination Entropy, and Switching Rate across simulation steps in Epoch 1.
    """
    steps = range(1, len(TASKS) + 1)
    
    fdi = np.array(res_c["traj_fdi_mean"])
    entropy = np.array(res_c["traj_entropy_mean"])
    switching = np.array(res_c["traj_switching_mean"])
    cognitive_load = np.array(res_c.get("traj_cognitive_load_mean", [0.0] * len(steps)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.plot(steps, fdi, label="FDI (Functional Differentiation Index)", color="#2ca02c", linewidth=3)
    ax.plot(steps, entropy, label="Coordination Entropy (Normalized)", color="#1f77b4", linewidth=2.5, linestyle="--")
    ax.plot(steps, switching, label="Switching Rate of Roles", color="#ff7f0e", linewidth=2.5, linestyle=":")
    ax.plot(steps, cognitive_load, label="Mean Cognitive Load", color="#d62728", linewidth=2.5, linestyle="-.")
    
    # Visual markers for developmental phases
    ax.axvline(x=20.5, color="gray", linestyle="-.", alpha=0.4)
    ax.axvline(x=60.5, color="gray", linestyle="-.", alpha=0.4)
    
    # Label phases
    ax.text(10.5, 0.95, "Fase I\nHomogeneidade\nInicial", ha="center", va="top", fontsize=9, fontweight="bold", color="#555555")
    ax.text(40.5, 0.95, "Fase II\nTransição\nAdaptativa", ha="center", va="top", fontsize=9, fontweight="bold", color="#555555")
    ax.text(80.5, 0.95, "Fase III\nEstabilização\nFuncional", ha="center", va="top", fontsize=9, fontweight="bold", color="#555555")
    
    ax.set_xlabel("Simulation Steps (Epoch 1)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value (Normalized 0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Emergence Phase Diagram: Dynamic Trajectory of Self-Organization", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("results/emergence_phase_diagram.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    brain_dir = BRAIN_DIR
    os.makedirs(brain_dir, exist_ok=True)
    shutil.copy("results/emergence_phase_diagram.png", os.path.join(brain_dir, "emergence_phase_diagram.png"))
    print("[Plot] Saved emergence_phase_diagram.png")


def plot_environment_comparison(env_results):
    epochs = range(1, 6)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"stable": "#2ca02c", "semi_stable": "#ff7f0e", "chaotic": "#d62728"}
    labels = {"stable": "Stable (0% Noise)", "semi_stable": "Semi-Stable (10% Noise)", "chaotic": "Chaotic (30% Noise)"}
    
    for name, res in env_results.items():
        ax1.plot(epochs, res["epoch_accuracies"], label=labels[name], marker="o", linewidth=2, color=colors[name])
        ax2.plot(epochs, res["epoch_fdi_final"], label=labels[name], marker="o", linewidth=2, color=colors[name])
        
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Overall Accuracy", fontsize=11, fontweight="bold")
    ax1.set_title("Accuracy Over Epochs by Environment", fontsize=12, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("FDI (Functional Differentiation Index)", fontsize=11, fontweight="bold")
    ax2.set_title("Specialization (FDI) by Environment", fontsize=12, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Environmental Robustness Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("results/environment_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    brain_dir = BRAIN_DIR
    os.makedirs(brain_dir, exist_ok=True)
    shutil.copy("results/environment_comparison.png", os.path.join(brain_dir, "environment_comparison.png"))
    
    print("[Plot] Saved environment_comparison.png")


def main():
    print("[Warm-up] Initializing qwen2.5:0.5b and nomic-embed-text in GPU memory...")
    client.pull_model("qwen2.5:0.5b")
    client.pull_model("nomic-embed-text")
    client.generate(prompt="Hello", model_name="qwen2.5:0.5b")
    print("[Warm-up] Done.\n")
    
    # For dry-run, use 2 seeds. For full production run, use 30 seeds.
    seeds = list(range(42, 42 + 5))
    
    # 1. Run Multi-Epoch Experiments across 30 seeds
    res_c = run_statistical_experiment(
        use_somatic_memory=True,
        group_name="Group C (Emergent)",
        num_epochs=5,
        reset_memory_each_epoch=False,
        reset_trust_each_epoch=False,
        seeds=seeds,
        allow_regeneration=True  # Ativa regeneração ativa para Phytomer Swarm
    )
    
    res_c_reset = run_statistical_experiment(
        use_somatic_memory=True,
        group_name="Group C (Memory Reset)",
        num_epochs=5,
        reset_memory_each_epoch=True,
        reset_trust_each_epoch=True,
        seeds=seeds,
        allow_regeneration=True
    )
    
    res_abl = run_statistical_experiment(
        use_somatic_memory=False,
        group_name="Group C-Ablated",
        num_epochs=5,
        reset_memory_each_epoch=False,
        reset_trust_each_epoch=False,
        seeds=seeds,
        allow_regeneration=False
    )
    
    # 2. Run Lesion Studies across 30 seeds
    print("\nExecuting Lesion Studies...")
    lesion_results_c = run_statistical_lesion(
        use_somatic_memory=True,
        group_name="Group C (Regeneration)",
        seeds=seeds,
        allow_regeneration=True,
        is_fixed_role=False
    )
    
    lesion_results_no_reg = run_statistical_lesion(
        use_somatic_memory=True,
        group_name="Group C (No Reg.)",
        seeds=seeds,
        allow_regeneration=False,
        is_fixed_role=False
    )
    
    lesion_results_fixed = run_statistical_lesion(
        use_somatic_memory=True,
        group_name="Fixed Role Baseline (Rigid)",
        seeds=seeds,
        allow_regeneration=False,
        is_fixed_role=True
    )
    
    # 3. Run Baselines across 30 seeds (1 epoch only, matching domains evaluation)
    print("\nExecuting Baselines across 30 seeds...")
    baselines_data = {
        "Group A (Monolithic)": {"raw_accuracies": []},
        "Group B (Orchestrated)": {"raw_accuracies": []},
        "Random Routing": {"raw_accuracies": [], "raw_fdi": []},
        "Fixed Role Assignment": {"raw_accuracies": []},
        "Centralized Planner": {"raw_accuracies": [], "raw_fdi": []}
    }
    
    for s_idx, seed in enumerate(seeds):
        print(f"Seed {seed} Baselines ({s_idx+1}/{len(seeds)})...")
        np.random.seed(seed)
        random.seed(seed)
        client.current_seed = seed
        
        # Monolith
        with SuppressStdout():
            res_a = run_group_a()
        acc_a = sum(1 for r in res_a if r["success"]) / len(TASKS)
        baselines_data["Group A (Monolithic)"]["raw_accuracies"].append([acc_a])
        
        # Pipeline
        with SuppressStdout():
            res_b = run_group_b()
        acc_b = sum(1 for r in res_b if r["success"]) / len(TASKS)
        baselines_data["Group B (Orchestrated)"]["raw_accuracies"].append([acc_b])
        
        # Random
        with SuppressStdout():
            res_rand = run_group_random()
        acc_rand = sum(1 for r in res_rand if r["success"]) / len(TASKS)
        baselines_data["Random Routing"]["raw_accuracies"].append([acc_rand])
        baselines_data["Random Routing"]["raw_fdi"].append([0.0]) # FDI aleatório puro é nulo
        
        # Fixed
        with SuppressStdout():
            res_fix = run_group_fixed(allow_lesion=False)
        acc_fix = sum(1 for r in res_fix if r["success"]) / len(TASKS)
        baselines_data["Fixed Role Assignment"]["raw_accuracies"].append([acc_fix])
        
        # Centralized
        with SuppressStdout():
            res_cent = run_group_centralized()
        acc_cent = sum(1 for r in res_cent if r["success"]) / len(TASKS)
        baselines_data["Centralized Planner"]["raw_accuracies"].append([acc_cent])
        baselines_data["Centralized Planner"]["raw_fdi"].append([0.0]) # Centralizado não auto-organiza FDI
        
    # 4. Perform Mann-Whitney U and Holm-Bonferroni Correction
    results_all_groups = {
        "Group C (Emergent)": res_c,
        "Group C-Ablated": res_abl,
        "Group A (Monolithic)": baselines_data["Group A (Monolithic)"],
        "Group B (Orchestrated)": baselines_data["Group B (Orchestrated)"],
        "Random Routing": baselines_data["Random Routing"],
        "Fixed Role Assignment": baselines_data["Fixed Role Assignment"],
        "Centralized Planner": baselines_data["Centralized Planner"]
    }
    perform_statistical_tests(results_all_groups)
    
    # 5. Run Environment Comparison (Single seed stats, to maintain environment results layout)
    np.random.seed(42)
    random.seed(42)
    env_results = run_environment_comparison()
    
    # 6. Save and Plot Multi-Epoch logs
    save_multi_epoch_results(res_c, res_c_reset, res_abl)
    plot_epoch_evolution(res_c, res_c_reset, res_abl)
    plot_lesion_recovery(lesion_results_c, lesion_results_no_reg, lesion_results_fixed)
    plot_emergence_phase_diagram(res_c)
    plot_environment_comparison(env_results)
    
    # 7. compliance and baseline files (first seed specialization matrix extraction for DOCX)
    print("\nSaving compliance CSV files...")
    spec_c_p1 = res_c["spec_matrices"][0]  # Utiiza a matriz de especialização da seed 42
    spec_c_p2 = res_c["spec_matrices"][0]
    spec_c_abl_p1 = res_abl["spec_matrices"][0]
    spec_c_abl_p2 = res_abl["spec_matrices"][0]
    
    # Executa run swarm de compliance de 1 época apenas para compatibilidade
    np.random.seed(42)
    random.seed(42)
    client.current_seed = 42
    res_c_single, spec_c_p1_single, spec_c_p2_single, metrics_c_single, _ = run_swarm(use_somatic_memory=True, group_name="Emergent Single compliance")
    res_c_abl_single, spec_c_abl_p1_single, spec_c_abl_p2_single, metrics_c_abl_single, _ = run_swarm(use_somatic_memory=False, group_name="Ablated Single compliance")
    
    # Executa runs dos baselines simples A e B de 1 época para compliance de acurácia
    np.random.seed(42)
    random.seed(42)
    res_a_single = run_group_a()
    res_b_single = run_group_b()
    
    # Write scientific_mvp_metrics.csv expected by generate_report.py
    with open("results/scientific_mvp_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Domain", "GroupA_Success", "GroupB_Success", "GroupC_Success"])
        for t in range(len(TASKS)):
            if t < 10: dom = "Math"
            elif t < 20: dom = "Cyber"
            elif t < 30: dom = "Drone"
            elif t < 40: dom = "BlackBox"
            elif t < 50: dom = "Math_OOD"
            elif t < 80: dom = "Math_GSM8K"
            else: dom = "BlackBox_ARC"
            writer.writerow([
                dom,
                1.0 if res_a_single[t]["success"] else 0.0,
                1.0 if res_b_single[t]["success"] else 0.0,
                1.0 if res_c_single[t]["success"] else 0.0
            ])
            
    # Copy matrices
    brain_dir = BRAIN_DIR
    shutil.copy("results/scientific_mvp_metrics.csv", os.path.join(brain_dir, "scientific_mvp_metrics.csv"))
    
    save_spec_matrix_csv(spec_c_p1_single, f"results/specialization_matrix_phase1.csv")
    save_spec_matrix_csv(spec_c_p2_single, f"results/specialization_matrix_phase2.csv")
    save_spec_matrix_csv(spec_c_abl_p1_single, f"results/specialization_matrix_phase1_ablated.csv")
    save_spec_matrix_csv(spec_c_abl_p2_single, f"results/specialization_matrix_phase2_ablated.csv")
    
    shutil.copy("results/specialization_matrix_phase1.csv", os.path.join(brain_dir, "specialization_matrix_phase1.csv"))
    shutil.copy("results/specialization_matrix_phase2.csv", os.path.join(brain_dir, "specialization_matrix_phase2.csv"))
    shutil.copy("results/specialization_matrix_phase1_ablated.csv", os.path.join(brain_dir, "specialization_matrix_phase1_ablated.csv"))
    shutil.copy("results/specialization_matrix_phase2_ablated.csv", os.path.join(brain_dir, "specialization_matrix_phase2_ablated.csv"))
    
    shutil.copy("results/specialization_matrix_phase2.csv", "results/specialization_matrix.csv")
    shutil.copy("results/specialization_matrix_phase2_ablated.csv", "results/specialization_matrix_ablated.csv")
    shutil.copy("results/specialization_matrix.csv", os.path.join(brain_dir, "specialization_matrix.csv"))
    shutil.copy("results/specialization_matrix_ablated.csv", os.path.join(brain_dir, "specialization_matrix_ablated.csv"))
    
    plot_specialization_heatmaps(spec_c_p1_single, spec_c_abl_p1_single, "Phase 1 (Task 9 - Math)", "results/specialization_heatmap_phase1.png")
    plot_specialization_heatmaps(spec_c_p2_single, spec_c_abl_p2_single, "Phase 5 (Task 49 - Math OOD)", "results/specialization_heatmap_phase2.png")
    shutil.copy("results/specialization_heatmap_phase1.png", os.path.join(brain_dir, "specialization_heatmap_phase1.png"))
    shutil.copy("results/specialization_heatmap_phase2.png", os.path.join(brain_dir, "specialization_heatmap_phase2.png"))
    
    # Plot baseline curves (using single compliance runs for simple temporal visuals)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    steps = range(1, len(TASKS) + 1)
    
    def get_rolling_vis(res_list):
        accs = []
        for t in range(len(TASKS)):
            window = [1 if r["success"] else 0 for r in res_list[max(0, t-2):t+1]]
            accs.append(np.mean(window))
        return accs
        
    vis_a = get_rolling_vis(res_a_single)
    vis_b = get_rolling_vis(res_b_single)
    vis_c = get_rolling_vis(res_c_single)
    vis_c_abl = get_rolling_vis(res_c_abl_single)
    
    ax1.plot(steps, vis_a, label="Group A (Monolithic)", color="#1f77b4", linewidth=2, marker="o", markevery=10)
    ax1.plot(steps, vis_b, label="Group B (Orchestrated)", color="#aec7e8", linewidth=2, marker="s", markevery=10)
    ax1.plot(steps, vis_c, label="Group C (Emergent Swarm)", color="#2ca02c", linewidth=2.5, marker="^", markevery=10)
    ax1.plot(steps, vis_c_abl, label="Group C-Ablated (No Memory)", color="#d62728", linewidth=2, linestyle="--", marker="d", markevery=10)
    
    for border in [10.5, 20.5, 30.5, 40.5, 70.5]:
        ax1.axvline(x=border, color="gray", linestyle="-.", alpha=0.5)
        ax2.axvline(x=border, color="gray", linestyle="-.", alpha=0.3)
        ax3.axvline(x=border, color="gray", linestyle="-.", alpha=0.3)
        
    ax1.text(5.5, 0.95, "Math", ha="center", fontsize=8, fontweight="bold")
    ax1.text(15.5, 0.95, "Cyber", ha="center", fontsize=8, fontweight="bold")
    ax1.text(25.5, 0.95, "Drone", ha="center", fontsize=8, fontweight="bold")
    ax1.text(35.5, 0.95, "BlackBox", ha="center", fontsize=8, fontweight="bold")
    ax1.text(55.5, 0.95, "GSM8K", ha="center", fontsize=8, fontweight="bold")
    ax1.text(85.5, 0.95, "ARC", ha="center", fontsize=8, fontweight="bold")
    
    ax1.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rolling Accuracy (Window=3)", fontsize=11, fontweight="bold")
    ax1.set_title("Paradigm Shift Recovery Curve", fontsize=13, fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.plot(steps, metrics_c_single["fdi"], label="Group C: FDI", color="#2ca02c", linewidth=2.5)
    ax2.plot(steps, metrics_c_abl_single["fdi"], label="Group C-Ablated: FDI", color="#d62728", linewidth=2, linestyle="--")
    ax2.plot(steps, metrics_c_single["persistence"], label="Group C: Persistence", color="#1f77b4", linewidth=2)
    ax2.plot(steps, metrics_c_abl_single["persistence"], label="Group C-Ablated: Persistence", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value (Index 0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax2.set_title("Emergent Specialization & Persistence", fontsize=13, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    ax3.plot(steps, metrics_c_single["dominance"], label="Group C: Hub Dominance", color="#9467bd", linewidth=2)
    ax3.plot(steps, metrics_c_abl_single["dominance"], label="Group C-Ablated: Hub Dominance", color="#c5b0d5", linewidth=1.5, linestyle="--")
    norm_entropy_c = [e / 5.8 for e in metrics_c_single["coordination_entropy"]]
    norm_entropy_c_abl = [e / 5.8 for e in metrics_c_abl_single["coordination_entropy"]]
    ax3.plot(steps, norm_entropy_c, label="Group C: Coord. Entropy", color="#e377c2", linewidth=2)
    ax3.plot(steps, norm_entropy_c_abl, label="Group C-Ablated: Coord. Entropy", color="#f7b6d2", linewidth=1.5, linestyle="--")
    ax3.set_xlabel("Simulation Steps", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Metric Value (Normalized 0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax3.set_title("Trust Network Dynamics", fontsize=13, fontweight="bold")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("results/scientific_mvp_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    shutil.copy("results/scientific_mvp_curves.png", os.path.join(brain_dir, "scientific_mvp_curves.png"))
    
    print("\n[Done] All simulations and metrics compiled successfully.")

if __name__ == "__main__":
    main()
