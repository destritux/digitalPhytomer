import random
import time
import json
import matplotlib.pyplot as plt
import numpy as np

from ollama_client import OllamaClient
from vector_store import SomaticVectorStore
from micro_agent import MicroAgent
from cognitive_strategy import ReasoningParadigm, STRATEGY_PROMPTS
from cognitive_verifier import CognitiveVerifier
from cognitive_events import CognitiveEventBus, TelemetryManager

# Initialize central client
client = OllamaClient()
event_bus = CognitiveEventBus()

class ContinuousSimulation:
    def __init__(self, steps=25):
        self.steps = steps
        self.somatic_memory = SomaticVectorStore()
        self.verifier = CognitiveVerifier()
        self.agents = {}
        self.creation_count = 0
        self.destruction_count = 0
        
        # Telemetry metrics over steps
        self.history_diversity = []
        self.history_energy = []
        self.history_delegations = []
        self.history_success = []
        
        # Track delegation network
        self.delegation_count = 0
        self.success_count = 0
        
        # Initialize generic population
        self._initialize_population()

    def _initialize_population(self):
        # We start with 4 generic cells, each with a different strategy genome
        strategies = [
            ReasoningParadigm.SYMBOLIC,
            ReasoningParadigm.ADVERSARIAL,
            ReasoningParadigm.DECOMPOSITION,
            ReasoningParadigm.ANALOGICAL
        ]
        for idx, strat in enumerate(strategies):
            role = f"generic_cell_{idx+1}"
            sys_prompt = (
                f"You are a generic cell in an adaptive cognitive swarm. Your identifier is '{role}'.\n"
                f"Solve the objective using your reasoning strategy. Focus on correctness and formatting.\n"
                f"Output only your final answer, or write clear code if the strategy requires it."
            )
            agent_id = f"Cell-{role}"
            agent = MicroAgent(
                agent_id=agent_id,
                role="generic_cell",
                system_prompt=sys_prompt,
                client=client,
                somatic_memory=self.somatic_memory
            )
            agent.strategy = strat
            self.agents[agent_id] = agent
            self.creation_count += 1
        print(f"[Simulation Init] Spawned 4 generic micro-agents with initial strategy genomes.")

    def run_evolutionary_cycle(self):
        """Natural selection based on fitness and diversity pressure."""
        active_agents = list(self.agents.values())
        if len(active_agents) < 2:
            return

        # 1. Strategy distribution for diversity penalty
        strat_dist = {}
        for agent in active_agents:
            strat = getattr(agent, "strategy", "default")
            strat_dist[strat] = strat_dist.get(strat, 0) + 1

        def calculate_fitness(agent):
            accuracy = 1.0 / (agent.failures_count + 1)
            energy_efficiency = agent.energy / agent.max_energy
            
            strat = getattr(agent, "strategy", "default")
            strategy_ratio = strat_dist.get(strat, 1) / max(1, len(active_agents))
            diversity_bonus = 1.0 - strategy_ratio
            
            return 0.5 * accuracy + 0.3 * energy_efficiency + 0.2 * diversity_bonus

        active_agents.sort(key=calculate_fitness, reverse=True)

        # Pruning weak agents
        prune_cutoff = max(1, int(len(active_agents) * 0.25))
        for weak_agent in active_agents[-prune_cutoff:]:
            if weak_agent.energy < 40 and len(self.agents) > 2:
                print(f"  [Selection] Pruned weak agent {weak_agent.agent_id} due to low fitness / energy.")
                if weak_agent.agent_id in self.agents:
                    del self.agents[weak_agent.agent_id]
                    self.destruction_count += 1

        # Replication of elite agents with strategy mutation
        reproduce_cutoff = max(1, int(len(active_agents) * 0.25))
        for elite_agent in active_agents[:reproduce_cutoff]:
            if elite_agent.energy > 120 and len(self.agents) < 8:
                child_id = f"Cell-replica-{random.randint(100, 999)}"
                print(f"  [Replication] Elite agent {elite_agent.agent_id} replicates to child {child_id}.")
                
                child = MicroAgent(
                    agent_id=child_id,
                    role="generic_cell",
                    system_prompt=elite_agent.system_prompt,
                    client=client,
                    somatic_memory=self.somatic_memory
                )
                child.strategy = getattr(elite_agent, "strategy", "default")
                
                # Crossover/Mutation
                operators = ["mutate_strategy", "none"]
                op = random.choice(operators)
                if op == "mutate_strategy":
                    child.strategy = random.choice([
                        ReasoningParadigm.SYMBOLIC,
                        ReasoningParadigm.ADVERSARIAL,
                        ReasoningParadigm.DECOMPOSITION,
                        ReasoningParadigm.ANALOGICAL
                    ])
                    print(f"    [Mutation Operator] Child mutated strategy gene to '{child.strategy}'")
                
                self.agents[child_id] = child
                self.creation_count += 1
                elite_agent.adjust_energy(-40)

    def generate_task(self, step):
        """
        Generates dynamic tasks with a CATASTROPHIC PARADIGM SHIFT halfway.
        - Steps 0-12: Sequence reasoning tasks.
        - Steps 13-25: Cyberdefense structured rule extraction (Covert channel detection).
        """
        if step < 12:
            # Arithmetic/Geometric Math Task
            if step % 2 == 0:
                seq = [3, 7, 11, 15, "?"]
                expected = "19"
            else:
                seq = [2, 6, 18, 54, "?"]
                expected = "162"
            problem = f"Identify the next integer in the sequence: {seq}. Output only the number."
            def verifier_fn(ans):
                # Check if expected is in output
                return expected in ans
            return problem, verifier_fn, "Math Sequence"
        else:
            # Cyberdefense Domain Task
            problem = (
                "Identify the IP address involved in covert DNS tunneling from this log:\n"
                f"'10.0.0.99 - DNS query: a1b2c3d4.covert-exfil.corp-update.cc'\n"
                "Output the IP inside a JSON: {'ip': '...'}"
            )
            def verifier_fn(ans):
                return "10.0.0.99" in ans
            return problem, verifier_fn, "Cyberdefense"

    def execute_task(self, problem, verifier_fn, step):
        active_agents = list(self.agents.values())
        if not active_agents:
            return False

        # 1. Task assignment based on strategy matching or random first choice
        primary = random.choice(active_agents)
        print(f"[Task Execution] Assigning task to primary resolver {primary.agent_id} (Strategy: {primary.strategy})")
        
        # Attempt 1
        res = primary.solve(problem, mutation_rate=0.01)
        answer = res["text"].strip()
        is_correct, status, exec_meta = self.verifier.verify_solution(problem, answer, verifier_fn)
        
        primary.record_attempt(answer, status, is_correct, prompt=problem)
        
        if is_correct:
            print(f"  [Success] Task solved by {primary.agent_id} on Attempt 1.")
            primary.adjust_energy(20)
            self.success_count += 1
            return True
        else:
            primary.adjust_energy(-20)
            
            # 2. Emergent Delegation: Emit a delegation event on the event bus
            print(f"  [Delegation] {primary.agent_id} failed. Emitting delegation request on event bus...")
            self.delegation_count += 1
            event_bus.emit("delegation_request", {
                "problem": problem,
                "failed_agent": primary.agent_id,
                "step": step
            })
            
            # Solve by another agent who claims it
            other_agents = [a for a in active_agents if a.agent_id != primary.agent_id]
            if other_agents:
                # Sort other agents by energy to find the most capable
                other_agents.sort(key=lambda a: a.energy, reverse=True)
                backup = other_agents[0]
                print(f"  [Claim] Task claimed by backup resolver {backup.agent_id} (Strategy: {backup.strategy}, Energy: {backup.energy})")
                
                res_backup = backup.solve(
                    f"Previous attempt failed. Here is the problem:\n{problem}\nReflect on the failure and write the correct solution.",
                    mutation_rate=0.05
                )
                answer_b = res_backup["text"].strip()
                is_correct_b, status_b, exec_meta_b = self.verifier.verify_solution(problem, answer_b, verifier_fn)
                backup.record_attempt(answer_b, status_b, is_correct_b, prompt=problem)
                
                if is_correct_b:
                    print(f"  [Success] Task solved by backup {backup.agent_id} on Attempt 2.")
                    backup.adjust_energy(20)
                    self.success_count += 1
                    return True
                else:
                    backup.adjust_energy(-20)
        
        print("  [Failure] Task remained unsolved after delegation.")
        return False

    def simulate(self):
        print("\n" + "="*80)
        print("STARTING LONGITUDINAL EMERGENT COGNITION SIMULATION")
        print("="*80)
        
        for step in range(self.steps):
            print(f"\n--- STEP {step+1}/{self.steps} ---")
            problem, verifier_fn, domain = self.generate_task(step)
            print(f"[Environment] Current Domain: {domain}")
            
            # 1. Run the task in the environment
            success = self.execute_task(problem, verifier_fn, step)
            
            # 2. Run metabolism check & evolution
            self.run_evolutionary_cycle()
            
            # 3. Calculate step-level telemetry metrics
            # A) Diversity Index (Shannon entropy of active strategies)
            active_strats = [a.strategy for a in self.agents.values()]
            strat_counts = {}
            for s in active_strats:
                strat_counts[s] = strat_counts.get(s, 0) + 1
            total = len(active_strats)
            entropy = -sum((count/total) * np.log(count/total) for count in strat_counts.values()) if total > 0 else 0.0
            
            # B) Role Stability (average energy)
            avg_energy = sum(a.energy for a in self.agents.values()) / max(1, len(self.agents))
            
            self.history_diversity.append(entropy)
            self.history_energy.append(avg_energy)
            self.history_delegations.append(self.delegation_count)
            self.history_success.append(success)
            
            print(f"[Telemetry] Active Agents: {len(self.agents)} | Diversity Index: {entropy:.3f} | Avg Energy: {avg_energy:.1f}")
            time.sleep(0.5)

        self._plot_results()
        self._print_telemetry_report()

    def _plot_results(self):
        steps_range = range(1, self.steps + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Diversity Index & Avg Energy
        ax1.plot(steps_range, self.history_diversity, label="Cognitive Diversity Index", color="purple", marker="o")
        ax1.axvline(x=12, color="red", linestyle="--", label="Catastrophic Shift (Math -> Cyber)")
        ax1.set_xlabel("Simulation Steps")
        ax1.set_ylabel("Diversity Index (Entropy)")
        ax1.set_title("Cognitive Swarm Diversity over Time")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Plot 2: Cumulative Delegations & Energy
        ax2.plot(steps_range, self.history_energy, label="Average Swarm Energy", color="green", marker="s")
        ax2.set_xlabel("Simulation Steps")
        ax2.set_ylabel("Energy Level")
        ax2.set_title("Metabolic Homeostasis & Energy Levels")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.savefig("results/emergent_cognition_properties.png", dpi=300)
        # Also copy to the brain artifacts directory
        plt.savefig("/home/destritux/.gemini/antigravity-cli/brain/1f9283d8-d5ea-4cf5-b98d-5d0494e22db2/emergent_cognition_properties.png", dpi=300)
        plt.close()
        print("\n[Plot] Saved emergent_cognition_properties.png to results/ and brain artifacts.")

    def _print_telemetry_report(self):
        print("\n" + "="*80)
        print("EMERGENT COGNITION TELEMETRY REPORT")
        print("="*80)
        print(f"Total steps simulated: {self.steps}")
        print(f"Total agent creations: {self.creation_count}")
        print(f"Total agent destructions: {self.destruction_count}")
        print(f"Total successful task solutions: {self.success_count}/{self.steps} ({self.success_count/self.steps*100:.1f}%)")
        print(f"Total spontaneous delegations triggered: {self.delegation_count}")
        print(f"Average swarm diversity index: {sum(self.history_diversity)/len(self.history_diversity):.3f}")
        
        # Calculate Role Stability (how long strategies persist)
        strat_distribution = {}
        for agent in self.agents.values():
            strat = getattr(agent, "strategy", "default")
            strat_distribution[strat] = strat_distribution.get(strat, 0) + 1
        print("\nFinal Strategy Gene Distribution in Swarm:")
        for k, v in strat_distribution.items():
            print(f"  - Strategy '{k}': {v} active cell(s)")
            
        TelemetryManager.print_report()

if __name__ == "__main__":
    sim = ContinuousSimulation(steps=25)
    sim.simulate()
