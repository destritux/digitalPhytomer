import uuid
import re
import json
from micro_agent import MicroAgent
from vector_store import SomaticVectorStore

class TreeController:
    def __init__(self, domain, client):
        self.domain = domain
        self.client = client
        self.agents = {} # agent_id -> MicroAgent instance
        self.somatic_memory = SomaticVectorStore()
        
        # History for tracking ontogenetic plasticity (growth/death curves)
        self.creation_count = 0
        self.destruction_count = 0
        self.plasticity_log = [] # List of tuples: (time_step, active_count, creations, destructions)
        
        # Rolling stats for stress-induced hypermutation & legacy tissue isolation
        self.rolling_history = [] # Tracks boolean success of last 4 steps
        self.consecutive_failures = 0
        
        # Models configuration (Asymmetric allocation)
        self.default_model = "qwen2.5:0.5b"
        self.high_density_model = "qwen2.5:1.5b"

    def log_plasticity(self, time_step):
        active_count = len(self.agents)
        self.plasticity_log.append((time_step, active_count, self.creation_count, self.destruction_count))
        print(f"[Plasticity Log] Step {time_step}: Active={active_count}, Created={self.creation_count}, Destroyed={self.destruction_count}")

    def monitor_agents(self):
        """
        Garbage collection: Prunes agents that have depleted their energy.
        Also runs the evolutionary cycle.
        """
        dead_ids = [aid for aid, agent in self.agents.items() if agent.is_dead()]
        for aid in dead_ids:
            role = self.agents[aid].role
            print(f"[GC] Pruning dead agent {aid} ({role}) due to energy depletion.")
            if aid in self.agents:
                del self.agents[aid]
                self.destruction_count += 1
        
        # Run evolutionary swarm cycle
        self.run_evolutionary_cycle()

    def run_evolutionary_cycle(self):
        """
        Runs natural selection and reproduction on the current agent population.
        Applies diversity pressure to prevent agent collapse.
        """
        import random
        active_agents = list(self.agents.values())
        if len(active_agents) < 2:
            return

        # 1. Calculate strategy distribution for diversity penalty
        strat_dist = {}
        for agent in active_agents:
            strat = getattr(agent, "strategy", "default")
            strat_dist[strat] = strat_dist.get(strat, 0) + 1

        # 2. Fitness function: accuracy, energy efficiency, and diversity bonus
        def calculate_fitness(agent):
            accuracy = 1.0 / (agent.failures_count + 1)
            energy_efficiency = agent.energy / agent.max_energy
            
            strat = getattr(agent, "strategy", "default")
            strategy_ratio = strat_dist.get(strat, 1) / max(1, len(active_agents))
            diversity_bonus = 1.0 - strategy_ratio
            
            return 0.5 * accuracy + 0.3 * energy_efficiency + 0.2 * diversity_bonus

        # Sort active agents by fitness descending
        active_agents.sort(key=calculate_fitness, reverse=True)

        # 3. Natural Selection: Prune bottom 25% of weak agents if under stress
        prune_cutoff = max(1, int(len(active_agents) * 0.25))
        for weak_agent in active_agents[-prune_cutoff:]:
            if weak_agent.energy < 40:
                print(f"[Evolutionary Selection] Pruning weak agent {weak_agent.agent_id} due to low fitness / energy.")
                if weak_agent.agent_id in self.agents:
                    del self.agents[weak_agent.agent_id]
                    self.destruction_count += 1

        # 4. Reproduction: Elite top 25% replicate with partial semantic inheritance and mutation operators
        reproduce_cutoff = max(1, int(len(active_agents) * 0.25))
        for elite_agent in active_agents[:reproduce_cutoff]:
            if elite_agent.energy > 125:
                # Replicate child agent
                child_id = f"MA-{elite_agent.role}-replica-{random.randint(100, 999)}"
                print(f"[Evolutionary Selection] Elite agent {elite_agent.agent_id} replicates to child {child_id}.")
                
                child = self.create_agent(elite_agent.role, elite_agent.system_prompt)
                child.strategy = getattr(elite_agent, "strategy", "default")
                
                # Dynamic strategy mutation on child
                operators = ["mutate_strategy", "mutate_threshold", "none"]
                op = random.choice(operators)
                if op == "mutate_strategy":
                    from cognitive_strategy import ReasoningParadigm
                    child.strategy = random.choice([
                        ReasoningParadigm.SYMBOLIC,
                        ReasoningParadigm.ADVERSARIAL,
                        ReasoningParadigm.DECOMPOSITION,
                        ReasoningParadigm.ANALOGICAL
                    ])
                    print(f"  [Mutation Operator] Child mutated strategy gene to '{child.strategy}'")
                elif op == "mutate_threshold":
                    # Mutate local failures threshold
                    child.max_failures = random.choice([2, 3, 4])
                    print(f"  [Mutation Operator] Child mutated failures threshold to {child.max_failures}")

                # Inherit semantic memory reference, not episodic raw contamination
                child.memory.vector_store = elite_agent.memory.vector_store
                elite_agent.adjust_energy(-40) # Metabolic division cost

    def create_agent(self, role, system_prompt):
        """
        Instantiates a new MicroAgent, retrieving somatic memory context for epigenetic learning.
        """
        agent_id = f"MA-{role}-{str(uuid.uuid4())[:8]}"
        
        # Epigenetic learning: Search somatic memory for relevant past solutions
        epigenetic_context = None
        if self.somatic_memory.size() > 0:
            query_str = f"Role: {role}. Task domain: {self.domain}."
            query_emb = self.client.get_embeddings(query_str)
            hits = self.somatic_memory.query(query_emb, limit=2, min_similarity=0.2)
            if hits:
                epigenetic_context = [h["text"] for h in hits]
                print(f"[Epigenetics] Injected {len(epigenetic_context)} past solution pattern(s) into {agent_id}.")

        agent = MicroAgent(
            agent_id=agent_id,
            role=role,
            system_prompt=system_prompt,
            client=self.client,
            somatic_memory=self.somatic_memory
        )
        
        self.agents[agent_id] = agent
        self.creation_count += 1
        return agent

    def _parse_aggregator_response(self, text):
        """
        Attempts to parse JSON output from the aggregator: {"final_answer": "...", "confidence": 95}
        """
        # Look for JSON block in the text
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("final_answer", text), int(data.get("confidence", 0))
        except Exception:
            pass
        
        # Fallback regex search for confidence
        conf_match = re.search(r"confidence.*?(\d+)", text, re.IGNORECASE)
        confidence = int(conf_match.group(1)) if conf_match else 50
        # If no JSON, clean up the text to extract a clean final answer
        clean_text = text.replace("final_answer", "").strip()
        return clean_text, confidence

    def solve_task(self, problem, verifier_fn, time_step):
        """
        Coordinates the attempt to solve the task. 
        Uses a Primary MA first. If it fails repeatedly, triggers the Assembly Mechanism.
        """
        self.monitor_agents()
        self.log_plasticity(time_step)

        # Calculate rolling accuracy over the last 4 steps
        if len(self.rolling_history) > 0:
            rolling_acc = sum(self.rolling_history) / len(self.rolling_history)
        else:
            rolling_acc = 1.0

        # Calculate dynamic stress-induced mutation rate (scales from 1% to 30%)
        mutation_rate = 0.01 + 0.29 * (1.0 - rolling_acc)
        print(f"[Vascular System] Stress State: Rolling Accuracy={rolling_acc*100:.1f}%, Dynamic Mutation Rate={mutation_rate*100:.1f}%")

        # Asymmetric model allocation configuration
        solver_model = self.default_model
        skip_epigenetic = False

        # Damaged Tissue Isolation: If 2 consecutive failures occur, prune legacy and scale to higher density model
        if self.consecutive_failures >= 2:
            solver_model = self.high_density_model
            print(f"[Tissue Isolation] HIGH ENVIRONMENTAL STRESS DETECTED ({self.consecutive_failures} failures). Pruning legacy primary solver to avoid paradigm anchoring.")
            pruned_any = False
            for aid, agent in list(self.agents.items()):
                if agent.role == "primary_solver":
                    agent.energy = 0
                    pruned_any = True
            
            if pruned_any:
                self.monitor_agents() # GC the solver agent immediately
            
            skip_epigenetic = True  # Block epigenetic inheritance for the new solver to ensure it is virgin
            print(f"[Tissue Isolation] Instantiating primary solver on high-density model: '{solver_model}' to break cognitive stagnancy.")

        task_emb = self.client.get_embeddings(problem)
        active_somatic_texts = []

        # 1. Check somatic memory for cache hit (only if not skipping to bypass anchoring bias)
        if not skip_epigenetic and self.somatic_memory.size() > 0:
            exact_hits = self.somatic_memory.query(task_emb, limit=1, min_similarity=0.95)
            if exact_hits:
                retrieved_answer = exact_hits[0]["metadata"]["solution"]
                print(f"[Somatic Memory] Instant hit! Bypassing LLM generation and returning: '{retrieved_answer}'")
                
                # Update stats & reinforce matching memory node
                active_somatic_texts.append(exact_hits[0]["text"])
                self.somatic_memory.apply_feedback(active_somatic_texts, success=True)
                self.somatic_memory.apply_temporal_decay(active_somatic_texts)
                self.somatic_memory.prune_low_weight_vectors()
                
                self.rolling_history.append(True)
                self.rolling_history = self.rolling_history[-4:]
                self.consecutive_failures = 0
                
                self.log_plasticity(time_step)
                return {
                    "solved": True,
                    "answer": retrieved_answer
                }

        # Fetch epigenetic context (only if not skipping to ensure virgin learning)
        epigenetic_context = None
        if not skip_epigenetic and self.somatic_memory.size() > 0:
            hits = self.somatic_memory.query(task_emb, limit=2, min_similarity=0.3)
            if hits:
                epigenetic_context = [h["text"] for h in hits]
                active_somatic_texts.extend(epigenetic_context)
                print(f"[Vascular System] Epigenetic context found for task: {len(epigenetic_context)} items.")

        # 2. Get or create a Primary Solver Agent
        primary_solver = None
        for agent in self.agents.values():
            if agent.role == "primary_solver" and not agent.is_dead():
                primary_solver = agent
                primary_solver.failures_count = 0
                primary_solver.local_memory = []
                break
        
        if not primary_solver:
            if skip_epigenetic:
                print(f"[Vascular System] Creating new VIRGIN primary solver agent on high-density model '{solver_model}' (epigenetic inheritance blocked).")
            else:
                print(f"[Vascular System] Creating new primary solver agent on default model '{solver_model}'.")
            
            system_prompt = (
                "You are a sequence reasoning specialist. Analyze the input sequence, "
                "identify the mathematical logical pattern (addition, multiplication, etc.), "
                "and output the single correct next number in the sequence."
            )
            primary_solver = self.create_agent("primary_solver", system_prompt)

        # 3. Try to solve using the primary agent
        max_attempts = getattr(self, "max_attempts", 3)
        solved = False
        final_answer = ""
        is_mutated_hit = False

        for attempt in range(max_attempts):
            print(f"[Vascular System] Primary solver {primary_solver.agent_id} attempting solve (Attempt {attempt+1}/{max_attempts}).")
            # Pass the stress-induced mutation rate and the selected model name
            result = primary_solver.solve(problem, epigenetic_context=epigenetic_context, mutation_rate=mutation_rate, model_name=solver_model)
            
            if result["escalate"]:
                print(f"[Vascular System] Primary solver escalated task.")
                break

            answer = result["text"].strip()
            from cognitive_verifier import CognitiveVerifier
            verifier = CognitiveVerifier()
            is_correct, status_msg, exec_meta = verifier.verify_solution(problem, answer, verifier_fn)
            
            # Record the attempt locally (Restricted Autonomy loop)
            feedback = status_msg
            primary_solver.record_attempt(answer, feedback, is_correct, prompt=problem)
            
            if is_correct:
                print(f"[Vascular System] Primary solver successfully solved the task!")
                primary_solver.adjust_energy(20)
                solved = True
                final_answer = answer
                is_mutated_hit = result["is_mutated"]
                break
            else:
                primary_solver.adjust_energy(-25)
                if primary_solver.is_dead():
                    break

        # 4. If primary solver failed or escalated, run the Assembly Mechanism
        if not solved:
            if getattr(self, "disable_assembly", False):
                # Return failed result immediately if multi-agent assembly is disabled for budget symmetry
                return {"solved": False, "answer": final_answer, "is_mutated": is_mutated_hit}
            
            print("[Vascular System] Primary solver failed. Activating Assembly Mechanism (Layer 2 Group Synthesis)...")
            
            # Instantiate specialized assembly agents
            gen_prompt = (
                "You are a sequence reasoning generator agent. Analyze the sequence, identify the pattern "
                "(which could be arithmetic addition or geometric multiplication), and propose the next number."
            )
            crit_prompt = (
                "You are a sequence critic agent. Analyze the proposed sequence prediction draft, verify if the math "
                "is correct, and point out logical flaws or arithmetic mistakes."
            )
            agg_prompt = (
                "You are an aggregator agent. Synthesize the sequence prediction draft and the critic's feedback. "
                "Determine the single correct next number. You must output your response in this format:\n"
                '{"final_answer": "your_number_here", "confidence": 95}'
            )
            
            generator = self.create_agent("generator", gen_prompt)
            critic = self.create_agent("critic", crit_prompt)
            aggregator = self.create_agent("aggregator", agg_prompt)
            
            # Step A: Generator creates draft (routine validation stays on low-cost default model)
            gen_res = generator.solve(f"Task:\n{problem}", epigenetic_context=epigenetic_context, mutation_rate=mutation_rate, model_name=self.default_model)
            draft = gen_res["text"]
            
            # Step B: Critic critiques draft (stays on low-cost default model)
            crit_res = critic.solve(f"Problem:\n{problem}\n\nProposed Draft Solution:\n{draft}", mutation_rate=mutation_rate, model_name=self.default_model)
            critique = crit_res["text"]
            
            # Step C: Aggregator synthesizes (stays on low-cost default model)
            agg_res = aggregator.solve(
                f"Problem:\n{problem}\n\nDraft Solution:\n{draft}\n\nCritique:\n{critique}",
                mutation_rate=mutation_rate,
                model_name=self.default_model
            )
            
            assembly_answer, confidence = self._parse_aggregator_response(agg_res["text"])
            print(f"[Assembly] Aggregator produced answer: '{assembly_answer}' with confidence {confidence}%")
            
            if confidence >= 90:
                from cognitive_verifier import CognitiveVerifier
                verifier = CognitiveVerifier()
                is_correct, status_msg, exec_meta = verifier.verify_solution(problem, assembly_answer, verifier_fn)
                if is_correct:
                    print(f"[Assembly] Assembly solved the task correctly (Confidence={confidence}% >= 90%).")
                    generator.adjust_energy(15)
                    critic.adjust_energy(15)
                    aggregator.adjust_energy(15)
                    solved = True
                    final_answer = assembly_answer
                    is_mutated_hit = gen_res["is_mutated"] or crit_res["is_mutated"] or agg_res["is_mutated"]
                else:
                    print(f"[Assembly] Assembly response was incorrect despite high confidence ({confidence}%). Penalizing.")
                    generator.adjust_energy(-20)
                    critic.adjust_energy(-20)
                    aggregator.adjust_energy(-20)
            else:
                print(f"[Assembly] Assembly confidence ({confidence}%) was below 90% threshold. Rejecting.")
                generator.adjust_energy(-10)
                critic.adjust_energy(-10)
                aggregator.adjust_energy(-10)

        # 5. Epigenetic learning: Store the hit in Somatic Memory
        if solved:
            print(f"[Somatic Memory] Storing successful solution in vector bank (mutant={is_mutated_hit}).")
            solution_context = f"Problem: {problem} -> Solution: {final_answer}"
            self.somatic_memory.add_document(solution_context, task_emb, {
                "problem": problem,
                "solution": final_answer,
                "mutant": is_mutated_hit
            })

        # Update stats & feedback history
        self.rolling_history.append(solved)
        self.rolling_history = self.rolling_history[-4:]
        
        if solved:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        # Apply somatic memory feedback reinforcement / decay
        if active_somatic_texts:
            self.somatic_memory.apply_feedback(active_somatic_texts, success=solved)
        
        # Apply conditional temporal decay (to inactive vectors only)
        self.somatic_memory.apply_temporal_decay(active_somatic_texts)
        
        # Prune low-weight vectors
        self.somatic_memory.prune_low_weight_vectors()

        self.monitor_agents()
        self.log_plasticity(time_step)

        return {
            "solved": solved,
            "answer": final_answer
        }

    def trigger_allostasis(self):
        """
        Allostasis (Homeostatic Recovery): Simulates the refractory period of the organism.
        Resets consecutive failures to 0 and fills the rolling history with successes to lower
        the system's 'stress hormones', allowing normal somatic memory retrieval.
        """
        print("\n[Allostasis] Triggered homeostatic recovery. Resetting stress markers and rolling history.")
        self.consecutive_failures = 0
        self.rolling_history = [True, True, True, True]

