import random
from cognitive_memory import HierarchicalMemory
from cognitive_strategy import STRATEGY_PROMPTS

class MicroAgent:
    def __init__(self, agent_id, role, system_prompt, client, initial_resource=100, max_resource=150, somatic_memory=None):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.resource = initial_resource
        self.max_resource = max_resource
        self.failures_count = 0
        self.strategy = "default"
        self.use_somatic_memory = True
        self.memory = HierarchicalMemory(somatic_memory)
        self.context_budget = 2000
        self.cognitive_load = 0.0
        self.ethylene_level = 0.0
        self.base_temperature = 0.3

    @property
    def local_memory(self):
        return self.memory.episodic_log

    @local_memory.setter
    def local_memory(self, val):
        self.memory.episodic_log = val

    def solve(self, problem, mutation_rate=0.01, model_name=None, agents=None):
        """
        Attempts to solve a problem locally. 
        """
        if self.failures_count >= 3:
            return {
                "success": False,
                "escalate": True,
                "text": f"Agent {self.agent_id} failed locally {self.failures_count} times. Escalating.",
                "is_mutated": False,
                "mycorrhizal_used": False
            }

        # Build prompt incorporating local memory
        prompt_parts = []
        
        # Hormesis Effect in RAG-VM (Inverted U-Curve)
        load = self.cognitive_load
        min_sim_dynamic = 0.12 * (load ** 2) - 0.22 * load + 0.45
        min_sim_dynamic = max(0.1, min(0.95, min_sim_dynamic))
        
        # Retrieve semantic context from somatic memory (local atomic facts)
        if self.use_somatic_memory:
            semantic_context = self.memory.retrieve_context(problem, self.client, min_similarity=min_sim_dynamic)
            if semantic_context:
                prompt_parts.append(semantic_context)

        # Local thermodynamic/soil connection (Global Memory Network Access)
        epigenetic_stress = self.cognitive_load * 0.5 + getattr(self, 'ethylene_level', 0.0) * 0.5
        if self.cognitive_load > 1.5 and epigenetic_stress > 1.0:
            GLOBAL_COMMUNICATION_COST = 2
            self.adjust_resource(-GLOBAL_COMMUNICATION_COST)
            q_emb = self.client.get_embeddings(problem)
            global_matches = self.memory.global_network.query(q_emb, limit=1, min_similarity=0.3)
            if global_matches:
                reflection = global_matches[0]["text"]
                facts = global_matches[0]["source_facts"]
                facts_str = "\n".join([f"- {f}" for f in facts])
                prompt_parts.append(
                    f"\n=== Global Shared Reflection (Soil Access) ===\n"
                    f"Reflection: {reflection}\n"
                    f"Supporting Atomic Facts:\n{facts_str}"
                )
                print(f"      [Soil Access] Agent {self.agent_id} retrieved global reflection: '{reflection[:40]}...'")

        # Mycorrhizal Symbiosis (Exocrine Memory P2P)
        used_mycorrhizal = False
        best_neighbor_id = None
        if self.failures_count >= 2 and self.cognitive_load > 1.5 and agents is not None:
            # Gated Communication check
            COMMUNICATION_COST = 2
            REWARD = 10
            
            # Query local memory score to evaluate uncertainty
            q_emb = self.client.get_embeddings(problem)
            matches = self.memory.vector_store.query(q_emb, limit=1, min_similarity=0.0)
            best_local_memory_score = matches[0]["score"] if matches else 0.0
            uncertainty = 1.0 - best_local_memory_score
            
            if (uncertainty * REWARD) > COMMUNICATION_COST:
                # Deduct communication cost from balance
                self.adjust_resource(-COMMUNICATION_COST)
                
                # Find low cognitive load active neighbors (< 1.0)
                active_peers = [bid for bid, b_agent in agents.items() if bid != self.agent_id and not b_agent.is_depleted()]
                low_load_peers = [bid for bid in active_peers if agents[bid].cognitive_load < 1.0]
                if low_load_peers:
                    low_load_peers.sort(key=lambda bid: self.trust_scores.get(bid, 0.5), reverse=True)
                    best_neighbor_id = low_load_peers[0]
                    best_neighbor = agents[best_neighbor_id]
                    
                    # Mycorrhizal policy distillation
                    policy = best_neighbor.memory.distill_expert_policy(problem, self.client)
                    if policy:
                        prompt_parts.append(f"\n[DIRETRIZ FEDERADA DO ESPECIALISTA]: {policy}")
                        # Deduct synthesis cost (-3 resources) from helper_id (Especialista)
                        best_neighbor.adjust_resource(-3)
                        used_mycorrhizal = True
                        print(f"      [Mycorrhizal Symbiosis] Agent {self.agent_id} retrieved distilled policy from {best_neighbor_id} (helper paid synthesis cost).")
            else:
                print(f"      [Gated Communication] Agent {self.agent_id} gated mycorrhizal query (Uncertainty: {uncertainty:.2f}).")
                return {
                    "success": False,
                    "escalate": False,
                    "text": "Mycorrhizal communication gated due to low expected value.",
                    "is_mutated": False,
                    "mycorrhizal_used": False,
                    "mycorrhizal_helper_id": None
                }

        # Retrieve episodic local attempts
        episodic_context = self.memory.get_memory_context()
        if episodic_context:
            prompt_parts.append(episodic_context)

        # Inject strategy instructions if active
        if self.strategy in STRATEGY_PROMPTS:
            prompt_parts.append(STRATEGY_PROMPTS[self.strategy])

        prompt_parts.append(f"Task: {problem}\nAnswer:")
        
        # Dynamic Mutation Check (simplifies temperature control, removes text warnings)
        is_mutated = random.random() < mutation_rate
        temp = 0.8 if is_mutated else getattr(self, 'base_temperature', 0.3)
        
        full_prompt = "\n".join(prompt_parts)

        # Call the LLM client
        resp = self.client.generate(
            prompt=full_prompt,
            system_prompt=self.system_prompt,
            temperature=temp,
            model_name=model_name
        )

        return {
            "success": resp["success"],
            "escalate": False,
            "text": resp["text"],
            "is_mutated": is_mutated,
            "mycorrhizal_used": used_mycorrhizal,
            "mycorrhizal_helper_id": best_neighbor_id if used_mycorrhizal else None,
            "distilled_policy": policy if (used_mycorrhizal and 'policy' in locals()) else None
        }

    def record_attempt(self, output, feedback, success, prompt=""):
        """
        Record the outcome of the solve attempt in memory.
        """
        self.memory.add_episode(
            prompt=prompt,
            response=output,
            evaluation_meta={"stderr": feedback, "success": success},
            client=self.client
        )
        if success:
            self.failures_count = 0
        else:
            self.failures_count += 1
            # ppGpp-like stress signaling (acute ethylene/ppGpp synthesis delta)
            self.ethylene_level = getattr(self, 'ethylene_level', 0.0) + 0.5

    def adjust_resource(self, amount):
        """
        Adjust agent resources.
        """
        self.resource += amount
        if self.resource > self.max_resource:
            self.resource = self.max_resource
        elif self.resource < 0:
            self.resource = 0
            
        print(f"[Resource] Agent {self.agent_id} ({self.role}) resource adjusted by {amount:+.1f}. Current Resource: {self.resource}")

    def is_depleted(self):
        return self.resource <= 0
