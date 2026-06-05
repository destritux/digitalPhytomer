import random
from cognitive_memory import HierarchicalMemory

# Simple strategy paradigm prompts to simulate strategy genomes
STRATEGY_PROMPTS = {
    "default": "Analyze the problem and return the single direct answer.",
    "symbolic": "Solve the problem using formal logic, rules, and mathematical relations.",
    "adversarial": "Approach the problem by assuming the first quick thought is incorrect. Critique alternative answers before finalizing.",
    "decomposition": "Break the task down into sub-problems, solve each step explicitly, and join the results.",
    "analogical": "Find a similar logical pattern in memory, compare the structures, and map the relation to the current problem."
}

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

    @property
    def local_memory(self):
        return self.memory.episodic_log

    @local_memory.setter
    def local_memory(self, val):
        self.memory.episodic_log = val

    def solve(self, problem, mutation_rate=0.01, base_temp=0.3, model_name=None):
        """Attempts to solve a problem locally."""
        if self.failures_count >= 3:
            return {
                "success": False,
                "escalate": True,
                "text": f"Agent {self.agent_id} failed locally {self.failures_count} times. Escalating.",
                "is_mutated": False
            }

        # Build prompt incorporating local memory
        prompt_parts = []
        
        # Retrieve semantic context from somatic memory
        if self.use_somatic_memory:
            semantic_context = self.memory.retrieve_context(problem, self.client)
            if semantic_context:
                prompt_parts.append(semantic_context)

        # Retrieve episodic local attempts
        episodic_context = self.memory.get_memory_context()
        if episodic_context:
            prompt_parts.append(episodic_context)

        # Inject strategy instructions if active
        if self.strategy in STRATEGY_PROMPTS:
            prompt_parts.append(STRATEGY_PROMPTS[self.strategy])

        prompt_parts.append(f"Task: {problem}\nAnswer:")
        
        # Dynamic Mutation Check
        is_mutated = random.random() < mutation_rate
        temp = 0.8 if is_mutated else base_temp
        
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
            "is_mutated": is_mutated
        }

    def record_attempt(self, output, feedback, success, prompt=""):
        """Record the outcome of the solve attempt in memory."""
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

    def adjust_resource(self, amount):
        """Adjust agent resources (vascular flow)."""
        self.resource += amount
        if self.resource > self.max_resource:
            self.resource = self.max_resource
        elif self.resource < 0:
            self.resource = 0
            
        print(f"[Resource] Agent {self.agent_id} ({self.role}) resource adjusted by {amount:+.1f}. Current Resource: {self.resource}")

    def is_depleted(self):
        return self.resource <= 0

    # Compatibility alias for code referencing legacy tree_controller
    @property
    def energy(self):
        return self.resource

    @energy.setter
    def energy(self, val):
        self.resource = val

    @property
    def max_energy(self):
        return self.max_resource

    def is_dead(self):
        return self.is_depleted()
