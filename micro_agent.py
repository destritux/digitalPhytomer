import random

class MicroAgent:
    def __init__(self, agent_id, role, system_prompt, client, initial_energy=100, max_energy=150):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.client = client
        self.energy = initial_energy
        self.max_energy = max_energy
        self.failures_count = 0
        self.local_memory = [] # List of dicts tracking previous attempts/failures for this session

    def solve(self, problem, epigenetic_context=None, mutation_rate=0.01, model_name=None):
        """
        Attempts to solve a problem locally. 
        Escalates if repeated local failures exceed the threshold (Restricted Autonomy).
        """
        if self.failures_count >= 3:
            # Escalation to Tree Controller
            return {
                "success": False,
                "escalate": True,
                "text": f"Agent {self.agent_id} failed locally {self.failures_count} times. Escalating to Tree Controller.",
                "is_mutated": False
            }

        # Build prompt incorporating local memory and epigenetic context (if any)
        prompt_parts = []
        
        if epigenetic_context:
            prompt_parts.append("=== Epigenetic Collective Memory (Context from other successful agents) ===")
            for ctx in epigenetic_context:
                prompt_parts.append(f"- Past successful solution pattern: {ctx}")
            prompt_parts.append("=========================================================================\n")

        if self.local_memory:
            prompt_parts.append("=== Local Memory (Your previous attempts on this task) ===")
            for idx, attempt in enumerate(self.local_memory):
                prompt_parts.append(f"Attempt {idx+1}: {attempt['output']}")
                prompt_parts.append(f"Feedback: {attempt['feedback']}")
            prompt_parts.append("Learn from your previous mistakes and try a different approach.\n")

        prompt_parts.append(f"Problem to solve:\n{problem}")
        
        # Dynamic Mutation Check
        is_mutated = random.random() < mutation_rate
        temp = 0.3
        max_tokens = None
        
        if is_mutated:
            # Check if this is stress-induced hypermutation (mutation rate >= 5%)
            if mutation_rate >= 0.05:
                temp = 0.6  # Lower temperature to maintain structured syntactic compliance
                max_tokens = 100  # Token limit to allow code validation while preventing infinite loops
                print(f"[{self.agent_id}] [HYPERMUTATION ENGAGED] Stress level triggers temp=0.6, max_tokens=100, and Code-Generation on model {model_name or self.client.default_model}.")
                prompt_parts.append(
                    "\n[WARNING: HIGH SYSTEMIC STRESS DETECTED - HYPERMUTATION ENGAGED]\n"
                    "Do NOT assume previous arithmetic patterns. The logical pattern has changed (e.g. geometric ratio, square, exponent, etc.).\n"
                    "Write a conceptual Python script or mathematical formula describing the rule and computing the next number.\n"
                    "Once you have written the script/formula, output your calculated final answer as an integer inside a JSON block at the very end:\n"
                    '{"final_answer": "your_calculated_number_here"}'
                )
            else:
                temp = 1.2
                print(f"[{self.agent_id}] [Stochastic Mutation Engaged] Temp=1.2.")
                prompt_parts.append("\n[Stochastic Mutation Rule Engaged: Take a creative, non-obvious logical leap. Ignore standard steps and explore new solutions!]")

        full_prompt = "\n".join(prompt_parts)

        # Call the LLM client
        resp = self.client.generate(
            prompt=full_prompt,
            system_prompt=self.system_prompt,
            temperature=temp,
            model_name=model_name,
            max_tokens=max_tokens
        )

        return {
            "success": resp["success"],
            "escalate": False,
            "text": resp["text"],
            "is_mutated": is_mutated
        }

    def record_attempt(self, output, feedback, success):
        """
        Record the outcome of the solve attempt in local memory.
        """
        self.local_memory.append({
            "output": output,
            "feedback": feedback,
            "success": success
        })
        if success:
            self.failures_count = 0
        else:
            self.failures_count += 1

    def adjust_energy(self, amount):
        """
        Adjust agent energy according to utility.
        """
        self.energy += amount
        if self.energy > self.max_energy:
            self.energy = self.max_energy
        elif self.energy < 0:
            self.energy = 0
            
        print(f"[Metabolism] Agent {self.agent_id} ({self.role}) energy adjusted by {amount:+.1f}. Current Energy: {self.energy}")

    def is_dead(self):
        return self.energy <= 0
