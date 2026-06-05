import re
import numpy as np

class LSystemRegenerator:
    def __init__(self, agent_id, original_neighbors):
        """
        Initializes the L-system regenerator for a depleted agent.
        
        :param agent_id: The ID of the agent undergoing regeneration (e.g. 'Cell-001')
        :param original_neighbors: List of 3 agent IDs representing the neighbors of the agent at its depletion time.
        """
        self.agent_id = agent_id
        self.original_neighbors = list(original_neighbors)
        self.string = "A"

    def step(self, agents, active_ids):
        """
        Evolves the L-system string by one step and processes transitions.
        
        :param agents: The dict of all agents in the orchestrator.
        :param active_ids: The list of active (non-depleted) agent IDs.
        :return: List of 3 agent IDs to connect to if C(3) is resolved, otherwise None.
        """
        # Parse symbols from string
        tokens = re.findall(r'A|W\(\d+\)|M|T|C\(\d+\)', self.string)
        
        # Rule 4, 5, 6: Check if we have T and C(3)
        if "T" in tokens:
            c_token = None
            for tok in tokens:
                if tok.startswith("C("):
                    c_token = tok
                    break
            if c_token:
                # Rule 4: evaluate local vs global resource gradient
                local_resources = []
                for neighbor_id in self.original_neighbors:
                    if neighbor_id in agents:
                        local_resources.append(agents[neighbor_id].resource)
                    else:
                        local_resources.append(0.0)
                local_mean = np.mean(local_resources) if local_resources else 0.0
                
                global_resources = [agents[aid].resource for aid in active_ids if aid in agents]
                global_mean = np.mean(global_resources) if global_resources else 0.0
                
                print(f"      [L-System T] Agent {self.agent_id} evaluation: Local Mean = {local_mean:.2f}, Global Mean = {global_mean:.2f}")
                
                if local_mean > global_mean:
                    # Rule 5: local regeneration
                    resolved_connections = self.original_neighbors
                    print(f"      [L-System Rule 5] Local Mean > Global Mean. Choosing original neighbors: {resolved_connections}")
                else:
                    # Rule 6: topological plasticity (3 agents with highest resources)
                    candidates = [aid for aid in active_ids if aid != self.agent_id]
                    candidates.sort(key=lambda aid: agents[aid].resource, reverse=True)
                    resolved_connections = candidates[:3]
                    print(f"      [L-System Rule 6] Local Mean <= Global Mean. Choosing top resource agents: {resolved_connections}")
                
                self.string = ""  # Fully resolved
                return resolved_connections

        new_tokens = []
        # Check if there is W(n) and M
        has_w = any(tok.startswith("W(") for tok in tokens)
        
        if has_w:
            for tok in tokens:
                if tok.startswith("W("):
                    n = int(tok[2:-1])
                    if n > 1:
                        new_tokens.append(f"W({n-1})")
                    else:
                        # n == 1 becomes 0, wait timer finishes.
                        # Rule 2: When n reaches zero, the string rewrites to process variable M.
                        pass
                elif tok == "M":
                    # Check if the wait timer is ending in this step
                    w_tokens = [t for t in tokens if t.startswith("W(")]
                    if w_tokens:
                        w_token = w_tokens[0]
                        w_n = int(w_token[2:-1])
                        if w_n == 1:
                            # M is activated! Rule 3: M rewrites to T C(3)
                            new_tokens.append("T C(3)")
                        else:
                            new_tokens.append("M")
                    else:
                        new_tokens.append("M")
                else:
                    new_tokens.append(tok)
        else:
            for tok in tokens:
                if tok == "A":
                    # Rule 1: A rewrites to W(10) M
                    new_tokens.append("W(10) M")
                elif tok == "M":
                    # Rule 3: M rewrites to T C(3)
                    new_tokens.append("T C(3)")
                else:
                    new_tokens.append(tok)
                    
        self.string = " ".join(new_tokens)
        return None
