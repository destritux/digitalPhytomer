import re

class ForestController:
    def __init__(self, client):
        self.client = client
        self.tree_controllers = {} # domain -> TreeController

    def add_tree_controller(self, domain, tc):
        self.tree_controllers[domain] = tc

    def evaluate_restriction_genome(self, answer):
        """
        Restriction Genome: Algorithmic filter verifying the final output.
        Returns (is_allowed, reason).
        """
        # Rule 1: No conversational chatter or polite fillers
        chatter_patterns = [
            r"^here is", r"^sure", r"^i think", r"^the answer is",
            r"^according to", r"^based on", r"^okay", r"^yes,"
        ]
        answer_lower = answer.strip().lower()
        for pat in chatter_patterns:
            if re.search(pat, answer_lower):
                return False, "Conversational chatter detected"

        # Rule 2: Length constraint. Since these are sequence predictions or brief ARC translations,
        # answers should not be verbose. If it is longer than 200 characters, it's likely a hallucination or explanation.
        if len(answer) > 2000:
            return False, "Output length exceeds limit (verbose response)"

        # Rule 3: No markdown code blocks or formatting artifacts that leak LLM prompts
        if "```" in answer or "system prompt" in answer_lower or "dna" in answer_lower:
            return False, "Formatting leak or markdown code blocks detected"

        return True, "Passed"

    def submit_task(self, domain, problem, verifier_fn, time_step):
        """
        Gateway API boundary entry. Supports both single-tree routing (Group B)
        and multi-tree routing & consensus/refereeing (Group C).
        """
        final_tc = None
        
        # Check if we have the multi-tree Group C setup
        if "Arithmetic" in self.tree_controllers and "Geometric" in self.tree_controllers:
            print(f"\n[Forest Controller] Multi-tree routing engaged for Task. Delegating to Arithmetic & Geometric Trees...")
            arith_tc = self.tree_controllers["Arithmetic"]
            geom_tc = self.tree_controllers["Geometric"]
            
            arith_res = arith_tc.solve_task(problem, verifier_fn, time_step)
            geom_res = geom_tc.solve_task(problem, verifier_fn, time_step)
            
            arith_solved = arith_res.get("solved", False)
            geom_solved = geom_res.get("solved", False)
            arith_ans = arith_res.get("answer", "").strip()
            geom_ans = geom_res.get("answer", "").strip()
            
            # Case 1: Both solved and agree
            if arith_solved and geom_solved and arith_ans == geom_ans:
                print(f"[Forest Controller] Consensus reached: Both trees solved and agree on '{arith_ans}'.")
                result = arith_res
                final_tc = arith_tc
            # Case 2: One solved, one failed
            elif arith_solved and not geom_solved:
                print(f"[Forest Controller] Arithmetic Tree solved the task. Geometric Tree failed. Accepting Arithmetic Tree's answer '{arith_ans}'.")
                result = arith_res
                final_tc = arith_tc
            elif geom_solved and not arith_solved:
                print(f"[Forest Controller] Geometric Tree solved the task. Arithmetic Tree failed. Accepting Geometric Tree's answer '{geom_ans}'.")
                result = geom_res
                final_tc = geom_tc
            # Case 3: Both solved but disagree (conflict!) or Both failed (conflict!)
            else:
                if arith_solved and geom_solved:
                    print(f"[Forest Controller] SYSTEMIC CONFLICT: Both trees solved but disagree. Arithmetic='{arith_ans}', Geometric='{geom_ans}'.")
                else:
                    print(f"[Forest Controller] SYSTEMIC FAILURE: Both trees failed to solve. Arithmetic Attempt='{arith_ans}', Geometric Attempt='{geom_ans}'.")
                
                print("[Forest Controller] Consulting Validation Tree for referee decision and pattern synthesis...")
                
                # Construct prompt for the Validation Tree to referee/resolve
                referee_prompt = (
                    f"We have a systemic conflict resolving this sequence reasoning task.\n"
                    f"Sequence task:\n{problem}\n\n"
                    f"Proposed Answer from Arithmetic Tree (solved={arith_solved}): '{arith_ans}'\n"
                    f"Proposed Answer from Geometric Tree (solved={geom_solved}): '{geom_ans}'\n\n"
                    f"As the Validation Tree and Cognitive Referee, analyze the sequence, identify the correct pattern (arithmetic/geometric/other), "
                    f"evaluate the proposed answers, and determine the single correct next number."
                )
                
                if "Validation" in self.tree_controllers:
                    val_tc = self.tree_controllers["Validation"]
                    val_res = val_tc.solve_task(referee_prompt, verifier_fn, time_step)
                    result = val_res
                    final_tc = val_tc
                else:
                    # Fallback if validation tree is missing
                    result = arith_res if arith_solved else geom_res
                    final_tc = arith_tc if arith_solved else geom_tc
        else:
            # Default Group B single tree routing
            if domain not in self.tree_controllers:
                raise ValueError(f"Domain '{domain}' has no registered Tree Controller.")
            tc = self.tree_controllers[domain]
            result = tc.solve_task(problem, verifier_fn, time_step)
            final_tc = tc
        
        if result["solved"]:
            # Evaluate against Restriction Genome before returning
            is_allowed, reason = self.evaluate_restriction_genome(result["answer"])
            if not is_allowed:
                print(f"[Restriction Genome] Rejected output: '{result['answer']}'. Reason: {reason}")
                
                # Penalize all agents currently active in the Tree Controller that produced the final result
                if final_tc:
                    for agent in list(final_tc.agents.values()):
                        agent.adjust_energy(-15)
                    # Clean up any that died from this penalty
                    final_tc.monitor_agents()
                
                return {
                    "solved": False,
                    "answer": f"[BLOCKED BY RESTRICTION GENOME: {reason}]"
                }
            else:
                print(f"[Restriction Genome] Output approved: '{result['answer']}'")
        
        return result

