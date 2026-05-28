class ReasoningParadigm:
    SYMBOLIC = "symbolic_reasoning"       # Focus on formal equations, logic rules, code representation
    ADVERSARIAL = "adversarial_search"    # Look for counterexamples, stress test assertions
    DECOMPOSITION = "split_and_conquer"   # Break down task into atomic parts before executing
    ANALOGICAL = "analogical_mapping"     # Retrieve past vector solutions and map structures

STRATEGY_PROMPTS = {
    ReasoningParadigm.SYMBOLIC: (
        "\n[COGNITIVE STRATEGY: SYMBOLIC REASONING]\n"
        "Treat this challenge as a symbolic math or formal logic problem. "
        "Solve it by modeling the state transitions, writing formal step-by-step mathematical steps, "
        "or drafting a Python script model to calculate the next sequence number."
    ),
    ReasoningParadigm.ADVERSARIAL: (
        "\n[COGNITIVE STRATEGY: ADVERSARIAL ANALYSIS]\n"
        "Adopt an adversarial mindset. Search for boundary conditions, "
        "potential flaws in typical logic, and test cases where default assumptions fail. "
        "Consider if there are alternative patterns (like square/exponential shifts) that contradict your first guess."
    ),
    ReasoningParadigm.DECOMPOSITION: (
        "\n[COGNITIVE STRATEGY: TASK DECOMPOSITION]\n"
        "Decompose the problem. Split the task into intermediate sub-problems, "
        "solve each sub-problem sequentially, and then synthesize the final answer."
    ),
    ReasoningParadigm.ANALOGICAL: (
        "\n[COGNITIVE STRATEGY: ANALOGICAL RETRIEVAL]\n"
        "Draw analogies to known algorithms (e.g. shifts, geometric progressions, "
        "matrix operations). Relate the current parameters to past solved patterns retrieved in your context."
    )
}
