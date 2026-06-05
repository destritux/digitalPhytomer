# Digital Phytomer: Emergent Swarm Cognition with Generative L-System Regeneration

An advanced agentic simulation framework that models emergent cooperation, localized specialization, and dynamic topological adaptation in decentralized multi-agent networks (e.g., autonomous drone swarms or edge-computing architectures).

## Overview

The **Digital Phytomer** architecture draws inspiration from plant physiology (metameric growth and vascular systems) to design self-organizing multi-agent swarms. It mitigates the fragility of classical centralized controllers by utilizing localized trust networks, somatic vector memory, and a generative L-System-based topological regeneration system to recover from paradigm shifts and node lesions.

---

## Architectural Features

### 1. Emergent Swarm Cognition
- **Cooperative Bidding**: Agents evaluate incoming tasks based on localized competence (measured via somatic memory query similarity) and a monopoly tax to prevent node dominance.
- **P2P Peer Delegation**: If a primary agent fails to resolve a task, it delegates the task to its top trusted neighbors in the local communication mesh, paying coordination/energy costs.
- **Symmetric Trust Updates**: Relational trust scores between agents are adjusted dynamically based on cooperative delegation outcomes.

### 2. Somatic & Episodic Memory
- **Somatic Memory**: Powered by local vector stores that cache success patterns. It applies temporal decay to model forgetting and prunes low-relevance vectors.
- **Episodic Log**: Keeps track of localized sequential attempts to refine prompts during hard tasks.

### 3. Generative L-System Regeneration
Replaces static reconnection with a generative grammar processor that controls node abscission (logical death) and meristematic regrowth when resources reach zero:
- **Initial Axiom (`A`)**: Queued when an agent is depleted of resources.
- **Wait Timer (`W(n)`)**: Decrements by 1 step at each global iteration.
- **Meristematic Trigger (`M`)**: Rewrites to `T C(3)` once the wait timer finishes (`n = 0`).
- **Context-Sensitive Gradient Evaluator (`T`)**: Compares the resource average of the dead node's 3 original neighbors with the global resource average of the active swarm:
  - **Local Regeneration (Rule 5)**: If the local resource average is higher than the global network average, the node regenerates and reconnects to its 3 original neighbors.
  - **Topological Plasticity (Rule 6)**: If the local area is depleted (local average $\le$ global average), the node adapts its connection by binding to the 3 highest-resource agents in the swarm.
- **Connection Agent Instantiation (`C(3)`)**: Instantiates the new agent with a completely clean episodic memory, resets its energy to base settings, and injects the 3 target neighbors directly into its Trust table.

