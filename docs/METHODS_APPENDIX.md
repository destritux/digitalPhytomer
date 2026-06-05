# Appendix A: Experimental Methods and Hyper-parameter Configurations

This appendix provides a comprehensive technical reference of all parameters, settings, and hardware details to ensure the exact reproducibility of the Digital Phytomer simulation and statistical validations.

---

## 1. Multi-Agent Swarm Parametrization

The following table summarizes the core swarm parameters utilized across all experimental groups (A, B, C, C-Ablated, and Baselines):

| Parameter | Value | Description |
| :--- | :--- | :--- |
| Swarm Size ($N$) | 8 | Total active processing units in the swarm. |
| Initial Energy Resource | 100 | Initial energy assigned to each unit at the start of Epoch 1. |
| Max Energy Capacity | 150 | Upper limit of energy units a cell can accumulate. |
| Energy Success Reward | +10 | Energy rewarded to a unit upon a successful task resolution. |
| Energy Failure Penalty (Primary) | -5 | Energy deducted from the primary unit upon failure. |
| Energy Failure Penalty (Helper) | -2 | Energy deducted from the helper unit upon delegation failure. |
| Abscission Threshold | $\le 0$ | Resource level at which a unit undergoes cellular turnover. |
| Initial Trust Score | 0.5 | Default relational trust value between any two units. |
| Trust Increase Step (Success) | +0.03 | Amount added to trust score for successful delegation (Primary to Helper). |
| Trust Reciprocity Step (Success)| +0.01 | Reciprocal trust added from Helper to Primary. |
| Trust Penalty Step (Failure) | -0.02 | Trust deducted from Helper score upon delegation failure. |

---

## 2. Somatic Memory and Monopoly Tax

### Somatic Memory Pruning & Decay
Somatic memory is modeled using a localized vector database with temporal decay:
- **Decay Rate ($\gamma$)**: $0.90$ per step for inactive memory vectors.
- **Episodic Capacity**: Max 10 episodic logs before compression into semantic lessons.
- **Similarity Metric**: Cosine distance over word-hashing vectors (256-dimensional).
- **Pruning Threshold**: Memory nodes with relevance score below $0.25$ are pruned.

### Monopoly Tax Algorithm
To stabilize specialized roles while avoiding monopoly-driven bottlenecking, a local domain tax is applied:
$$\text{Bid Score} = \text{Trust} \times (1.0 - \text{Monopoly Tax})$$
- **Base Monopoly Tax**: $0.03$ per consecutive success in a domain.
- **Decay**: Halves if another agent resolves a task in the domain.

---

## 3. Large Language Model (LLM) Server Configuration

All inference calls were processed locally to ensure data security and avoid API rate limitations:
- **Server Platform**: Ollama (v0.1.48)
- **Local Host URL**: `http://localhost:11434`
- **Models**:
  - Primary Swarm Units: `qwen2.5:0.5b` (Q4_K_M quantization)
  - Baselines & Orchestrators: `qwen2.5:1.5b` (where specified in baselines)
- **Inference Parameters**:
  - Temperature: $0.7$
  - Top-P: $0.9$
  - Max Tokens (`num_predict`): $256$

---

## 4. Hardware Environment

All simulations were executed on the following workstation setup:
- **Operating System**: Linux (Ubuntu 22.04 LTS, Kernel 6.5)
- **CPU**: Intel Core i7-12700H (14 cores, 20 threads)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB Dedicated GDDR6 VRAM)
- **RAM**: 16GB DDR4 @ 3200MHz
- **Simulation Latency**: Average of 85ms per LLM inference call under local GPU execution.

---

## 5. Statistical Framework and Multi-Seed Settings

- **Number of Seeds**: 30 independent seeds (Seeds 42 through 71).
- **LLM Cache isolation**: Unique key per seed (`hash(prompt + seed + agent + model)`) to prevent cross-seed leakage.
- **Statistical Test**: Mann-Whitney U test (one-sided, alternative = "greater" for Group C).
- **Multiple Comparisons Correction**: Holm-Bonferroni step-down correction applied to control family-wise error rate (FWER) across all baselines.
