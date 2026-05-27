import time
import re
import matplotlib.pyplot as plt
from ollama_client import OllamaClient
from dataset import DatasetGenerator
from forest_controller import ForestController
from tree_controller import TreeController

# Configuration
MODEL_NAME = "qwen2.5:0.5b"

def verify_answer(answer_text, expected_str):
    """
    Checks if the expected answer is present as a standalone number/token in the LLM response.
    """
    # Cast to string in case a raw number/int is returned
    answer_str = str(answer_text)
    # Clean up non-digits except minus sign
    numbers = re.findall(r'-?\d+', answer_str)
    if numbers:
        # Check if the expected number is in the found numbers
        return expected_str in numbers
    return False

def run_group_a(client, tasks):
    """
    Group A (Baseline): Isolated local LLM instance.
    """
    print("\n" + "="*50)
    print("RUNNING GROUP A (BASELINE - SINGLE LLM)")
    print("="*50)
    
    client.reset_stats()
    results = []
    start_time = time.time()
    
    for idx, task in enumerate(tasks):
        print(f"[Group A] Processing Task {task['id']} ({task['type']})...")
        prompt = task["prompt"]
        
        # Isolated single call with default settings
        resp = client.generate(prompt=prompt, system_prompt=None, temperature=0.7, model_name=MODEL_NAME)
        
        is_correct = verify_answer(resp["text"], task["expected"])
        results.append({
            "id": task["id"],
            "correct": is_correct,
            "answer": resp["text"],
            "expected": task["expected"],
            "tokens": resp["prompt_tokens"] + resp["completion_tokens"]
        })
        print(f"  Expected: {task['expected']} | Answer: {resp['text'].strip()} | Correct: {is_correct}")
        # Add a tiny sleep to let the CPU rest between LLM calls
        time.sleep(0.5)
        
    duration = time.time() - start_time
    stats = client.get_stats()
    
    accuracy = sum(1 for r in results if r["correct"]) / len(tasks)
    
    print(f"\n[Group A Summary] Accuracy: {accuracy*100:.1f}%, Total Tokens: {stats['total_tokens']}, Time: {duration:.2f}s")
    return results, stats, duration

def run_group_b(client, tasks, fc):
    """
    Group B (Holarchic System): Forest > Tree > Micro-Agents with memory & metabolism.
    """
    print("\n" + "="*50)
    print("RUNNING GROUP B (HOLARCHIC AGENT SYSTEM)")
    print("="*50)
    
    client.reset_stats()
    results = []
    start_time = time.time()
    
    tc = fc.tree_controllers["Sequence Reasoning"]
    # Clear somatic memory from previous runs
    tc.somatic_memory.clear()
    
    last_tokens = 0
    for idx, task in enumerate(tasks):
        print(f"\n[Group B] Processing Task {task['id']} ({task['type']})...")
        prompt = task["prompt"]
        
        # Submit to Forest Controller
        res = fc.submit_task(
            domain="Sequence Reasoning",
            problem=prompt,
            verifier_fn=lambda ans: verify_answer(ans, task["expected"]),
            time_step=idx
        )
        
        is_correct = verify_answer(res["answer"], task["expected"])
        current_stats = client.get_stats()
        current_tokens = current_stats["total_tokens"]
        task_tokens = current_tokens - last_tokens
        last_tokens = current_tokens
        
        results.append({
            "id": task["id"],
            "correct": is_correct,
            "answer": res["answer"],
            "expected": task["expected"],
            "tokens": task_tokens
        })
        print(f"  Expected: {task['expected']} | Answer: {res['answer'].strip()} | Correct: {is_correct} | Tokens: {task_tokens}")
        time.sleep(0.5)

    duration = time.time() - start_time
    stats = client.get_stats()
    
    accuracy = sum(1 for r in results if r["correct"]) / len(tasks)
    
    print(f"\n[Group B Summary] Accuracy: {accuracy*100:.1f}%, Total Tokens: {stats['total_tokens']}, Time: {duration:.2f}s")
    return results, stats, duration

def run_group_c(client, tasks, fc_c):
    """
    Group C (Forest Coordination - 3 Trees): Forest Controller managing Arithmetic, Geometric, and Validation trees.
    """
    print("\n" + "="*50)
    print("RUNNING GROUP C (FOREST COORDINATION - 3 TREES)")
    print("="*50)
    
    client.reset_stats()
    results = []
    start_time = time.time()
    
    # Clear somatic memory for all tree controllers
    for tc in fc_c.tree_controllers.values():
        tc.somatic_memory.clear()
        
    last_tokens = 0
    for idx, task in enumerate(tasks):
        print(f"\n[Group C] Processing Task {task['id']} ({task['type']})...")
        prompt = task["prompt"]
        
        # Route to Forest Controller (routes to Arithmetic and Geometric, resolves with Validation)
        res = fc_c.submit_task(
            domain="Sequence Reasoning",
            problem=prompt,
            verifier_fn=lambda ans: verify_answer(ans, task["expected"]),
            time_step=idx
        )
        
        is_correct = verify_answer(res["answer"], task["expected"])
        current_stats = client.get_stats()
        current_tokens = current_stats["total_tokens"]
        task_tokens = current_tokens - last_tokens
        last_tokens = current_tokens
        
        results.append({
            "id": task["id"],
            "correct": is_correct,
            "answer": res["answer"],
            "expected": task["expected"],
            "tokens": task_tokens
        })
        print(f"  Expected: {task['expected']} | Answer: {res['answer'].strip()} | Correct: {is_correct} | Tokens: {task_tokens}")
        time.sleep(0.5)

    duration = time.time() - start_time
    stats = client.get_stats()
    
    accuracy = sum(1 for r in results if r["correct"]) / len(tasks)
    
    print(f"\n[Group C Summary] Accuracy: {accuracy*100:.1f}%, Total Tokens: {stats['total_tokens']}, Time: {duration:.2f}s")
    return results, stats, duration

def run_retention_test(client, tasks, group_name, solved_ids, fc=None):
    """
    Retention Index Test: Re-evaluates solved tasks to measure forgetting vs somatic recall.
    For Group B & C, we intercept exact matches using the vector database(s).
    """
    print("\n" + "-"*50)
    print(f"RUNNING RETENTION TEST FOR {group_name}")
    print("-"*50)
    
    is_group_c = False
    if fc:
        is_group_c = "Arithmetic" in fc.tree_controllers and "Geometric" in fc.tree_controllers
        if is_group_c:
            for tc in fc.tree_controllers.values():
                tc.trigger_allostasis()
        else:
            tc = fc.tree_controllers["Sequence Reasoning"]
            tc.trigger_allostasis()
        
    # We choose tasks that were successfully solved
    test_tasks = [t for t in tasks if t["id"] in solved_ids]
    if not test_tasks:
        print(f"[{group_name} Retention] No solved tasks found in main run. Testing first 5 tasks.")
        test_tasks = tasks[:5]
    else:
        test_tasks = test_tasks[:5]
        
    client.reset_stats()
    start_time = time.time()
    correct_count = 0
    
    for task in test_tasks:
        print(f"[{group_name} Retention] Re-evaluating Task {task['id']}...")
        
        if fc:
            task_emb = client.get_embeddings(task["prompt"])
            hits = []
            
            if is_group_c:
                # Group C: check somatic memory of both Arithmetic and Geometric trees
                arith_tc = fc.tree_controllers["Arithmetic"]
                geom_tc = fc.tree_controllers["Geometric"]
                
                hits = arith_tc.somatic_memory.query(task_emb, limit=1, min_similarity=0.95)
                if not hits:
                    hits = geom_tc.somatic_memory.query(task_emb, limit=1, min_similarity=0.95)
            else:
                # Group B: check somatic memory of Sequence Reasoning tree
                tc = fc.tree_controllers["Sequence Reasoning"]
                hits = tc.somatic_memory.query(task_emb, limit=1, min_similarity=0.95)
            
            if hits:
                ans = hits[0]["metadata"]["solution"]
                print(f"  [Somatic Memory Hit] Instant recall: '{ans}' (Similarity={hits[0]['similarity']:.3f})")
                is_correct = verify_answer(ans, task["expected"])
            else:
                # Fallback to FC solve
                res = fc.submit_task(
                    domain="Sequence Reasoning",
                    problem=task["prompt"],
                    verifier_fn=lambda ans: verify_answer(ans, task["expected"]),
                    time_step=999
                )
                ans = res["answer"]
                is_correct = verify_answer(ans, task["expected"])
        else:
            # Group A: Baseline single LLM call
            resp = client.generate(prompt=task["prompt"], system_prompt=None, temperature=0.7, model_name=MODEL_NAME)
            ans = resp["text"]
            is_correct = verify_answer(ans, task["expected"])
            
        if is_correct:
            correct_count += 1
            
        print(f"  Expected: {task['expected']} | Recalled: {ans.strip()} | Correct: {is_correct}")
        
    duration = time.time() - start_time
    stats = client.get_stats()
    accuracy = correct_count / len(test_tasks) if test_tasks else 0.0
    
    print(f"[{group_name} Retention Summary] Accuracy: {accuracy*100:.1f}%, Tokens Used: {stats['total_tokens']}, Duration: {duration:.4f}s")
    return accuracy, stats["total_tokens"], duration

def compile_group_c_plasticity(fc_c):
    """
    Compiles plasticity logs from all 3 tree controllers in Group C.
    """
    arith_log = fc_c.tree_controllers["Arithmetic"].plasticity_log
    geom_log = fc_c.tree_controllers["Geometric"].plasticity_log
    val_log = fc_c.tree_controllers["Validation"].plasticity_log
    
    step_data = {}
    
    def merge_log(log):
        for step, active, creations, destructions in log:
            # Ignore step 999 (which is the retention test logging)
            if step == 999:
                continue
            if step not in step_data:
                step_data[step] = {"active": 0, "creations": 0, "destructions": 0}
            step_data[step]["active"] += active
            step_data[step]["creations"] += creations
            step_data[step]["destructions"] += destructions
            
    merge_log(arith_log)
    merge_log(geom_log)
    merge_log(val_log)
    
    sorted_steps = sorted(step_data.keys())
    compiled_log = []
    for step in sorted_steps:
        data = step_data[step]
        compiled_log.append((step, data["active"], data["creations"], data["destructions"]))
    return compiled_log

def plot_plasticity_comparison(b_log, c_log):
    """
    Plots a comparative analysis of Group B and Group C active agents and cumulative lifecycle events.
    """
    if not b_log or not c_log:
        print("[Plot Warning] Plasticity logs empty. Skipping plot.")
        return
        
    b_steps, b_active, b_creations, b_destructions = zip(*[x for x in b_log if x[0] != 999])
    c_steps, c_active, c_creations, c_destructions = zip(*c_log)
    
    plt.figure(figsize=(12, 7))
    
    # Active agents
    plt.plot(b_steps, b_active, label="Group B (Single-Tree) Active MAs", color="blue", linewidth=2.5, marker="o")
    plt.plot(c_steps, c_active, label="Group C (Forest) Combined Active MAs", color="darkgreen", linewidth=2.5, marker="s")
    
    # Creations & Destructions
    plt.plot(b_steps, b_creations, label="Group B Cumulative Creations", color="blue", linestyle=":", alpha=0.6)
    plt.plot(b_steps, b_destructions, label="Group B Cumulative Destructions (GC)", color="red", linestyle=":", alpha=0.6)
    
    plt.plot(c_steps, c_creations, label="Group C Cumulative Creations", color="darkgreen", linestyle="--", alpha=0.6)
    plt.plot(c_steps, c_destructions, label="Group C Cumulative Destructions (GC)", color="orange", linestyle="--", alpha=0.6)
    
    plt.title("Ontogenetic Plasticity: Group B (Single-Tree) vs Group C (Forest Coordinator)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Time Step (Task Index)", fontsize=12)
    plt.ylabel("Agent Counts", fontsize=12)
    
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    
    # Save the graph
    plt.tight_layout()
    plt.savefig("ontogenetic_plasticity.png", dpi=300)
    print("\n[Plot] Saved comparative ontogenetic plasticity graph to 'ontogenetic_plasticity.png'")

def main():
    # 1. Initialize client and pull models
    client = OllamaClient(default_model=MODEL_NAME)
    pulled = client.pull_model(MODEL_NAME)
    if not pulled:
        print("[Fatal] Could not connect to Ollama or pull default model. Exiting.")
        return
        
    high_density_model = "qwen2.5:1.5b"
    pulled_hd = client.pull_model(high_density_model)
    if not pulled_hd:
        print(f"[Warning] Could not pull high-density model {high_density_model}. Sticking to default.")

    # 2. Generate Synthetic Dataset
    dg = DatasetGenerator(seed=42)
    tasks = dg.generate_tasks()
    print(f"\n[Dataset] Generated {len(tasks)} tasks.")
    print(f"  - Tasks 0-9: Arithmetic sequence (Rule A)")
    print(f"  - Tasks 10-19: Geometric sequence (Rule B - Paradigm Shift)")
    print("  - 30% of data occluded using '?' masks.")

    # 3. Setup Group B Forest Controller & Tree Controller
    fc = ForestController(client)
    tc = TreeController("Sequence Reasoning", client)
    fc.add_tree_controller("Sequence Reasoning", tc)

    # 4. Run Group A (Baseline)
    a_results, a_stats, a_duration = run_group_a(client, tasks)

    # 5. Run Group B (Holarchic System - 1 Tree)
    b_results, b_stats, b_duration = run_group_b(client, tasks, fc)

    # 6. Setup Group C Forest Controller & 3 Tree Controllers
    fc_c = ForestController(client)
    arith_tc = TreeController("Arithmetic", client)
    geom_tc = TreeController("Geometric", client)
    val_tc = TreeController("Validation", client)
    
    fc_c.add_tree_controller("Arithmetic", arith_tc)
    fc_c.add_tree_controller("Geometric", geom_tc)
    fc_c.add_tree_controller("Validation", val_tc)

    # 7. Run Group C (Forest Coordinator - 3 Trees)
    c_results, c_stats, c_duration = run_group_c(client, tasks, fc_c)

    # 8. Run Retention Tests on tasks solved successfully by each group
    a_solved_ids = [r["id"] for r in a_results if r["correct"]]
    b_solved_ids = [r["id"] for r in b_results if r["correct"]]
    c_solved_ids = [r["id"] for r in c_results if r["correct"]]
    
    # Re-run retention for Group A
    a_ret_acc, a_ret_tok, a_ret_dur = run_retention_test(client, tasks, "Group A", a_solved_ids)
    # Re-run retention for Group B
    b_ret_acc, b_ret_tok, b_ret_dur = run_retention_test(client, tasks, "Group B", b_solved_ids, fc=fc)
    # Re-run retention for Group C
    c_ret_acc, c_ret_tok, c_ret_dur = run_retention_test(client, tasks, "Group C", c_solved_ids, fc=fc_c)

    # 9. Print and Save Comparative Metrics
    print("\n" + "="*50)
    print("FINAL EXPERIMENTAL COMPARATIVE METRICS")
    print("="*50)
    
    # Generalization Rates
    a_acc_overall = sum(1 for r in a_results if r["correct"]) / len(tasks)
    b_acc_overall = sum(1 for r in b_results if r["correct"]) / len(tasks)
    c_acc_overall = sum(1 for r in c_results if r["correct"]) / len(tasks)
    
    # Pre-shift vs Post-shift Generalization
    a_pre_shift = sum(1 for r in a_results[:10] if r["correct"]) / 10
    b_pre_shift = sum(1 for r in b_results[:10] if r["correct"]) / 10
    c_pre_shift = sum(1 for r in c_results[:10] if r["correct"]) / 10
    
    a_post_shift = sum(1 for r in a_results[10:] if r["correct"]) / 10
    b_post_shift = sum(1 for r in b_results[10:] if r["correct"]) / 10
    c_post_shift = sum(1 for r in c_results[10:] if r["correct"]) / 10

    # Token/Accuracy Efficiency (Metabolism): total tokens / correct answers
    a_correct_count = sum(1 for r in a_results if r["correct"])
    b_correct_count = sum(1 for r in b_results if r["correct"])
    c_correct_count = sum(1 for r in c_results if r["correct"])
    
    a_efficiency = a_stats["total_tokens"] / a_correct_count if a_correct_count > 0 else float("inf")
    b_efficiency = b_stats["total_tokens"] / b_correct_count if b_correct_count > 0 else float("inf")
    c_efficiency = c_stats["total_tokens"] / c_correct_count if c_correct_count > 0 else float("inf")

    print(f"Emergent Generalization Rate (Overall):")
    print(f"  Group A (Baseline):  {a_acc_overall*100:.1f}%")
    print(f"  Group B (Holarchic): {b_acc_overall*100:.1f}%")
    print(f"  Group C (Forest):    {c_acc_overall*100:.1f}%")
    print(f"\nGeneralization Under Paradigm Shift (Out-of-Distribution):")
    print(f"  Pre-Shift (Tasks 0-9):   Group A: {a_pre_shift*100:.1f}% | Group B: {b_pre_shift*100:.1f}% | Group C: {c_pre_shift*100:.1f}%")
    print(f"  Post-Shift (Tasks 10-19): Group A: {a_post_shift*100:.1f}% | Group B: {b_post_shift*100:.1f}% | Group C: {c_post_shift*100:.1f}%")
    
    print(f"\nToken/Accuracy Efficiency (Metabolic Cost per Correct Solution):")
    print(f"  Group A (Baseline):  {a_efficiency:.1f} tokens/correct_ans")
    print(f"  Group B (Holarchic): {b_efficiency:.1f} tokens/correct_ans")
    print(f"  Group C (Forest):    {c_efficiency:.1f} tokens/correct_ans")
    
    num_solved_a = len(a_solved_ids[:5])
    num_solved_b = len(b_solved_ids[:5])
    num_solved_c = len(c_solved_ids[:5])
    print(f"\nRetention Index (Somatic Memory Retrieval on Solved Tasks):")
    print(f"  Group A (Baseline - {num_solved_a} tasks):  Accuracy: {a_ret_acc*100:.1f}% | Tokens: {a_ret_tok} | Time: {a_ret_dur:.4f}s (Catastrophic Forgetting Risk)")
    print(f"  Group B (Holarchic - {num_solved_b} tasks): Accuracy: {b_ret_acc*100:.1f}% | Tokens: {b_ret_tok} | Time: {b_ret_dur:.4f}s (Instant Recall)")
    print(f"  Group C (Forest - {num_solved_c} tasks):    Accuracy: {c_ret_acc*100:.1f}% | Tokens: {c_ret_tok} | Time: {c_ret_dur:.4f}s")

    # Compute Advanced Academic Metrics
    a_pre_tokens = sum(r["tokens"] for r in a_results[:10])
    a_post_tokens = sum(r["tokens"] for r in a_results[10:])
    a_delta_C = a_post_tokens - a_pre_tokens
    a_yield = (a_acc_overall * 100.0 * 1000000.0) / a_stats["total_tokens"] if a_stats["total_tokens"] > 0 else 0.0
    a_turnover = 0.0
    a_cfi = a_pre_shift - a_ret_acc
    a_allostatic_load = a_stats["prompt_tokens"]
    
    b_pre_tokens = sum(r["tokens"] for r in b_results[:10])
    b_post_tokens = sum(r["tokens"] for r in b_results[10:])
    b_delta_C = b_post_tokens - b_pre_tokens
    b_yield = (b_acc_overall * 100.0 * 1000000.0) / b_stats["total_tokens"] if b_stats["total_tokens"] > 0 else 0.0
    b_tc = fc.tree_controllers["Sequence Reasoning"]
    b_turnover = b_tc.destruction_count / b_tc.creation_count if b_tc.creation_count > 0 else 0.0
    b_cfi = b_pre_shift - b_ret_acc
    b_allostatic_load = b_stats["prompt_tokens"]
    
    c_pre_tokens = sum(r["tokens"] for r in c_results[:10])
    c_post_tokens = sum(r["tokens"] for r in c_results[10:])
    c_delta_C = c_post_tokens - c_pre_tokens
    c_yield = (c_acc_overall * 100.0 * 1000000.0) / c_stats["total_tokens"] if c_stats["total_tokens"] > 0 else 0.0
    c_creations = sum(tc.creation_count for tc in fc_c.tree_controllers.values())
    c_destructions = sum(tc.destruction_count for tc in fc_c.tree_controllers.values())
    c_turnover = c_destructions / c_creations if c_creations > 0 else 0.0
    c_cfi = c_pre_shift - c_ret_acc
    c_allostatic_load = c_stats["prompt_tokens"]

    import csv
    sequence_headers = [
        "Group", "Overall_Accuracy", "Pre_Shift_Accuracy", "Post_Shift_Accuracy",
        "Metabolic_Efficiency", "Retention_Accuracy", "Retention_Tokens", "Retention_Time",
        "Metabolic_Yield", "Custo_Adaptacao_Fronteira", "Cell_Turnover", "Catastrophic_Forgetting_Index", "Allostatic_Load"
    ]
    sequence_data = [
        ["Group A (Baseline)", a_acc_overall, a_pre_shift, a_post_shift, a_efficiency, a_ret_acc, a_ret_tok, a_ret_dur, a_yield, a_delta_C, a_turnover, a_cfi, a_allostatic_load],
        ["Group B (Single-Tree)", b_acc_overall, b_pre_shift, b_post_shift, b_efficiency, b_ret_acc, b_ret_tok, b_ret_dur, b_yield, b_delta_C, b_turnover, b_cfi, b_allostatic_load],
        ["Group C (Forest)", c_acc_overall, c_pre_shift, c_post_shift, c_efficiency, c_ret_acc, c_ret_tok, c_ret_dur, c_yield, c_delta_C, c_turnover, c_cfi, c_allostatic_load]
    ]
    with open("sequence_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(sequence_headers)
        writer.writerows(sequence_data)
    print("[CSV] Saved sequence_metrics.csv with advanced metrics.")

    # 10. Compile and Plot Comparison
    c_compiled_log = compile_group_c_plasticity(fc_c)
    plot_plasticity_comparison(tc.plasticity_log, c_compiled_log)

if __name__ == "__main__":
    main()
