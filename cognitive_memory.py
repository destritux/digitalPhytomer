import time
import numpy as np

class HierarchicalMemory:
    def __init__(self, vector_store, capacity=3):
        self.vector_store = vector_store  # SomaticVectorStore instance (local atomic facts)
        self.atomic_facts = vector_store  # Alias for model architecture clarity
        self.episodic_log = []            # list of episodic dicts (raw episodes)
        self.episodic_logs = self.episodic_log # Alias
        self.capacity = capacity
        self.abstractions_count = 0
        from vector_store import global_memory_network
        self.global_network = global_memory_network

    def add_episode(self, prompt: str, response: str, evaluation_meta: dict, client=None):
        self.episodic_log.append({
            "prompt": prompt,
            "response": response,
            "output": response,
            "feedback": evaluation_meta.get("stderr", ""),
            "success": evaluation_meta.get("success", False),
            "evaluation": evaluation_meta
        })
        if len(self.episodic_log) >= self.capacity:
            self.compress_to_semantic(client)

    def compress_to_semantic(self, client):
        """Converts raw episodic memories into generalized semantic lessons to prevent context poisoning."""
        if not client or not self.episodic_log:
            self.episodic_log = self.episodic_log[-1:]
            return

        print("\n[Memory System] Capacity reached. Compressing episodic logs into atomic facts...")
        
        episodes_context = []
        for i, ep in enumerate(self.episodic_log):
            err_msg = ep["evaluation"].get("stderr") or "Logical validation failed"
            episodes_context.append(f"Attempt {i+1} Response: {ep['response'][:200]}\nOutcome/Error: {err_msg}")

        summary_prompt = (
            "Review these previous failed attempts to solve a computational/logical challenge.\n"
            "Extract 2 to 3 objective, factual guidelines or atomic rules (statements of what is technically correct, isolated from the errors).\n"
            "Avoid mentioning the error itself or agent names. Keep facts generalized and reusable.\n\n"
            + "\n\n".join(episodes_context) + "\n\n"
            "Output ONLY a JSON object with a 'facts' key containing a list of strings."
        )
        
        try:
            resp = client.generate(
                prompt=summary_prompt, 
                system_prompt="You are a technical cognitive processor extracting atomic facts from execution logs.",
                json_format=True
            )
            import json
            data = json.loads(resp["text"])
            facts = data.get("facts", [])
            
            for fact in facts:
                fact = fact.strip()
                if fact:
                    # Save the atomic fact in the local vector store
                    emb = client.get_embeddings(fact)
                    self.vector_store.add_document(
                        text=fact,
                        embedding=emb,
                        metadata={"type": "atomic_fact", "timestamp": time.time()}
                    )
                    print(f"[Memory System] Saved local atomic fact: '{fact[:60]}...'")
            
            # Synthesize Reflections
            self.synthesize_reflections(client)
            
        except Exception as e:
            print(f"[Memory System] Failed to extract atomic facts: {e}")
        
        self.abstractions_count += 1
        # Clear episodic log (keeping only the latest/state log)
        self.episodic_log = self.episodic_log[-1:]

    def synthesize_reflections(self, client):
        if not client or not self.vector_store:
            return
        
        docs = self.vector_store.documents
        if len(docs) < 3:
            return
        
        N = 3
        cluster = []
        for i, doc1 in enumerate(docs):
            current_cluster = [doc1]
            emb1 = doc1["embedding"]
            for j, doc2 in enumerate(docs):
                if i != j:
                    emb2 = doc2["embedding"]
                    sim = float(np.dot(emb1, emb2))
                    if sim > 0.70:
                        current_cluster.append(doc2)
            if len(current_cluster) >= N:
                cluster = current_cluster[:N]
                break
        
        if len(cluster) >= N:
            facts_texts = [doc["text"] for doc in cluster]
            print(f"\n[Memory Reflection] Consolidating {N} similar facts: {facts_texts}")
            
            prompt = (
                "Review these related atomic technical facts and consolidate them into a single "
                "Higher-Order Reflection (a general, reusable technical heuristic/principle).\n"
                f"Facts to consolidate:\n" + "\n".join([f"- {t}" for t in facts_texts]) + "\n\n"
                "Output ONLY a JSON object with a 'reflection' key containing the consolidated rule in under 40 words."
            )
            
            try:
                resp = client.generate(
                    prompt=prompt,
                    system_prompt="You are a cognitive consolidator synthesizing facts into general principles.",
                    json_format=True
                )
                import json
                data = json.loads(resp["text"])
                reflection = data.get("reflection", "").strip()
                
                if reflection:
                    ref_emb = client.get_embeddings(reflection)
                    self.global_network.push_reflection(reflection, ref_emb, facts_texts)
                    print(f"[Memory Reflection] Pushed Higher-Order Reflection to Global Server: '{reflection}'")
                    
                    self.vector_store.documents = [doc for doc in docs if doc["text"] not in facts_texts]
                    print(f"[Memory Reflection] Local edge memory pruned. Removed {len(facts_texts)} facts.")
            except Exception as e:
                print(f"[Memory Reflection Error] Reflection synthesis failed: {e}")

    def distill_expert_policy(self, query_text: str, client) -> str:
        if not client or not self.vector_store:
            return ""
        
        try:
            q_emb = client.get_embeddings(query_text)
            matches = self.vector_store.query(q_emb, limit=3, min_similarity=0.3)
            if not matches:
                return ""
            
            matches_text = "\n".join([f"- {m['text']}" for m in matches])
            
            prompt = (
                "Você é um especialista. Baseado nestas memórias recuperadas:\n"
                f"{matches_text}\n\n"
                "Destile uma heurística universal ou diretriz de prompt em menos de 30 palavras "
                f"para ajudar outro agente a resolver o problema: '{query_text}'. Não revele dados brutos."
            )
            
            resp = client.generate(
                prompt=prompt,
                system_prompt="Você é um especialista destilando políticas heurísticas concisas.",
                temperature=0.3,
                max_tokens=60
            )
            
            policy = resp["text"].strip()
            print(f"      [Policy Distillation] Distilled heuristic: '{policy}'")
            return policy
        except Exception as e:
            print(f"[Policy Distillation Error] Failed: {e}")
            return ""

    def retrieve_context(self, query_text: str, client, min_similarity=0.45) -> str:
        """Retrieves semantic lessons matching the current query embedding."""
        if not client or not self.vector_store:
            return ""
        try:
            q_emb = client.get_embeddings(query_text)
            matches = self.vector_store.query(q_emb, limit=2, min_similarity=min_similarity)
            
            context_parts = []
            if matches:
                context_parts.append("\n=== Retrieved Semantic Knowledge ===")
                for m in matches:
                    context_parts.append(f"- Fact: {m['text']}")
            return "\n".join(context_parts)
        except Exception as e:
            print(f"[Memory System] Retrieval failed: {e}")
            return ""

    def get_memory_context(self) -> str:
        if not self.episodic_log:
            return ""
        parts = ["=== Previous attempts for this task ==="]
        for idx, ep in enumerate(self.episodic_log):
            parts.append(f"Attempt {idx+1}: {ep['response']}")
            err = ep["evaluation"].get("stderr") or "Logical validation failed"
            parts.append(f"Feedback/Error: {err}")
        parts.append("Correct the errors from previous attempts.")
        return "\n".join(parts)
