import time

class HierarchicalMemory:
    def __init__(self, vector_store, capacity=3):
        self.vector_store = vector_store  # SomaticVectorStore instance
        self.episodic_log = []
        self.capacity = capacity
        self.abstractions_count = 0

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
            # Simple capacity limit prune if client is missing
            self.episodic_log = self.episodic_log[-1:]
            return

        print("\n[Memory System] Capacity reached. Compressing episodic logs into semantic memory...")
        
        episodes_context = []
        for i, ep in enumerate(self.episodic_log):
            err_msg = ep["evaluation"].get("stderr") or "Logical validation failed"
            episodes_context.append(f"Attempt {i+1} Response: {ep['response'][:200]}\nOutcome/Error: {err_msg}")

        summary_prompt = (
            "Review these previous attempts to solve a computational/logical challenge.\n"
            "Analyze why they failed and write a general guideline detailing:\n"
            "1. What logical approach or format was incorrect and should be avoided.\n"
            "2. What correction or strategy should be applied next.\n"
            "Keep the lessons summarized, factual, and strictly under 80 words.\n\n"
            + "\n\n".join(episodes_context)
        )
        
        try:
            resp = client.generate(
                prompt=summary_prompt, 
                system_prompt="You are a cognitive processor distilling experiences into semantic rules."
            )
            semantic_lesson = resp["text"].strip()
            
            # Save the semantic lesson into the shared database for future searches
            emb = client.get_embeddings(semantic_lesson)
            self.vector_store.add_document(
                text=semantic_lesson,
                embedding=emb,
                metadata={"type": "semantic_lesson", "timestamp": time.time()}
            )
            print(f"[Memory System] Saved semantic lesson to vector db: '{semantic_lesson[:60]}...'")
        except Exception as e:
            print(f"[Memory System] Failed to generate semantic lesson: {e}")
        
        self.abstractions_count += 1
        # Clear episodic log (keeping only the latest/state log)
        self.episodic_log = self.episodic_log[-1:]

    def retrieve_context(self, query_text: str, client) -> str:
        """Retrieves semantic lessons matching the current query embedding."""
        if not client or not self.vector_store:
            return ""
        try:
            q_emb = client.get_embeddings(query_text)
            matches = self.vector_store.query(q_emb, limit=2, min_similarity=0.45)
            
            context_parts = []
            if matches:
                context_parts.append("\n=== Retrieved Semantic Knowledge ===")
                for m in matches:
                    context_parts.append(f"- Lesson: {m['text']}")
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
