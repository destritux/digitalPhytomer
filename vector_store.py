import numpy as np

class SomaticVectorStore:
    def __init__(self):
        # List of dicts: {"text": str, "embedding": np.ndarray, "weight": float, "reinforcements": int, "metadata": dict}
        self.documents = []

    def add_document(self, text, embedding, metadata=None):
        if not embedding:
            return False
        
        # Convert embedding to numpy array
        emb_arr = np.array(embedding, dtype=float)
        
        # Normalize the embedding vector
        norm = np.linalg.norm(emb_arr)
        if norm > 0:
            emb_arr = emb_arr / norm
        else:
            return False

        # Initialize weight to 1.0, reinforcements to 0
        self.documents.append({
            "text": text,
            "embedding": emb_arr,
            "weight": 1.0,
            "reinforcements": 0,
            "metadata": metadata or {}
        })
        return True

    def query(self, query_embedding, limit=3, min_similarity=0.3):
        if not query_embedding or not self.documents:
            return []

        # Convert query embedding to numpy array and normalize
        q_arr = np.array(query_embedding, dtype=float)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm
        else:
            return []

        results = []
        for doc in self.documents:
            # Cosine similarity (dot product of normalized vectors)
            cosine_sim = float(np.dot(q_arr, doc["embedding"]))
            # Score is cosine similarity scaled by the vector's current weight
            score = cosine_sim * doc.get("weight", 1.0)
            
            if score >= min_similarity:
                results.append((score, cosine_sim, doc["weight"], doc["text"], doc["metadata"]))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [
            {
                "score": score,
                "similarity": sim,
                "weight": weight,
                "text": text,
                "metadata": meta
            } for score, sim, weight, text, meta in results[:limit]
        ]

    def apply_feedback(self, active_texts, success: bool):
        """
        Applies feedback to vectors that were part of the solution context.
        Reinforces on success, decays on failure (calibrated to 0.85).
        """
        for doc in self.documents:
            if doc["text"] in active_texts:
                old_w = doc["weight"]
                if success:
                    doc["weight"] = min(1.0, doc["weight"] + 0.1)
                    doc["reinforcements"] = doc.get("reinforcements", 0) + 1
                    print(f"[Somatic Store] Reinforcing vector (success): '{doc['text'][:40]}...' weight {old_w:.2f} -> {doc['weight']:.2f} (reinforcements={doc['reinforcements']})")
                else:
                    doc["weight"] = max(0.0, doc["weight"] * 0.85) # Calibrated fail decay (0.85 instead of 0.5)
                    print(f"[Somatic Store] Penalizing vector (failure): '{doc['text'][:40]}...' weight {old_w:.2f} -> {doc['weight']:.2f}")

    def apply_temporal_decay(self, active_texts):
        """
        Applies passive temporal decay ONLY to vectors that were NOT active in this step.
        Uses habituation: decay rate decreases (memory hardens) as reinforcements increase.
        """
        for doc in self.documents:
            if doc["text"] not in active_texts:
                old_w = doc["weight"]
                reinforcements = doc.get("reinforcements", 0)
                
                # Habituation scaling: base decay 0.95, reinforces move it closer to 0.99
                decay_rate = 0.95 + 0.04 * min(1.0, reinforcements / 10.0)
                decay_rate = round(decay_rate, 4)
                
                doc["weight"] = round(doc["weight"] * decay_rate, 4)
                if old_w != doc["weight"]:
                    print(f"[Somatic Store] Passive temporal decay on inactive vector (decay_rate={decay_rate:.4f}): '{doc['text'][:40]}...' weight {old_w:.2f} -> {doc['weight']:.2f}")

    def prune_low_weight_vectors(self, threshold=0.25):
        """
        Permanently prunes/deletes vectors whose relevance weight drops below the threshold.
        """
        initial_count = len(self.documents)
        pruned_texts = [doc["text"] for doc in self.documents if doc["weight"] < threshold]
        
        self.documents = [doc for doc in self.documents if doc["weight"] >= threshold]
        pruned_count = initial_count - len(self.documents)
        
        for text in pruned_texts:
            print(f"[Synaptic Pruning] Deleting low-relevance memory node: '{text[:50]}...'")
            
        return pruned_count

    def clear(self):
        self.documents = []

    def size(self):
        return len(self.documents)
