import numpy as np
import time

class SomaticVectorStore:
    def __init__(self):
        # List of dicts: {"text": str, "embedding": np.ndarray, "relevance": float, "usage_count": int, "metadata": dict}
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

        # Memory Interference: High similarity (>0.8) reduces existing document's relevance
        for doc in self.documents:
            similarity = float(np.dot(emb_arr, doc["embedding"]))
            if similarity > 0.8:
                old_rel = doc["relevance"]
                doc["relevance"] = round(doc["relevance"] * 0.95, 4)
                print(f"[Memory Interference] Similar memory detected. Relevance of existing memory reduced from {old_rel:.2f} to {doc['relevance']:.2f}")

        # Metadata including timestamp for recency
        meta = metadata or {}
        if "timestamp" not in meta:
            meta["timestamp"] = time.time()

        # Add new document with relevance=1.0, usage_count=0
        self.documents.append({
            "text": text,
            "embedding": emb_arr,
            "relevance": 1.0,
            "usage_count": 0,
            "metadata": meta
        })

        # Memory Capacity Limit: cap at 10 documents, prune lowest relevance
        if len(self.documents) > 10:
            # Sort by relevance ascending
            self.documents.sort(key=lambda x: x["relevance"])
            removed = self.documents.pop(0)
            print(f"[Memory Capacity] Pruned lowest relevance memory node: '{removed['text'][:40]}...'")

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
            # Score is cosine similarity scaled by the vector's current relevance
            score = cosine_sim * doc.get("relevance", 1.0)
            
            # Competitive Recency Retrieval: add a recency bonus
            timestamp = doc["metadata"].get("timestamp", 0)
            if timestamp > 0:
                age = time.time() - timestamp
                recency_bonus = 0.1 * np.exp(-max(0.0, age) / 600.0)
                score *= (1.0 + recency_bonus)

            if score >= min_similarity:
                results.append((score, cosine_sim, doc["relevance"], doc["text"], doc["metadata"]))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)
        
        return [
            {
                "score": score,
                "similarity": sim,
                "relevance": relevance,
                "text": text,
                "metadata": meta
            } for score, sim, relevance, text, meta in results[:limit]
        ]

    def apply_feedback(self, active_texts, success: bool):
        """
        Applies feedback to vectors that were part of the solution context.
        Reinforces on success, decays on failure.
        """
        for doc in self.documents:
            if doc["text"] in active_texts:
                old_rel = doc["relevance"]
                if success:
                    doc["relevance"] = min(1.0, doc["relevance"] + 0.1)
                    doc["usage_count"] = doc.get("usage_count", 0) + 1
                    print(f"[Memory Store] Reinforcing vector (success): '{doc['text'][:40]}...' relevance {old_rel:.2f} -> {doc['relevance']:.2f} (usage_count={doc['usage_count']})")
                else:
                    doc["relevance"] = max(0.0, round(doc["relevance"] * 0.85, 4))
                    print(f"[Memory Store] Penalizing vector (failure): '{doc['text'][:40]}...' relevance {old_rel:.2f} -> {doc['relevance']:.2f}")

    def apply_temporal_decay(self, active_texts, global_decay_factor=0.90):
        """
        Applies passive temporal decay with Gaussian noise to inactive vectors.
        """
        for doc in self.documents:
            if doc["text"] not in active_texts:
                old_rel = doc["relevance"]
                noise = np.random.normal(0, 0.02)
                decay_rate = max(0.7, min(0.99, global_decay_factor + noise))
                
                doc["relevance"] = round(doc["relevance"] * decay_rate, 4)
                if old_rel != doc["relevance"]:
                    print(f"[Memory Store] Passive temporal decay on inactive vector (decay_rate={decay_rate:.4f}): '{doc['text'][:40]}...' relevance {old_rel:.2f} -> {doc['relevance']:.2f}")

    def prune_low_relevance_vectors(self, threshold=0.25):
        """
        Permanently prunes/deletes vectors whose relevance drops below the threshold.
        """
        initial_count = len(self.documents)
        pruned_texts = [doc["text"] for doc in self.documents if doc["relevance"] < threshold]
        
        self.documents = [doc for doc in self.documents if doc["relevance"] >= threshold]
        pruned_count = initial_count - len(self.documents)
        
        for text in pruned_texts:
            print(f"[Memory Pruning] Deleting low-relevance memory node: '{text[:50]}...'")
            
        return pruned_count

    def clear(self):
        self.documents = []

    def size(self):
        return len(self.documents)
