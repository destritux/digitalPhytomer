import numpy as np
import time

class SomaticVectorStore:
    def __init__(self):
        # List of dicts: {"text": str, "raw_embedding": np.ndarray, "diffused_embedding": np.ndarray, "embedding": np.ndarray, "relevance": float, "usage_count": int, "metadata": dict}
        self.documents = []

    def _update_diffused_embeddings(self):
        if not self.documents:
            return
        
        n = len(self.documents)
        if n <= 1:
            for doc in self.documents:
                doc["diffused_embedding"] = doc["raw_embedding"]
                doc["embedding"] = doc["raw_embedding"]
            return
        
        # Extract raw embeddings: N x d matrix
        phi = np.array([doc["raw_embedding"] for doc in self.documents])
        
        # Parameters for RAG-VM
        alpha = 0.5
        sigma = 1.0
        kappa = 0.5
        k = min(3, n - 1)
        
        # Build adjacency matrix W
        W = np.zeros((n, n))
        for i in range(n):
            dists = np.linalg.norm(phi - phi[i], axis=1)
            neighbors = np.argsort(dists)[1:k+1]
            for j in neighbors:
                dist_sq = dists[j] ** 2
                w_val = np.exp(-dist_sq / (sigma ** 2)) * np.exp(-kappa * dist_sq)
                W[i, j] = w_val
                W[j, i] = w_val
                
        # Normalized Laplacian L = I - D^{-1/2} W D^{-1/2}
        degrees = np.sum(W, axis=1)
        D_inv_sqrt = np.zeros(n)
        for i in range(n):
            if degrees[i] > 0:
                D_inv_sqrt[i] = 1.0 / np.sqrt(degrees[i])
        D_inv_sqrt_mat = np.diag(D_inv_sqrt)
        
        L = np.eye(n) - D_inv_sqrt_mat @ W @ D_inv_sqrt_mat
        
        # e^{-alpha L}
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        exp_diag = np.exp(-alpha * eigenvalues)
        exp_L = eigenvectors @ np.diag(exp_diag) @ eigenvectors.T
        
        # Compute diffused embeddings: phi_alpha = exp_L @ phi
        phi_alpha = exp_L @ phi
        
        # Store diffused embeddings and normalize them
        for i, doc in enumerate(self.documents):
            v = phi_alpha[i]
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            doc["diffused_embedding"] = v
            doc["embedding"] = v

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
            raw_emb = doc.get("raw_embedding")
            if raw_emb is None:
                raw_emb = doc["embedding"]
            similarity = float(np.dot(emb_arr, raw_emb))
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
            "raw_embedding": emb_arr,
            "diffused_embedding": emb_arr,
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

        # Run RAG-VM spectral diffusion offline
        self._update_diffused_embeddings()

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

        # Ensure diffused embeddings are up to date
        self._update_diffused_embeddings()

        results = []
        for doc in self.documents:
            # Dual-index retrieval
            # 1. Cosine similarity on raw embeddings
            raw_emb = doc.get("raw_embedding")
            if raw_emb is None:
                raw_emb = doc["embedding"]
            cosine_sim_raw = float(np.dot(q_arr, raw_emb))
            
            # 2. Cosine similarity on diffused embeddings (RAG-VM)
            diff_emb = doc.get("diffused_embedding")
            if diff_emb is None:
                diff_emb = doc["embedding"]
            cosine_sim_diff = float(np.dot(q_arr, diff_emb))
            
            # Combine raw and diffused similarities (50% each)
            lambda_diff = 0.5
            cosine_sim = (1.0 - lambda_diff) * cosine_sim_raw + lambda_diff * cosine_sim_diff
            
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
