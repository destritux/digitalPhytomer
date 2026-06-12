import requests
import json
import sys
import re
import os
import hashlib
import atexit

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", default_model="qwen2.5:0.5b"):
        self.base_url = base_url
        self.default_model = default_model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.model_stats = {}
        self.current_seed = None
        self.cache_file = "results/llm_cache.json"
        self.cache = {}
        self.load_cache()
        atexit.register(self.save_cache)

    def load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"[Ollama Cache] Loaded {len(self.cache)} entries.")
        except Exception as e:
            print(f"[Ollama Cache Warning] Failed to load cache: {e}")

    def save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Ollama Cache Warning] Failed to save cache: {e}")

    def pull_model(self, model_name=None):
        if model_name is None:
            model_name = self.default_model
        
        # Check if model is already pulled
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10.0)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                names = [m["name"] for m in models]
                if model_name in names or f"{model_name}:latest" in names:
                    print(f"[Ollama] Model '{model_name}' is already available.")
                    return True
        except Exception as e:
            print(f"[Ollama Warning] Failed to check tags: {e}")

        print(f"[Ollama] Pulling model '{model_name}'... (this might take a minute)")
        url = f"{self.base_url}/api/pull"
        data = {"name": model_name, "stream": False}
        
        try:
            resp = requests.post(url, json=data, timeout=300.0)
            if resp.status_code == 200:
                print(f"[Ollama] Successfully pulled '{model_name}'.")
                return True
            else:
                print(f"[Ollama Error] Failed to pull model: {resp.text}")
                return False
        except Exception as e:
            print(f"[Ollama Error] Network error during pull: {e}")
            return False

    def generate(self, prompt, system_prompt=None, temperature=0.7, model_name=None, max_tokens=None):
        if model_name is None:
            model_name = self.default_model
        if max_tokens is None:
            max_tokens = 256

        # Construir chave única para o cache baseada no prompt, seed, temperatura e modelo
        seed_val = str(self.current_seed) if self.current_seed is not None else "no_seed"
        cache_str = f"{prompt}||{system_prompt}||{temperature}||{model_name}||{max_tokens}||{seed_val}"
        cache_key = hashlib.md5(cache_str.encode('utf-8')).hexdigest()

        if cache_key in self.cache:
            cached = self.cache[cache_key]
            
            # Replicar impacto estatístico nas métricas (telemetria idêntica à execução real)
            self.total_calls += 1
            self.total_prompt_tokens += cached["prompt_tokens"]
            self.total_completion_tokens += cached["completion_tokens"]
            if model_name not in self.model_stats:
                self.model_stats[model_name] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
            self.model_stats[model_name]["prompt_tokens"] += cached["prompt_tokens"]
            self.model_stats[model_name]["completion_tokens"] += cached["completion_tokens"]
            self.model_stats[model_name]["calls"] += 1
            
            return {
                "text": cached["text"],
                "prompt_tokens": cached["prompt_tokens"],
                "completion_tokens": cached["completion_tokens"],
                "success": cached.get("success", True)
            }

        url = f"{self.base_url}/api/generate"
        options = {
            "temperature": temperature
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": options
        }
        if system_prompt:
            payload["system"] = system_prompt

        self.total_calls += 1
        try:
            resp = requests.post(url, json=payload, timeout=60.0)
            if resp.status_code == 200:
                result = resp.json()
                response_text = result.get("response", "")
                
                # Retrieve token counts from metadata if available (Ollama provides prompt_eval_count and eval_count)
                prompt_tokens = result.get("prompt_eval_count", 0)
                completion_tokens = result.get("eval_count", 0)
                
                # Fallback estimation if token counts are 0
                if prompt_tokens == 0:
                    prompt_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
                if completion_tokens == 0:
                    completion_tokens = len(response_text.split())

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                
                # Model-specific tracking
                if model_name not in self.model_stats:
                    self.model_stats[model_name] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                self.model_stats[model_name]["prompt_tokens"] += prompt_tokens
                self.model_stats[model_name]["completion_tokens"] += completion_tokens
                self.model_stats[model_name]["calls"] += 1
                
                # Salvar no cache
                self.cache[cache_key] = {
                    "text": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "success": True
                }

                return {
                    "text": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "success": True
                }
            else:
                print(f"[Ollama Error] Status code {resp.status_code}: {resp.text}")
                return {"text": "", "prompt_tokens": 0, "completion_tokens": 0, "success": False}
        except Exception as e:
            print(f"[Ollama Error] Connection exception: {e}")
            return {"text": "", "prompt_tokens": 0, "completion_tokens": 0, "success": False}

    def get_embeddings(self, text, model_name="nomic-embed-text"):
        """
        Generates a semantic embedding using Ollama's nomic-embed-text model.
        Falls back to a robust 768-dimensional local feature hashing if Ollama is unavailable
        or fails, preserving dimension consistency and execution safety.
        """
        cache_key = f"emb||{model_name}||{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model_name,
            "prompt": text
        }
        try:
            resp = requests.post(url, json=payload, timeout=5.0)
            if resp.status_code == 200:
                result = resp.json()
                embedding = result.get("embedding", [])
                if embedding:
                    self.cache[cache_key] = embedding
                    return embedding
        except Exception as e:
            pass

        # Fallback: Deterministic Feature Hashing of size 768 to match nomic-embed-text
        import numpy as np
        dimensions = 768
        vec = [0.0] * dimensions
        
        # Tokenize words, removing punctuation
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec
            
        for w in words:
            # Hash each word into an index and sign
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % dimensions
            sign = 1 if ((h // dimensions) % 2 == 0) else -1
            vec[idx] += sign
            
        # Normalize the vector
        vec_np = np.array(vec, dtype=float)
        norm = np.linalg.norm(vec_np)
        if norm > 0:
            vec_np = vec_np / norm
        return vec_np.tolist()

    def get_stats(self):
        weighted_tokens = self.get_weighted_tokens()
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "weighted_tokens": weighted_tokens
        }

    def get_weighted_tokens(self, weight_map=None):
        if weight_map is None:
            weight_map = {
                "qwen2.5:0.5b": 1.0,
                "qwen2.5:1.5b": 3.0
            }
        weighted = 0.0
        for mname, stats in self.model_stats.items():
            weight = weight_map.get(mname, 1.0)
            weighted += (stats["prompt_tokens"] + stats["completion_tokens"]) * weight
        return weighted

    def reset_stats(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.model_stats = {}
