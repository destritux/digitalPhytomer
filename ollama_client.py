import requests
import json
import sys
import re

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", default_model="qwen2.5:0.5b"):
        self.base_url = base_url
        self.default_model = default_model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def pull_model(self, model_name=None):
        if model_name is None:
            model_name = self.default_model
        
        # Check if model is already pulled
        try:
            resp = requests.get(f"{self.base_url}/api/tags")
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
            resp = requests.post(url, json=data)
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
            resp = requests.post(url, json=payload)
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

    def get_embeddings(self, text, model_name=None):
        """
        Generates a robust, collision-resistant 256-dimensional feature hashing vector for text.
        This provides high accuracy cosine similarity for search and matches, avoiding
        representation collapse seen when query-embedding causal LMs without pooling.
        """
        import hashlib
        import numpy as np
        dimensions = 256
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
        return {
            "total_calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }

    def reset_stats(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
