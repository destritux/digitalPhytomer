import requests
import json
import re

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", default_model="qwen2.5:0.5b"):
        self.base_url = base_url
        self.default_model = default_model
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.model_stats = {}

    def pull_model(self, model_name=None):
        if model_name is None:
            model_name = self.default_model
        
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
                
                prompt_tokens = result.get("prompt_eval_count", 0)
                completion_tokens = result.get("eval_count", 0)
                
                if prompt_tokens == 0:
                    prompt_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
                if completion_tokens == 0:
                    completion_tokens = len(response_text.split())

                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                
                if model_name not in self.model_stats:
                    self.model_stats[model_name] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                self.model_stats[model_name]["prompt_tokens"] += prompt_tokens
                self.model_stats[model_name]["completion_tokens"] += completion_tokens
                self.model_stats[model_name]["calls"] += 1
                
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
        import hashlib
        import numpy as np
        dimensions = 256
        vec = [0.0] * dimensions
        
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec
            
        for w in words:
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % dimensions
            sign = 1 if ((h // dimensions) % 2 == 0) else -1
            vec[idx] += sign
            
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
                "qwen2.5:1.5b": 2.5
            }
        
        total = 0.0
        for model, stats in self.model_stats.items():
            weight = weight_map.get(model, 1.0)
            tokens = stats["prompt_tokens"] + stats["completion_tokens"] * 1.5
            total += tokens * weight
        
        # Add basic count for general client usage if stats are empty
        if total == 0:
            total = self.total_prompt_tokens + self.total_completion_tokens * 1.5
            
        return float(total)

    def reset_stats(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        self.model_stats = {}
