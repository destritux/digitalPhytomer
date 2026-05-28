import sys
import subprocess
import tempfile
import ast
import os
import time

# We conditionally import resource to ensure OS compatibility (e.g. Linux)
try:
    import resource
except ImportError:
    resource = None

class RestrictedPythonExecutor:
    """
    Executes generated Python code inside a restricted subprocess environment with
    CPU time limits, memory limits, and blocked module imports.
    """
    def __init__(self, max_cpu_time=3, max_memory_mb=128):
        self.max_cpu_time = max_cpu_time
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.blocked_modules = ["os", "subprocess", "socket", "shutil", "pathlib", "sys", "builtins.open"]

    def _limit_resources(self):
        """Called in the child process just before exec to enforce limits (Unix-only)."""
        if resource:
            try:
                # Enforce CPU time (hard and soft limits)
                resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_time, self.max_cpu_time + 1))
                # Enforce virtual memory (address space) limit
                resource.setrlimit(resource.RLIMIT_AS, (self.max_memory_bytes, self.max_memory_bytes))
            except Exception as e:
                # Fallback silent exit if settings fail in child process
                sys.exit(99)

    def pre_run_ast_scan(self, code_str: str) -> tuple[bool, str]:
        """Scans code AST to block dangerous imports and file opens before execution."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"

        for node in ast.walk(tree):
            # Block 'import os', 'import subprocess', etc.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.blocked_modules:
                        return False, f"Security Violation: Import of '{alias.name}' is prohibited."
            # Block 'from os import ...'
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.blocked_modules:
                    return False, f"Security Violation: Import from '{node.module}' is prohibited."
            # Block calls to 'open()' or 'eval()'
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["open", "eval", "exec", "globals", "locals"]:
                        return False, f"Security Violation: Call to builtin function '{node.func.id}' is prohibited."
        return True, "AST scan passed"

    def execute(self, code_str: str) -> dict:
        """
        Executes code inside a restricted environment.
        Returns a structured metadata dict.
        """
        start_time = time.time()
        
        # 1. Static Security Scan
        is_safe, scan_msg = self.pre_run_ast_scan(code_str)
        if not is_safe:
            return {
                "syntax_valid": False,
                "runtime_success": False,
                "execution_time_s": 0.0,
                "stdout": "",
                "stderr": scan_msg,
                "security_violation": True
            }

        # Boilerplate prepended to disable imports dynamically
        sandbox_prep = (
            "import sys\n"
            "def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
            f"    blocked = {self.blocked_modules}\n"
            "    if name.split('.')[0] in blocked:\n"
            "        raise ImportError(f'Import of module {name} is blocked for security.')\n"
            "    return original_import(name, globals, locals, fromlist, level)\n"
            "original_import = __import__\n"
            "__builtins__.__import__ = restricted_import\n\n"
        )
        
        full_code = sandbox_prep + code_str

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp:
            temp.write(full_code)
            temp_path = temp.name

        try:
            res = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                preexec_fn=self._limit_resources if resource else None,
                timeout=self.max_cpu_time + 1
            )
            runtime_success = (res.returncode == 0)
            stderr = res.stderr
            stdout = res.stdout
        except subprocess.TimeoutExpired:
            runtime_success = False
            stdout = ""
            stderr = "Execution timed out (limits exceeded)."
        except Exception as e:
            runtime_success = False
            stdout = ""
            stderr = f"Sandbox startup failure: {str(e)}"
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

        return {
            "syntax_valid": True,
            "runtime_success": runtime_success,
            "execution_time_s": time.time() - start_time,
            "stdout": stdout,
            "stderr": stderr,
            "security_violation": False
        }

class CognitiveVerifier:
    def __init__(self):
        self.executor = RestrictedPythonExecutor()

    def verify_solution(self, problem: str, answer_text: str, verifier_fn=None) -> tuple[bool, str, dict]:
        """
        Verifies LLM solutions.
        Returns (is_correct, status_msg, execution_metadata).
        """
        # If output is code block, extract and sandbox execute it
        if "```python" in answer_text:
            from evaluate_blackbox import extract_python_code
            code = extract_python_code(answer_text)
            res = self.executor.execute(code)
            if not res["runtime_success"]:
                return False, f"Code failed execution:\n{res['stderr']}", res
            
            # If a custom verifier function is supplied (e.g. check return value/stdout)
            if verifier_fn:
                try:
                    is_correct = verifier_fn(res["stdout"])
                    msg = "Logical validation passed." if is_correct else "Logical validation failed."
                    return is_correct, msg, res
                except Exception as e:
                    return False, f"Verifier check crashed: {str(e)}", res
            return True, "Code ran successfully.", res

        # Otherwise verify the raw text
        if verifier_fn:
            try:
                is_correct = verifier_fn(answer_text)
                msg = "Text validation passed." if is_correct else "Text validation failed."
                return is_correct, msg, {"syntax_valid": True, "runtime_success": is_correct}
            except Exception as e:
                return False, f"Text verifier check crashed: {str(e)}", {"syntax_valid": True, "runtime_success": False}

        return True, "Solution parsed as valid.", {"syntax_valid": True, "runtime_success": True}
