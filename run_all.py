import subprocess
import sys

def run_command(command, description):
    print("=" * 80)
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)
    
    # Run command and print output live
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[Error] {description} failed with return code {process.returncode}")
        sys.exit(process.returncode)
    else:
        print(f"\n[Success] {description} completed successfully.\n")

def main():
    python_bin = ".venv/bin/python"
    
    # 1. Run Unit Tests
    run_command(
        [python_bin, "-m", "unittest", "test_lsystem_regenerator.py"],
        "Running Unit Tests for L-System Regenerator"
    )
    
    # 2. Run Emergent Cognition Simulation
    run_command(
        [python_bin, "simulate_emergent_cognition.py"],
        "Running Emergent Cognition Simulation (simulate_emergent_cognition.py)"
    )
    
    # 3. Run Final Document Report Generator
    run_command(
        [python_bin, "generate_final_docx.py"],
        "Generating Final Word Document Report (generate_final_docx.py)"
    )
    
    print("=" * 80)
    print("ALL PIPELINE STEPS EXECUTED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
