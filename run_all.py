import subprocess
import sys
import time

def run_script(script_name):
    print("\n" + "="*80)
    print(f"RUNNING {script_name}...")
    print("="*80)
    start_time = time.time()
    
    # Run the script using the same python interpreter
    result = subprocess.run([sys.executable, script_name])
    
    duration = time.time() - start_time
    if result.returncode == 0:
        print(f"\n[SUCCESS] {script_name} finished in {duration:.2f}s")
        return True
    else:
        print(f"\n[FAILURE] {script_name} exited with code {result.returncode} after {duration:.2f}s")
        return False

def main():
    scripts = [
        "evaluate.py",
        "evaluate_blackbox.py",
        "evaluate_cyberdefense.py",
        "evaluate_robotics.py",
        "generate_report.py"
    ]
    
    overall_start = time.time()
    all_success = True
    
    for s in scripts:
        success = run_script(s)
        if not success:
            all_success = False
            
    overall_duration = time.time() - overall_start
    print("\n" + "="*80)
    print("ALL RUNS COMPLETE")
    print("="*80)
    print(f"Total time elapsed: {overall_duration:.2f}s")
    if all_success:
        print("Status: ALL SCRIPTS EXECUTED SUCCESSFULLY.")
    else:
        print("Status: SOME SCRIPTS ENCOUNTERED ERRORS.")

if __name__ == "__main__":
    main()
