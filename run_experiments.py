import subprocess
import sys
import itertools

# --- DEFINITION OF EXPERIMENTS ---
# Add whatever values you want to test here
param_grid = {
    "lr": [1e-3, 5e-4],
    "nodes": [10, 20],
    "seed": [42, 101] # Run twice to verify stability
}

def run_grid():
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"--- Queued {len(combinations)} Experiments ---")
    
    for i, params in enumerate(combinations):
        print(f"\n[Job {i+1}/{len(combinations)}] Starting: {params}")
        
        command = [sys.executable, "main.py"]
        
        # Add params to command
        for key, value in params.items():
            command.append(f"--{key}")
            command.append(str(value))
            
        # Optional: Add common flags (e.g., lower episodes for testing)
        command.extend(["--episodes", "500"])
        
        # Execute
        subprocess.run(command)
        
    print("\n--- ALL JOBS FINISHED ---")

if __name__ == "__main__":
    run_grid()