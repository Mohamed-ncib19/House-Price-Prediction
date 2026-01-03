import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n{'='*20} Running {os.path.basename(script_path)} {'='*20}")
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}")
        sys.exit(1)

def main():
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")
    
    # Scripts to run in order
    pipeline = [
        os.path.join(scripts_dir, "train.py"),    # Preprocess is imported by train.py
        os.path.join(scripts_dir, "evaluate.py"),
        os.path.join(scripts_dir, "predict.py")
    ]
    
    print("Starting House Price Prediction Pipeline...")
    
    for script in pipeline:
        run_script(script)
        
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
