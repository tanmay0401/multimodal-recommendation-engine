import subprocess
import os
import time

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    start_time = time.time()
    # Set PYTHONPATH to the current directory so that 'src' is discoverable
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    result = subprocess.run(command, shell=True, env=env)
    end_time = time.time()
    
    if result.returncode != 0:
        print(f"ERROR: Step failed with return code {result.returncode}")
        exit(1)
    
    print(f"COMPLETED in {end_time - start_time:.2f} seconds")

def main():
    if not os.path.exists("data/embeddings.npy"):
        print("Error: data/embeddings.npy not found. Please wait for embed.py to finish.")
        return

    # Phase 2: Training Retrieval Model
    run_step("python src/train.py", "Training Two-Tower Retrieval Model")

    # Phase 3: Indexing
    run_step("python src/index.py", "Building FAISS Index")

    # Phase 4: Ranking
    run_step("python src/rank_data.py", "Generating Ranking Dataset")
    run_step("python src/rank.py", "Training LightGBM Ranker")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print("You can now start the API using: python src/api.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
