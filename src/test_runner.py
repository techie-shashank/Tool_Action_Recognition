import os, sys
import json
import subprocess
import itertools
import time

import os

cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define all variations
models = ["lstm", "tcn"]
tools = ["electric_screwdriver", "pneumatic_rivet_gun", "pneumatic_screwdriver"]
data_balancing_options = [["oversample"], ["focal_loss"], [], ["resample"], ["weighted_sampling"], ["augment"]]
semi_supervised_strategies = [
    {"active": False},
    {"active": True, "strategy": "contrastive", "labelled_ratio": 0.25},
    {"active": True, "strategy": "pseudo_labeling", "labelled_ratio": 0.25}
]

config_path = os.path.join(r'../', "config.json")
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

def update_config(data_balancing, semi_supervised):
    with open(config_path, 'r') as f:
        config = json.load(f)

    config['data_balancing'] = data_balancing
    config['semi_supervised'].update(semi_supervised)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def run_test(model, tool, data_balancing, semi_supervised):
    # Update config.json
    update_config(data_balancing, semi_supervised)

    # Build the command
    cmd = [
        "python3", "main.py",
        "--model", model,
        "--tool", tool,
        "--sensor", "all"
    ]

    print(f"\n=== Running Test ===")
    print(f"Model: {model}, Tool: {tool}, Sensor: All")
    print(f"Data Balancing: {data_balancing}")
    print(f"Semi-supervised: {semi_supervised}")
    print(f"Command: {' '.join(cmd)}")

    # Directory of this script (test_runner.py)
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # Copy current environment variables and add PYTHONPATH pointing to src/
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir


    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=src_dir, env=env)
        print("✅ Test Passed")
    except subprocess.CalledProcessError as e:
        print("❌ Test Failed")
        print("Error Output:", e.stderr)
    time.sleep(1)  # Optional: Pause between runs to avoid system overload

def main():
    for model, tool, data_balancing, semi_supervised in itertools.product(
        models, tools, data_balancing_options, semi_supervised_strategies
    ):
        run_test(model, tool, data_balancing, semi_supervised)

if __name__ == "__main__":
    main()
