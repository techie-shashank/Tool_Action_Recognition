import os
import sys
import json
import subprocess
import time
import pandas as pd

from generate_plots import generate_all_plots
from main import train_and_test

# ========== Setup Paths ==========
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)

plot_runs_path = os.path.join(cwd, "..", "plot_runs.json")
config_path = os.path.join(cwd, "../", "config.json")

results_dir = os.path.join(cwd, "..", "results")
os.makedirs(results_dir, exist_ok=True)

# Path to results CSV inside the new directory
results_path = os.path.join(results_dir, "experiment_results.csv")

# ========== Config Handling ==========
def update_config(data_balancing, semi_supervised, data_ratio):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['data_balancing'] = data_balancing
    config['semi_supervised'].update(semi_supervised)
    config['data_ratio'] = data_ratio
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def run_exists(df, model, tool, sensor, data_balancing, semi_supervised, data_ratio):
    data_bal_str = ",".join(data_balancing) if data_balancing else "none"
    sensor_str = sensor if isinstance(sensor, str) else ",".join(sensor)
    semi_sup_str = json.dumps(semi_supervised, sort_keys=True)

    mask = (
        (df["model"] == model) &
        (df["tool"] == tool) &
        (df["sensor"] == sensor_str) &
        (df["Data Ratio"] == data_ratio) &
        (df["data_balancing"] == data_bal_str) &
        (df["semi_supervised"] == semi_sup_str)
    )
    return mask.any()


# ========== Parse Classification Report ==========
def parse_classification_report(file_path):
    metrics = {
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0
    }
    class_f1_scores = {}

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts or len(parts) < 2:
            continue
        if parts[0].lower() == "accuracy":
            try:
                metrics["accuracy"] = float(parts[-2])
            except:
                continue
        elif parts[0].lower() == "macro" and "avg" in parts[1]:
            try:
                metrics["macro_precision"] = float(parts[2])
                metrics["macro_recall"] = float(parts[3])
                metrics["macro_f1"] = float(parts[4])
            except:
                continue
        elif parts[0].lower() not in ["accuracy", "macro", "weighted", "micro"] and len(parts) >= 4:
            class_label = parts[0]
            try:
                f1 = float(parts[3])
                class_f1_scores[f"f1_{class_label}"] = f1
            except:
                continue

    return {**metrics, **class_f1_scores}

# ========== Get Latest Run Directory ==========
def get_latest_run_path(model):
    model_dir = os.path.join(cwd, "../", "experiments", model)
    if not os.path.exists(model_dir):
        return None
    run_dirs = [d for d in os.listdir(model_dir) if d.startswith("run_")]
    run_dirs = sorted(run_dirs, key=lambda x: int(x.split('_')[-1]), reverse=True)
    if not run_dirs:
        return None
    return os.path.join(model_dir, run_dirs[0], "metrics_results", "classification_report.txt"), run_dirs[0].split("_")[-1]

# ========== Run One Experiment ==========
def run_test(model, tool, sensor, data_balancing, semi_supervised, data_ratio):
    # Load results dataframe
    df = pd.read_csv(results_path)

    sensor_str = sensor if isinstance(sensor, str) else ",".join(sensor)

    if run_exists(df, model, tool, sensor_str, data_balancing, semi_supervised, data_ratio):
        print(
            f"⏭️ Skipping run: model={model}, tool={tool}, sensor={sensor_str}, data_ratio={data_ratio}, semi_supervised={semi_supervised}")
        return

    # Proceed with running test
    update_config(data_balancing, semi_supervised, data_ratio)

    print(f"\n=== Running Test ===")
    print(f"Model: {model}, Tool: {tool}, Sensor: {sensor_str}")
    print(f"Data Balancing: {data_balancing}")
    print(f"Semi-supervised: {semi_supervised}")

    try:
        train_and_test(model, tool, sensor_str)
        print("✅ Test Passed")

        report_path, run_no = get_latest_run_path(model)
        if report_path and os.path.exists(report_path):
            metrics = parse_classification_report(report_path)
        else:
            print("⚠️  No classification report found!")
            metrics = {"accuracy": 0.0, "macro_f1": 0.0}

        # Create result row
        new_row = {
            "Run No.": run_no,
            "model": model,
            "tool": tool,
            "sensor": sensor_str,
            "Data Ratio": data_ratio,
            "data_balancing": ",".join(data_balancing) if data_balancing else "none",
            "semi_supervised": json.dumps(semi_supervised, sort_keys=True),
            "strategy": semi_supervised.get("strategy", "none"),
            "labelled_ratio": semi_supervised.get("labelled_ratio", 1.0),
        }
        new_row.update(metrics)

        # Add any new class-level columns if missing
        for key in metrics:
            if key.startswith("f1_") and key not in df.columns:
                df[key] = None

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(results_path, index=False)

    except subprocess.CalledProcessError as e:
        print("❌ Test Failed")
        print("Error Output:", e.stderr)

    time.sleep(1)

def load_plot_configs(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    plot_configs = load_plot_configs(plot_runs_path)
    for config in plot_configs:
        run_test(
            config["model"],
            config["tool"],
            config["sensor"],
            config.get("data_balancing", []),
            config.get("semi_supervised", {"active": False}),
            config.get("data_ratio", 1.0)
        )
    generate_all_plots(results_path, os.path.join(results_dir, "plots"))

if __name__ == "__main__":
    main()
