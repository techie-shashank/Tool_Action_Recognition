import os
import json
import re

def extract_accuracy(report_path):
    """
    Extracts the 'accuracy' value from a classification_report.txt file.
    """
    try:
        with open(report_path, 'r') as f:
            content = f.read()
        match = re.search(r'accuracy\s+([\d.]+)', content)
        if match:
            return float(match.group(1))
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading {report_path}: {e}")
        return None


def extract_config(config_path):
    """
    Extracts relevant configuration details from a config.json file.
    Returns (display_strategy, labelled_ratio, is_semi_active_flag).
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        semi_supervised_config = config.get('semi_supervised', {})

        is_semi_active = semi_supervised_config.get('active', False)
        strategy = semi_supervised_config.get('strategy', 'no_strategy')
        labelled_ratio = semi_supervised_config.get('labelled_ratio', 1.0)

        # Map internal strategy names to desired display names
        if not is_semi_active:
            display_strategy = "No Semi-supervision"
        elif strategy == "pseudo_labeling":
            display_strategy = "Pseudo-Labeling"
        elif strategy == "contrastive":
            display_strategy = "Contrastive Learning"
        else:
            display_strategy = f"Unknown: {strategy}"

        return display_strategy, labelled_ratio, is_semi_active
    except FileNotFoundError:
        return None, None, None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {config_path}")
        return None, None, None
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
        return None, None, None

if __name__ == "__main__":

    experiments_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments'))

    # Sets to collect all unique parameters found in runs
    all_found_models = set()
    all_found_strategies = set()
    all_found_ratios = set()

    collected_run_data = {}  # (model_display_name, ratio, strategy_display_name) -> accuracy

    print("Scanning experiment runs to discover all configurations...\n")
    model_dir_name_map = {'tcn': 'TCN', 'lstm': 'LSTM'}  # Mapping internal names to display names

    if not os.path.isdir(experiments_base_path):
        print(f"Error: Experiments directory '{experiments_base_path}' not found. Please ensure the path is correct.")
    else:
        for model_internal_name in os.listdir(experiments_base_path):
            model_path = os.path.join(experiments_base_path, model_internal_name)
            if not os.path.isdir(model_path) or not model_internal_name.startswith(('tcn', 'lstm')):
                continue

            display_model_name = model_dir_name_map.get(model_internal_name, model_internal_name.upper())
            all_found_models.add(display_model_name)

            run_folders = []
            for run_name in os.listdir(model_path):
                if os.path.isdir(os.path.join(model_path, run_name)) and run_name.startswith('run_'):
                    try:
                        run_number = int(run_name.split('_')[1])
                        run_folders.append((run_number, run_name))
                    except ValueError:
                        continue

            # Sort runs by number in descending order to process latest first
            run_folders.sort(key=lambda x: x[0], reverse=True)

            for _, run_name in run_folders:
                run_path = os.path.join(model_path, run_name)
                config_path = os.path.join(run_path, 'config.json')
                report_path = os.path.join(run_path, 'metrics_results', 'classification_report.txt')

                display_strategy, labelled_ratio, is_semi_active = extract_config(config_path)
                accuracy = extract_accuracy(report_path)

                if display_strategy and labelled_ratio is not None and accuracy is not None and is_semi_active is not None:
                    # Store the latest accuracy for this exact combination
                    combo_key = (display_model_name, labelled_ratio, display_strategy)
                    if combo_key not in collected_run_data:  # Only store if not already set by a newer run
                        collected_run_data[combo_key] = f"{accuracy:.4f}"
                        all_found_strategies.add(display_strategy)
                        all_found_ratios.add(labelled_ratio)

    if not collected_run_data:
        print("No valid experiment data found in 'experiments' directory to generate the table.")
        exit()

    unique_models_sorted = sorted(list(all_found_models))
    unique_ratios_sorted = sorted(list(all_found_ratios))

    # Order strategies for columns, prioritizing "No Semi-supervision" and then alphabetical
    ordered_strategies = []
    if "No Semi-supervision" in all_found_strategies:
        ordered_strategies.append("No Semi-supervision")
        all_found_strategies.remove("No Semi-supervision")
    ordered_strategies.extend(sorted(list(all_found_strategies)))

    # Initialize the table data with "N/A"
    results_table_formatted = {}
    for model_name in unique_models_sorted:
        results_table_formatted[model_name] = {}
        for ratio_val in unique_ratios_sorted:
            results_table_formatted[model_name][ratio_val] = {strategy: "N/A" for strategy in ordered_strategies}

    # Populate the table data with collected accuracies
    for (model, ratio, strategy), acc_val in collected_run_data.items():
        results_table_formatted[model][ratio][strategy] = acc_val

    print("Generating results table...\n")

    # Print the Markdown table
    header_cols = ["Model", "Labelling Dataset Ratio"] + [f"{s} (Accuracy)" for s in ordered_strategies]
    print("| " + " | ".join(header_cols) + " |")
    print("| " + " | ".join([":----"] * len(header_cols)) + " |")

    for model_name in unique_models_sorted:
        for ratio_val in unique_ratios_sorted:
            row_data = [model_name, f"{ratio_val:.2f}"]
            for strategy in ordered_strategies:
                row_data.append(results_table_formatted[model_name][ratio_val][strategy])
            print("| " + " | ".join(row_data) + " |")

    print("\nTable generation complete.")
