import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Settings
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

import json

def label_method(row):
    # Parse semi_supervised JSON string to dict
    try:
        semi_sup = json.loads(row["semi_supervised"])
    except Exception:
        semi_sup = {"active": False}

    if not semi_sup.get("active", False):
        return "Supervised"
    method = row["strategy"]
    if method == "mean_teacher":
        return "Mean Teacher"
    elif method == "pseudo_label":
        return "Pseudo Labeling"
    elif method == "contrastive":
        return "Contrastive"
    else:
        return method.title()


def plot_ssl_vs_label_fraction(df: pd.DataFrame, output_dir: str):
    # Parse semi_supervised JSON strings to dicts
    df["semi_supervised_dict"] = df["semi_supervised"].apply(lambda x: json.loads(x) if pd.notna(x) else {"active": False})

    # Filter for tool and sensor
    df_ssl = df[
        (df["tool"] == "electric_screwdriver") &
        (df["sensor"].astype(str).str.contains("all"))
    ].copy()

    # Use the parsed dict's 'active' flag to filter supervised vs semi-supervised
    df_ssl = df_ssl[
        (
            (df_ssl["semi_supervised_dict"].apply(lambda d: d.get("active", False)) == False) &
            (df_ssl["Data Ratio"].isin([0.1, 0.25, 0.5, 0.75, 1.0]))  # only these for consistency
        ) |
        (
            (df_ssl["semi_supervised_dict"].apply(lambda d: d.get("active", False)) == True) &
            (df_ssl["labelled_ratio"].isin([0.1, 0.25, 0.5, 0.75]))
        )
    ]

    if df_ssl.empty:
        print("⚠️ No data available for SSL vs Label Fraction plot.")
        return

    # Create consistent x-axis column depending on supervised/semi-supervised
    df_ssl["x_fraction"] = df_ssl.apply(
        lambda row: row["Data Ratio"] if not row["semi_supervised_dict"].get("active", False) else row["labelled_ratio"],
        axis=1
    )

    for model in ["lstm", "tcn"]:
        subset = df_ssl[df_ssl["model"] == model]
        if subset.empty:
            continue

        plt.figure(figsize=(8, 4))
        sns.lineplot(
            data=subset,
            x="x_fraction",
            y="macro_f1",
            hue="method",
            style="method",
            marker="o"
        )
        plt.title(f"{model.upper()} - SSL Macro-F1 vs Labeled Data %")
        plt.xlabel("Labeled Data Fraction")
        plt.ylabel("Macro-F1 Score")
        plt.xticks([0.1, 0.25, 0.5, 0.75, 1.0], labels=["10%", "25%", "50%", "70%", "100%"])
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{model}_ssl_vs_label_frac.png")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")


def generate_all_plots(csv_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess CSV
    df = pd.read_csv(csv_path)
    df["method"] = df.apply(label_method, axis=1)

    plot_ssl_vs_label_fraction(df, output_dir)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cwd)
    results_dir = os.path.join(cwd, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Path to results CSV inside the new directory
    results_path = os.path.join(results_dir, "experiment_results.csv")
    generate_all_plots(results_path, output_dir=os.path.join(results_dir, "plots"))
