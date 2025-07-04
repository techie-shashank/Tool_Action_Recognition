import pandas as pd
import seaborn as sns
import os
import sys
import json

import matplotlib.pyplot as plt
from pandas.plotting import table

# Settings
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


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


def plot_sensor_ablation_study(df: pd.DataFrame, output_dir: str):
    """
    Creates grouped bar plots of macro-F1 scores for different sensors and tools,
    showing which sensors are important for which tool.
    """
    # Filter relevant columns and rows (tools and sensors)
    df_ablation = df[
        (df["model"].isin(["lstm"])) &  # You can expand or parameterize models if needed
        (df["tool"].notna()) &
        (df["sensor"].notna()) &
        (df["sensor"] != "all")
    ].copy()

    if df_ablation.empty:
        print("⚠️ No data available for sensor ablation study plot.")
        return

    # Simplify sensor list if stored as comma-separated strings or lists
    def sensor_label(sensor_val):
        if isinstance(sensor_val, str):
            # Make labels consistent, e.g. replace ',' with '+'
            return sensor_val.replace(",", "+")
        return str(sensor_val)

    df_ablation["sensor_label"] = df_ablation["sensor"].apply(sensor_label)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_ablation,
        x="sensor_label",
        y="macro_f1",
        hue="tool",
        ci="sd",
        capsize=0.1
    )
    plt.title("Ablation Study: Sensor Importance per Tool (Macro-F1 Scores)")
    plt.xlabel("Sensor Set")
    plt.ylabel("Macro-F1 Score")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Tool")
    plt.grid(axis="y")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "sensor_ablation_study.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✅ Saved sensor ablation study plot: {output_path}")


def plot_confidence_threshold_ablation(df: pd.DataFrame, output_dir: str):
    """
    Plot Macro-F1 scores vs confidence threshold for pseudo-labeling strategy.
    Assumes confidence threshold is present in the 'semi_supervised' JSON field.
    """
    import json

    # Extract confidence threshold from semi_supervised column
    def extract_threshold(row):
        try:
            config = json.loads(row["semi_supervised"]) if isinstance(row["semi_supervised"], str) else row["semi_supervised"]
            if isinstance(config, dict):
                return config.get("threshold", None)
        except Exception:
            return 0.9

    df = df.copy()
    df["threshold"] = df.apply(extract_threshold, axis=1)

    # Filter for pseudo-labeling runs with defined thresholds
    df_thresh = df[
        (df["strategy"] == "pseudo_labeling") &
        (df["threshold"].notna()) &
        (df["labelled_ratio"].isin([0.25]))  # optionally restrict to 25%
    ]

    if df_thresh.empty:
        print("⚠️ No pseudo-labeling data with confidence threshold found.")
        return

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_thresh,
        x="threshold",
        y="macro_f1",
        hue="model",
        marker="o",
        style="labelled_ratio",
        palette="Set2"
    )
    plt.title("Pseudo-Labeling: Confidence Threshold Ablation")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Macro-F1 Score")
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "pseudo_label_conf_threshold.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved pseudo-label confidence threshold ablation plot: {output_path}")


def plot_ssl_per_class_performance(df, output_dir, labelled_ratio, title):
    """
    Bar plots of per-class F1 scores for SSL methods at 10% and optionally 25% labeled data.
    """

    df["semi_supervised_dict"] = df["semi_supervised"].apply(lambda x: json.loads(x) if pd.notna(x) else {"active": False})

    # Filter for SSL active and labelled_ratio in [0.1, 0.25]
    df_ssl = df[
        (df["semi_supervised_dict"].apply(lambda d: d.get("active", False)) == True) &
        (df["labelled_ratio"].isin(labelled_ratio)) &
        (df["tool"] == "electric_screwdriver") &
        (df["sensor"].astype(str).str.contains("all"))
    ].copy()

    if df_ssl.empty:
        print("⚠️ No data available for SSL per-class performance plot.")
        return

    # Identify per-class f1 score columns dynamically
    per_class_cols = [col for col in df_ssl.columns if col.startswith("f1_")]
    if not per_class_cols:
        print("⚠️ No per-class F1 score columns found.")
        return

    for model in ["lstm", "tcn"]:
        subset = df_ssl[df_ssl["model"] == model]
        if subset.empty:
            continue

        # Melt the dataframe to long format for seaborn barplot
        # Keep method and labelled_ratio for grouping
        plot_df = subset.melt(
            id_vars=["method", "labelled_ratio"],
            value_vars=per_class_cols,
            var_name="Class",
            value_name="F1 Score"
        )

        # Clean class names, e.g., f1_class1 -> class1
        plot_df["Class"] = plot_df["Class"].str.replace("f1_", "")

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=plot_df,
            x="Class",
            y="F1 Score",
            hue="method",
            ci=None,
            palette="Set2"
        )
        plt.title(f"{model.upper()} - {title}")
        plt.xlabel("Class")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1)
        plt.legend(title="SSL Method")
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{model}_ssl_per_class_f1.png")
        plt.savefig(output_path)
        plt.close()
        print(f"✅ Saved: {output_path}")


def parse_ssl_config(row):
    if isinstance(row["semi_supervised"], str):
        return json.loads(row["semi_supervised"])
    return row["semi_supervised"]

def format_aug_list(aug_list):
    if not aug_list:
        return "None"
    return "+".join(sorted(aug_list))

def plot_contrastive_augmentation_ablation(df: pd.DataFrame, output_dir: str):
    # Parse and extract config
    df["semi_supervised"] = df.apply(parse_ssl_config, axis=1)
    df["augmentations"] = df["semi_supervised"].apply(lambda cfg: format_aug_list(cfg.get("contrastive_augmentations", [])))
    df["labelled_ratio"] = df["semi_supervised"].apply(lambda cfg: cfg.get("labelled_ratio", None))

    # Filter contrastive results at 10% labeled
    df_contrastive = df[
        (df["method"] == "Contrastive") &
        (df["labelled_ratio"] == 0.25)
    ].copy()

    if df_contrastive.empty:
        print("⚠️ No contrastive augmentation data found for ablation.")
        return

    # Group and average
    plot_data = (
        df_contrastive.groupby("augmentations")["macro_f1"]
        .mean()
        .reset_index()
        .sort_values("macro_f1", ascending=False)
    )

    # Generate short labels
    short_labels = {aug: f"A{i+1}" for i, aug in enumerate(plot_data["augmentations"])}
    plot_data["short_label"] = plot_data["augmentations"].map(short_labels)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_data, x="short_label", y="macro_f1", palette="viridis")
    plt.title("Contrastive Learning: Augmentation Ablation (10% Labeled)")
    plt.xlabel("Augmentation Config (A1, A2...)")
    plt.ylabel("Macro-F1 Score")
    plt.tight_layout()

    # Add legend outside
    legend_entries = [f"{short}: {full}" for full, short in short_labels.items()]
    plt.figtext(0.99, 0.01, "\n".join(legend_entries), horizontalalignment='right', fontsize=8, va="bottom")

    output_path = os.path.join(output_dir, "contrastive_augmentation_ablation.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved augmentation ablation plot to: {output_path}")


def plot_mean_teacher_consistency_ablation(df: pd.DataFrame, output_dir: str):
    """
    Plots macro F1 vs consistency weight for Mean Teacher strategy.
    """
    # Parse and extract config
    df["semi_supervised"] = df.apply(parse_ssl_config, axis=1)
    df["labelled_ratio"] = df["semi_supervised"].apply(lambda cfg: cfg.get("labelled_ratio", None))
    df["consistency_weight"] = df["semi_supervised"].apply(lambda cfg: cfg.get("consistency_weight", None))

    # Filter Mean Teacher at 25% labeled
    df_mt = df[
        (df["method"] == "Mean Teacher") &
        (df["labelled_ratio"] == 0.25)
    ].copy()

    if df_mt.empty:
        print("⚠️ No data available for Mean Teacher consistency weight ablation.")
        return

    # Sort weights numerically for cleaner plot
    df_mt["consistency_weight"] = pd.to_numeric(df_mt["consistency_weight"], errors='coerce')
    df_mt = df_mt.dropna(subset=["consistency_weight"])

    for model in df_mt["model"].unique():
        subset = df_mt[df_mt["model"] == model]

        if subset.empty:
            continue

        plt.figure(figsize=(6, 4))
        sns.lineplot(
            data=subset.sort_values("consistency_weight"),
            x="consistency_weight",
            y="macro_f1",
            marker="o"
        )
        plt.title(f"{model.upper()} - Mean Teacher: Consistency Weight Ablation")
        plt.xlabel("Consistency Weight")
        plt.ylabel("Macro-F1 Score")
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{model}_mean_teacher_consistency_ablation.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved: {save_path}")




def generate_ssl_comparison_table(df: pd.DataFrame, output_dir: str):
    """
    Generates a macro-F1 comparison table for each SSL method at various labeled data fractions,
    and saves both a CSV and a PNG image.
    """
    df["semi_supervised_dict"] = df["semi_supervised"].apply(
        lambda x: json.loads(x) if pd.notna(x) else {"active": False}
    )

    df_filtered = df[
        (df["tool"] == "electric_screwdriver") &
        (df["sensor"].astype(str).str.contains("all"))
    ].copy()

    df_filtered["x_fraction"] = df_filtered.apply(
        lambda row: row["Data Ratio"] if not row["semi_supervised_dict"].get("active", False)
        else row["labelled_ratio"],
        axis=1
    )

    df_filtered["x_fraction"] = df_filtered["x_fraction"].round(2)

    comparison = df_filtered[
        df_filtered["x_fraction"].isin([0.1, 0.25, 0.5, 1.0])
    ][["method", "x_fraction", "macro_f1"]]

    # Pivot and format table
    comparison_table = comparison.pivot_table(
        index="method",
        columns="x_fraction",
        values="macro_f1",
        aggfunc="mean"
    ).reindex(columns=[0.1, 0.25, 0.5, 1.0])

    comparison_table.columns = ["10%", "25%", "50%", "100%"]
    comparison_table = comparison_table.round(3)

    # Save CSV
    csv_path = os.path.join(output_dir, "ssl_comparison_summary.csv")
    comparison_table.to_csv(csv_path)
    print(f"✅ Saved table as CSV: {csv_path}")

    # Save as image
    fig, ax = plt.subplots(figsize=(8, 2 + 0.5 * len(comparison_table)))
    ax.axis("off")
    tbl = table(ax, comparison_table, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)

    image_path = os.path.join(output_dir, "ssl_comparison_summary.png")
    plt.tight_layout()
    plt.savefig(image_path, dpi=300)
    plt.close()
    print(f"🖼️ Saved table as image: {image_path}")


def generate_all_plots(csv_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess CSV
    df = pd.read_csv(csv_path)
    df["method"] = df.apply(label_method, axis=1)

    # plot_ssl_vs_label_fraction(df, output_dir)
    #
    # title = "Per-Class F1 Scores for SSL Methods at 25% Labeled Data"
    # plot_ssl_per_class_performance(df, output_dir, [0.25], title)

    # generate_ssl_comparison_table(df, output_dir)

    # plot_sensor_ablation_study(df, output_dir)

    # plot_confidence_threshold_ablation(df, output_dir)

    # plot_contrastive_augmentation_ablation(df, output_dir)

    plot_mean_teacher_consistency_ablation(df, output_dir)

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cwd)
    results_dir = os.path.join(cwd, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Path to results CSV inside the new directory
    results_path = os.path.join(results_dir, "experiment_results.csv")
    generate_all_plots(results_path, os.path.join(results_dir, "plots"))
