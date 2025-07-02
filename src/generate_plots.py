import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Settings
sns.set(style="whitegrid")
plt.rcParams["figure.dpi"] = 150


def generate_all_plots(csv_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    df = pd.read_csv(csv_path)

    # Helper: Clean method names for plotting
    def label_method(row):
        # Your CSV columns: "semi_supervised" (bool) and "strategy" (string)
        if not row.get("semi_supervised", False):
            return "Supervised"
        method = row.get("strategy", "")
        return {
            "mean_teacher": "Mean Teacher",
            "pseudo_label": "Pseudo Labeling",
            "contrastive": "Contrastive"
        }.get(method, method.title())

    df["method"] = df.apply(label_method, axis=1)

    # ========= 1. Bar Plot: Data Balancing Methods =========
    def plot_balancing():
        df_bal = df[
            (df.tool == "electric_screwdriver") &
            (df.sensor == "all") &
            (~df.data_balancing.isnull()) &
            (~df.data_balancing.str.strip().eq("[]")) &
            (df.semi_supervised == False)
        ].copy()

        df_bal["balancing"] = (
            df_bal["data_balancing"]
            .str.strip("[]")
            .str.replace("'", "")
            .str.replace("_", " ")
            .str.title()
        )

        plt.figure(figsize=(8, 4))
        sns.barplot(data=df_bal, x="balancing", y="macro_f1", hue="model")
        plt.title("Macro-F1 Score vs Data Balancing Technique")
        plt.ylabel("Macro-F1")
        plt.xlabel("Balancing Method")
        plt.xticks(rotation=30)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/data_balancing_barplot.png")
        plt.close()

    # ========= 2. Table: LSTM vs TCN Performance =========
    def save_model_comparison_table():
        table = df[
            (df.tool == "electric_screwdriver") &
            (df.sensor == "all") &
            (df.semi_supervised == False) &
            (df.data_balancing == "[]")
        ][["model", "accuracy", "macro_f1"]]
        table.to_csv(f"{output_dir}/baseline_model_comparison.csv", index=False)

    # ========= 3. Line Plot: SSL Performance vs Label % =========
    def plot_ssl_vs_label_fraction():
        df_ssl = df[
            (df.tool == "electric_screwdriver") &
            (df.sensor == "all") &
            ((df.labelled_ratio < 1.0) | (df.semi_supervised == False))
        ]

        for model in ["lstm", "tcn"]:
            subset = df_ssl[df_ssl.model == model]
            plt.figure(figsize=(8, 4))
            sns.lineplot(
                data=subset,
                x="labelled_ratio",
                y="macro_f1",
                hue="method",
                marker="o"
            )
            plt.title(f"{model.upper()} - SSL Macro-F1 vs Labeled Data %")
            plt.xlabel("Labeled Data Fraction")
            plt.ylabel("Macro-F1")
            plt.xticks([0.1, 0.25, 0.5, 1.0])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{model}_ssl_vs_label_frac.png")
            plt.close()

    # ========= 4. Bar Plot: Per-Class F1 at X% =========
    def plot_ssl_per_class(label_ratio=0.1):
        df_bar = df[
            (df.tool == "electric_screwdriver") &
            (df.sensor == "all") &
            (df.labelled_ratio == label_ratio)
        ]

        # Using your columns starting with 'f1_'
        class_cols = [col for col in df.columns if col.startswith("f1_") and col != "f1_"]
        if not class_cols:
            print(f"⚠️ Per-class F1 data not found for {label_ratio}. Skipping.")
            return

        data = []
        for _, row in df_bar.iterrows():
            method_name = label_method(row)
            for col in class_cols:
                data.append({
                    "class": col.replace("f1_", ""),
                    "method": method_name,
                    "f1": row[col]
                })

        plot_df = pd.DataFrame(data)
        plt.figure(figsize=(10, 4))
        sns.barplot(data=plot_df, x="class", y="f1", hue="method")
        plt.title(f"Per-Class F1 Scores at {int(label_ratio * 100)}% Labeled Data")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ssl_per_class_f1_{int(label_ratio * 100)}pct.png")
        plt.close()

    # ========= 5. SSL Comparison Summary Table =========
    def save_ssl_summary_table():
        rows = []
        for method in ["Supervised", "Contrastive", "Pseudo Labeling", "Mean Teacher"]:
            row = {"method": method}
            for r in [0.1, 0.25, 0.5, 1.0]:
                subset = df[
                    (df.tool == "electric_screwdriver") &
                    (df.sensor == "all") &
                    ((df.labelled_ratio == r) if r < 1.0 else (df.labelled_ratio == 1.0)) &
                    (df["method"] == method)
                ]
                score = subset["macro_f1"].mean()
                row[f"Macro-F1@{int(r * 100)}%"] = round(score, 3) if not pd.isna(score) else "-"
            rows.append(row)
        pd.DataFrame(rows).to_csv(f"{output_dir}/ssl_summary_table.csv", index=False)

    # ========= 6. Sensor Ablation Plot =========
    def plot_sensor_ablation():
        ablation_df = df[
            (df.semi_supervised == False) &
            (df.tool == "electric_screwdriver") &
            (df.model == "lstm") &
            (df.sensor != "all")
        ].copy()
        if ablation_df.empty:
            print("⚠️ No ablation data found.")
            return

        ablation_df["Sensor"] = ablation_df["sensor"].str.title()
        plt.figure(figsize=(6, 4))
        sns.barplot(data=ablation_df, x="Sensor", y="macro_f1")
        plt.title("Sensor Ablation - LSTM")
        plt.ylabel("Macro-F1")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sensor_ablation_lstm.png")
        plt.close()

    # ========== Execute All ==========
    plot_balancing()
    # save_model_comparison_table()
    # plot_ssl_vs_label_fraction()
    # plot_ssl_per_class(label_ratio=0.1)
    # plot_ssl_per_class(label_ratio=0.25)
    # save_ssl_summary_table()
    # plot_sensor_ablation()


def get_latest_subdir(parent_dir):
    subdirs = [
        os.path.join(parent_dir, d)
        for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
    ]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=os.path.getctime)
    return latest_dir


if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(src_dir, "..", "results")
    latest_results_dir = get_latest_subdir(results_dir)
    if latest_results_dir is None:
        raise FileNotFoundError(f"No subdirectories found in {results_dir}")
    csv = os.path.join(latest_results_dir, 'experiment_results.csv')
    generate_all_plots(csv, output_dir=os.path.join(latest_results_dir, "plots"))
