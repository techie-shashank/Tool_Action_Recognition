import streamlit as st
import os
import subprocess

st.set_page_config(page_title="Tool Action Recognition Evaluator", layout="centered")
st.title("🔍 Tool Action Recognition - Insights")

# Sidebar inputs
st.sidebar.header("⚙️ Evaluation Parameters")

# Model selection
tool_models = ["fcn", "cnn", "resnet"]  # Add more supported models as needed
selected_model = st.sidebar.selectbox("Model", tool_models, index=0)

# Tool selection
tools = ["electric_screwdriver", "pneumatic_rivet_gun","pneumatic_screwdriver"]
selected_tool = st.sidebar.selectbox("Tool", tools, index=0)

# Sensor selection
sensors = ["acc", "gyr", "mag", "mic"]
selected_sensor = st.sidebar.selectbox("Sensor", sensors, index=3)

# Checkpoint path input
checkpoint_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value="C:/Users/monik/Documents/Projects/ADLTS/Tool_Action_Recognition/experiments/fcn/run_20250525_212934/model.pt"
)

# Run evaluation button
if st.sidebar.button("🚀 Run Evaluation"):
    if not os.path.exists(checkpoint_path):
        st.error("Checkpoint path is invalid or file does not exist.")
    else:
        st.info("Running evaluation... please wait")

        cmd = [
            "python", "src/evaluate.py",
            "--model", selected_model,
            "--tool", selected_tool,
            "--sensor", selected_sensor,
            "--checkpoint", checkpoint_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("✅ Evaluation completed successfully!")
            st.text("CLI Output:")
            st.code(result.stdout)

            # Display artifacts
            artifact_dir = os.path.join("evaluations", selected_model)
            report_path = os.path.join(artifact_dir, f"classification_report_{selected_tool}.txt")
            cm_path = os.path.join(artifact_dir, f"confusion_matrix_{selected_tool}.png")

            if os.path.exists(report_path):
                st.subheader("📄 Classification Report")
                with open(report_path, "r") as f:
                    report_text = f.read()
                st.text(report_text)
            else:
                st.warning("Classification repor t not found.")

            if os.path.exists(cm_path):
                st.subheader("📊 Confusion Matrix")
                st.image(cm_path, use_column_width=True)
            else:
                st.warning("Confusion matrix image not found.")

        except subprocess.CalledProcessError as e:
            st.error("❌ Evaluation failed:")
            st.code(e.stderr)

# Footer Instructions
st.markdown("""
---
### 📌 Instructions:
- Make sure your Python environment includes all required packages (`pip install -r requirements.txt`).
- Provide a full path to a `.pt` model checkpoint file.
- Model, tool, and sensor selections should match what was used during training.
""")
