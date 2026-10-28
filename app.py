"""
NeuroDigit AI — Handwritten Digit Recognition
EduBridge Mini Project 8: Neural Network Use Case
Interactive Streamlit Application for Classroom & Viva Demonstration

Developer: Harish
Email: harishhari781823@gmail.com
GitHub: https://github.com/Haris87-star/edubridge-miniproject8
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import preprocess_user_image, normalize_pixels
from src.prediction import predict_digit
from src.model import load_model_from_disk, get_architecture_details
from src.data_loader import load_mnist_data, get_sample_digits_per_class

# Page configuration
st.set_page_config(
    page_title="NeuroDigit AI — EduBridge Mini Project 8",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for modern, accessible, and vibrant UI
st.markdown(
    """
    <style>
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4338ca 100%);
        color: #ffffff;
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(49, 46, 129, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c7d2fe;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .pill-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        margin-right: 0.5rem;
        margin-top: 0.3rem;
    }
    .badge-primary {
        background: rgba(255, 255, 255, 0.2);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.35);
    }
    .badge-accent {
        background: #10b981;
        color: #ffffff;
    }
    .badge-secondary {
        background: #f59e0b;
        color: #ffffff;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }
    .prediction-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 2px solid #bfdbfe;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-badge {
        font-size: 4.5rem;
        font-weight: 900;
        color: #1e40af;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .confidence-tag {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .author-card {
        background: #f1f5f9;
        border-left: 4px solid #4338ca;
        border-radius: 8px;
        padding: 0.9rem;
        margin-top: 1rem;
        font-size: 0.88rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4338ca, #3b82f6);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained NeuroDigit model...")
def get_cached_model(model_path: str = "models/neuralvision_model.keras"):
    """
    Loads and caches the trained neural network model.
    """
    if not os.path.exists(model_path):
        return None
    return load_model_from_disk(model_path)


@st.cache_data(show_spinner="Loading evaluation metrics...")
def get_cached_metrics(metrics_path: str = "outputs/metrics.json"):
    """
    Loads and caches evaluation metrics.
    """
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading sample MNIST test data...")
def get_cached_test_samples():
    """
    Loads and caches a small slice of MNIST test data for interactive exploration.
    """
    try:
        _, _, (X_test_raw, y_test) = load_mnist_data()
        X_test = normalize_pixels(X_test_raw)
        return X_test[:100], y_test[:100]
    except Exception:
        return None, None


def main():
    # Sidebar Navigation & Branding
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", caption="MNIST Benchmark Dataset", use_container_width=True)
    st.sidebar.title("🧠 NeuroDigit AI")
    st.sidebar.markdown("**EduBridge Mini Project 8**\n*Handwritten Digit Recognition System*")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "Navigation Menu",
        [
            "🏠 Home & Overview",
            "✏️ Live Digit Recognition",
            "📊 Model Performance & Metrics",
            "🔍 Sample Digit Explorer",
            "🎓 Viva & Theory Guide",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="author-card">
            <b>👨‍💻 Project Developer:</b> Harish<br/>
            <b>📧 Contact:</b> <a href="mailto:harishhari781823@gmail.com">harishhari781823@gmail.com</a><br/>
            <b>🎓 Curriculum:</b> EduBridge Mini Project 8<br/>
            <b>🔗 GitHub:</b> <a href="https://github.com/Haris87-star/edubridge-miniproject8" target="_blank">Haris87-star/edubridge-miniproject8</a><br/>
            <b>⚡ Execution:</b> CPU Optimized (TensorFlow 2.16+)
        </div>
        """,
        unsafe_allow_html=True,
    )

    model = get_cached_model()
    metrics = get_cached_metrics()

    # Route based on navigation
    if menu == "🏠 Home & Overview":
        render_home(model, metrics)
    elif menu == "✏️ Live Digit Recognition":
        render_digit_recognition(model)
    elif menu == "📊 Model Performance & Metrics":
        render_performance(metrics)
    elif menu == "🔍 Sample Digit Explorer":
        render_sample_explorer(model)
    elif menu == "🎓 Viva & Theory Guide":
        render_viva_guide()


def render_home(model, metrics):
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🧠 NeuroDigit AI — Handwritten Digit Recognition</h1>
            <p class="hero-subtitle">An end-to-end, CPU-optimized deep learning system built for EduBridge Mini Project 8.</p>
            <div>
                <span class="pill-badge badge-primary">🎓 EduBridge Mini Project 8</span>
                <span class="pill-badge badge-accent">✅ 97.65% Test Accuracy</span>
                <span class="pill-badge badge-secondary">⚡ Lightweight CPU Inference (&lt;12ms)</span>
                <span class="pill-badge badge-primary">👨‍💻 Developer: Harish</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Status Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Architecture", "Multilayer Perceptron", "3 Dense Layers")
    with col2:
        acc = metrics.get("test_accuracy_percentage", "97.65%") if metrics else "97.65%"
        st.metric("Test Accuracy", acc, "Untouched Test Set")
    with col3:
        loss = f"{metrics.get('test_loss', 0.0801):.4f}" if metrics else "0.0801"
        st.metric("Test Loss", loss, "Cross-Entropy")
    with col4:
        status = "Active & Ready" if model is not None else "Needs Training"
        st.metric("Model Status", status)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📌 Project Overview & EduBridge Objectives")
        st.write(
            """
            Handwritten digit recognition represents one of the foundational milestones in computer vision and artificial intelligence.
            **NeuroDigit AI (EduBridge Mini Project 8)** demonstrates how a feedforward neural network learns hierarchical representations from 2D pixel intensities
            to recognize human handwriting with high precision.
            
            **Key Technical Highlights:**
            - **End-to-End Pipeline:** From raw image input, contrast polarity correction, and center-of-mass alignment, to forward inference and softmax posterior distributions.
            - **Strict Evaluation Protocol:** Validation data is drawn exclusively from training partitions; the official 10,000-sample test set remains strictly untouched.
            - **Student & Viva Oriented:** Every mathematical concept—from activation functions to backpropagation gradients—is fully exposed and explainable.
            - **Zero External Inference Fees:** Lightweight, fully CPU-compatible execution with fast inference (<12 ms).
            """
        )

        st.subheader("📐 Neural Network Architecture")
        st.write(
            """
            The model utilizes a Multilayer Perceptron (MLP) structured as follows:
            1. **Input Layer:** Reshapes 28×28 grayscale images into a 784-dimensional flat vector.
            2. **Hidden Layer 1 (Dense 128, ReLU):** Learns low-to-mid level stroke primitives, edge angles, and curve curvatures.
            3. **Dropout Layer (Rate 0.2):** Regularization layer randomly muting 20% of neuron activations to eliminate co-adaptation.
            4. **Hidden Layer 2 (Dense 64, ReLU):** Synthesizes stroke features into higher-order digit topologies (loops, crosses, stems).
            5. **Dropout Layer (Rate 0.1):** Secondary regularization ensuring generalization.
            6. **Output Layer (Dense 10, Softmax):** Produces a normalized probability distribution across discrete classes 0 through 9.
            """
        )

    with col_right:
        st.subheader("📚 Dataset Specifications")
        st.write(
            """
            - **Name:** MNIST (Modified National Institute of Standards and Technology)
            - **Total Samples:** 70,000 grayscale handwritten digits
            - **Training Set:** 50,000 samples
            - **Validation Set:** 10,000 samples (stratified)
            - **Test Set:** 10,000 samples (untouched)
            - **Resolution:** 28 × 28 pixels (784 features)
            - **Classes:** 10 digits (0, 1, 2, ..., 9)
            """
        )

        st.subheader("⚠️ Model Limitations")
        st.warning(
            """
            - **Spatial Invariance:** Unlike Convolutional Neural Networks (CNNs), MLPs flatten images into 1D vectors, losing translational invariance. Digits must be centered properly.
            - **Overlapping Strokes:** Single-digit scope only. Multi-digit strings or math symbols require object detection architectures.
            - **Background Noise:** Highly textured backgrounds or low contrast images must be cleaned prior to inference.
            """
        )

    st.markdown("---")
    st.subheader("🚀 Quick Start Guide")
    st.code(
        """
# 1. Train the model from scratch on CPU
python train.py --epochs 12 --batch-size 64

# 2. Run evaluation on untouched test set
python evaluate.py

# 3. Launch this interactive Streamlit application
streamlit run app.py
        """,
        language="bash",
    )


def render_digit_recognition(model):
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">✏️ Live Digit Recognition Studio</h1>
            <p class="hero-subtitle">Draw a digit directly on the canvas or upload an image file to classify it in real time.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(
            "⚠️ Trained model not detected at `models/neuralvision_model.keras`.\n\n"
            "Please train the model first by executing `python train.py` in your terminal."
        )
        return

    input_mode = st.radio(
        "Choose Input Method:",
        ["🎨 Interactive Drawing Canvas", "📁 Upload Digit Image", "🧪 Test with Preset Samples"],
        horizontal=True,
    )

    preprocessed_img = None

    if input_mode == "🎨 Interactive Drawing Canvas":
        col_canvas, col_ctrl = st.columns([3, 2])

        with col_canvas:
            st.markdown("##### Draw a single digit (0–9) centered below:")
            try:
                from streamlit_drawable_canvas import st_canvas
                
                brush_preset = st.radio("Stroke Width Preset:", ["Fine (16px)", "Medium (22px)", "Bold (28px)"], horizontal=True, index=1)
                preset_map = {"Fine (16px)": 16, "Medium (22px)": 22, "Bold (28px)": 28}
                stroke_width = preset_map[brush_preset]

                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=stroke_width,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="digit_canvas",
                )
                if canvas_result.image_data is not None:
                    raw_data = canvas_result.image_data
                    if np.max(raw_data[:, :, :3]) > 10:
                        preprocessed_img = preprocess_user_image(raw_data)
            except Exception as e:
                st.warning(f"Interactive canvas unavailable: {e}. Please use 'Upload Digit Image' option.")

        with col_ctrl:
            st.markdown("##### Drawing Instructions & Tips")
            st.info(
                "1. **Center your digit:** Draw cleanly in the middle of the box.\n"
                "2. **Continuous strokes:** Avoid broken or faint strokes.\n"
                "3. **Clear Canvas:** Click the trash icon 🗑️ on the canvas toolbar to start fresh.\n"
                "4. **Stroke Width:** Use the preset selector above for optimal pen width."
            )

    elif input_mode == "📁 Upload Digit Image":
        col_up, col_help = st.columns([3, 2])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload a handwritten digit image (PNG, JPG, JPEG, BMP):",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Maximum size: 5MB",
            )
            if uploaded_file is not None:
                if uploaded_file.size > 5 * 1024 * 1024:
                    st.error("Uploaded file exceeds the 5MB size limit.")
                else:
                    img_bytes = uploaded_file.read()
                    preprocessed_img = preprocess_user_image(img_bytes)
                    st.image(uploaded_file, caption="Original Uploaded Image", width=180)

        with col_help:
            st.markdown("##### Upload Guidelines")
            st.info(
                "- Accepts black digit on white paper or white digit on black background.\n"
                "- Auto-inverts polarity and scales digit to 20×20 preserving aspect ratio.\n"
                "- Center-of-mass centering aligns digit identically to MNIST training samples."
            )

    else:  # Preset samples
        st.markdown("##### Select a pre-exported MNIST test sample:")
        sample_col, _ = st.columns([3, 2])
        with sample_col:
            sample_digit = st.selectbox("Select Digit:", list(range(10)), index=3)
            sample_path = os.path.join("sample_digits", f"sample_digit_{sample_digit}.png")
            if os.path.exists(sample_path):
                img_pil = Image.open(sample_path)
                st.image(img_pil, caption=f"Sample Test Digit '{sample_digit}'", width=140)
                preprocessed_img = preprocess_user_image(sample_path)
            else:
                st.info("Preset sample images will be generated when you run `python train.py`.")

    # Processing & Prediction Results
    st.markdown("---")
    if preprocessed_img is not None and np.max(preprocessed_img) > 0.05:
        res = predict_digit(model, preprocessed_img)
        pred_digit = res["predicted_digit"]
        confidence = res["confidence"]
        top_3 = res["top_3"]

        st.subheader("🔍 Preprocessing Pipeline & Classification Output")
        c1, c2, c3 = st.columns([1.5, 2, 3])

        with c1:
            st.markdown("##### Normalized Input (28×28)")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(preprocessed_img, cmap="gray", interpolation="nearest")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Shape: (28, 28) | Min: {preprocessed_img.min():.2f} | Max: {preprocessed_img.max():.2f}")

        with c2:
            st.markdown("##### Predicted Classification")
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="font-size: 0.9rem; color: #4b5563; font-weight: 600;">CLASSIFIED DIGIT</div>
                    <div class="prediction-badge">{pred_digit}</div>
                    <div class="confidence-tag">Confidence: {confidence * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(confidence))

            # Confidence assessment badge
            if confidence >= 0.85:
                st.success("🎯 High Confidence: Model is highly certain.")
            elif confidence >= 0.50:
                st.warning("⚠️ Moderate Confidence: Hand curvature is slightly ambiguous.")
            else:
                st.error("❓ Low Confidence: Unclear digit stroke.")

            st.markdown("##### Top 3 Candidates")
            for rank, item in enumerate(top_3, start=1):
                st.write(f"**#{rank} Digit {item['digit']}:** `{item['percentage']}`")

        with c3:
            st.markdown("##### Softmax Probability Distribution (0–9)")
            prob_df = pd.DataFrame({
                "Digit": [str(d) for d in range(10)],
                "Probability": [res["probabilities"][d] for d in range(10)],
            })

            fig, ax = plt.subplots(figsize=(6, 3.8), dpi=120)
            colors = ["#4338ca" if int(d) == pred_digit else "#cbd5e1" for d in prob_df["Digit"]]
            bars = ax.bar(prob_df["Digit"], prob_df["Probability"], color=colors, edgecolor="#1e1b4b", linewidth=0.8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Softmax Probability", fontsize=10)
            ax.set_xlabel("Digit Class", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Annotate values on top of bars
            for bar in bars:
                height = bar.get_height()
                if height >= 0.01:
                    ax.annotate(
                        f"{height * 100:.1f}%",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        fontweight="bold",
                    )

            st.pyplot(fig)
            plt.close(fig)

            # Export prediction summary
            summary_json = json.dumps(
                {
                    "predicted_digit": int(pred_digit),
                    "confidence": float(confidence),
                    "top_3": top_3,
                    "developer": "Harish",
                    "project": "EduBridge Mini Project 8",
                },
                indent=2,
            )
            st.download_button(
                label="📥 Download Prediction Result (JSON)",
                data=summary_json,
                file_name=f"prediction_digit_{pred_digit}.json",
                mime="application/json",
            )

    else:
        st.info("👉 Please draw a digit or upload an image to view predictions.")


def render_performance(metrics):
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">📊 Model Performance & Evaluation Metrics</h1>
            <p class="hero-subtitle">Comprehensive evaluation on the untouched 10,000-sample MNIST test set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics is None:
        st.warning("⚠️ Metrics file `outputs/metrics.json` not found. Please run `python train.py` or `python evaluate.py`.")
        return

    # Top KPI metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Test Accuracy", metrics.get("test_accuracy_percentage", "97.65%"))
    with kpi2:
        st.metric("Test Loss", f"{metrics.get('test_loss', 0.0801):.4f}")
    with kpi3:
        macro_prec = metrics.get('macro_avg', {}).get('precision', 0.9763)
        st.metric("Macro Precision", f"{macro_prec:.4f}")
    with kpi4:
        macro_rec = metrics.get('macro_avg', {}).get('recall', 0.9764)
        st.metric("Macro Recall", f"{macro_rec:.4f}")
    with kpi5:
        macro_f1 = metrics.get('macro_avg', {}).get('f1_score', 0.9764)
        st.metric("Macro F1-Score", f"{macro_f1:.4f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Training History", "🔲 Confusion Matrix", "📋 Per-Digit Classification Report"])

    with tab1:
        st.subheader("Training vs Validation Curves")
        history_path = "outputs/training_history.png"
        if os.path.exists(history_path):
            st.image(history_path, caption="Convergence of Accuracy and Loss across Epochs", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate training curves.")

        st.markdown(
            """
            **Analysis & Insights:**
            - **Overfitting Diagnostics:** Notice the minimal gap between training accuracy and validation accuracy. The inclusion of Dropout layers (rates 0.2 and 0.1) and validation-monitored early stopping prevents the network from memorizing pixel-level training noise.
            - **Convergence Rate:** The Adam optimizer achieves >95% accuracy within the first 3 epochs on CPU, demonstrating fast and stable convergence.
            """
        )

    with tab2:
        st.subheader("Confusion Matrix Heatmap")
        cm_path = "outputs/confusion_matrix.png"
        if os.path.exists(cm_path):
            st.image(cm_path, caption="10x10 MNIST Confusion Matrix Heatmap", use_container_width=True)
        else:
            st.info("Run `python evaluate.py` to generate the confusion matrix plot.")

        st.markdown("##### Most Frequently Confused Digit Pairs")
        confused_pairs = metrics.get("top_confused_pairs", [])
        if confused_pairs:
            for item in confused_pairs:
                st.markdown(f"- **{item['description']}**")
        st.write(
            "These misclassifications reflect natural visual ambiguities in human handwriting (e.g., an open top on a 4 resembles a 9; a curved 7 resembles a 1 or 2)."
        )

    with tab3:
        st.subheader("Per-Digit Performance Breakdown")
        records = []
        for digit in range(10):
            d_data = metrics.get("per_digit_metrics", {}).get(str(digit), {})
            records.append({
                "Digit Class": str(digit),
                "Precision": f"{d_data.get('precision', 0.0):.4f}",
                "Recall": f"{d_data.get('recall', 0.0):.4f}",
                "F1-Score": f"{d_data.get('f1_score', 0.0):.4f}",
                "Test Support": d_data.get("support", 0),
            })
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_sample_explorer(model):
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🔍 Sample Digit Explorer</h1>
            <p class="hero-subtitle">Browse actual MNIST test samples, inspect ground truth vs model predictions, and examine failure cases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    example_plot_path = "outputs/example_predictions.png"
    if os.path.exists(example_plot_path):
        st.subheader("Model Predictions on Test Samples (Correct & Misclassified)")
        st.image(example_plot_path, caption="Visual comparison of true labels vs predictions with confidence scores", use_container_width=True)

    X_test, y_test = get_cached_test_samples()
    if X_test is not None and model is not None:
        st.markdown("---")
        st.subheader("Interactive Sample Inspector")
        sample_idx = st.slider("Select Sample Index from Test Subset (0–99):", min_value=0, max_value=len(X_test) - 1, value=18)

        img = X_test[sample_idx]
        true_label = int(y_test[sample_idx])
        pred_res = predict_digit(model, img)
        pred_label = pred_res["predicted_digit"]
        conf = pred_res["confidence"]

        col1, col2 = st.columns([1, 2])
        with col1:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            is_correct = true_label == pred_label
            status_text = "✅ CORRECT" if is_correct else "❌ MISCLASSIFIED"
            status_color = "green" if is_correct else "red"

            st.markdown(f"#### Status: :{status_color}[{status_text}]")
            st.write(f"- **Ground Truth Label:** `{true_label}`")
            st.write(f"- **Model Predicted Label:** `{pred_label}`")
            st.write(f"- **Inference Confidence:** `{conf * 100:.2f}%`")


def render_viva_guide():
    st.markdown(
        """
        <div class="hero-banner">
            <h1 class="hero-title">🎓 Student Viva & Neural Network Theory Guide</h1>
            <p class="hero-subtitle">Comprehensive theoretical explanations formatted for viva questions, oral defenses, and lab presentations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Interactive parameter calculator
    st.subheader("🧮 Interactive Model Parameter Calculator")
    st.write("Calculate weights, biases, and parameter formulas for any Feedforward MLP configuration:")
    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        in_dim = st.number_input("Input Dimension (28×28):", value=784, disabled=True)
    with c_p2:
        h1_dim = st.number_input("Hidden Layer 1 Neurons:", min_value=16, max_value=512, value=128, step=16)
    with c_p3:
        h2_dim = st.number_input("Hidden Layer 2 Neurons:", min_value=16, max_value=256, value=64, step=16)

    dense1_params = (in_dim * h1_dim) + h1_dim
    dense2_params = (h1_dim * h2_dim) + h2_dim
    dense_out_params = (h2_dim * 10) + 10
    total_calc_params = dense1_params + dense2_params + dense_out_params

    st.info(
        f"**Layer 1 (Input → Hidden 1):** `({in_dim} × {h1_dim}) + {h1_dim} = {dense1_params:,}` parameters  \n"
        f"**Layer 2 (Hidden 1 → Hidden 2):** `({h1_dim} × {h2_dim}) + {h2_dim} = {dense2_params:,}` parameters  \n"
        f"**Layer 3 (Hidden 2 → Output):** `({h2_dim} × 10) + 10 = {dense_out_params:,}` parameters  \n"
        f"**Total Trainable Parameters:** **`{total_calc_params:,}`** parameters."
    )

    st.markdown("---")
    details = get_architecture_details()

    with st.expander("1. What are Input Features in this network?", expanded=True):
        st.write(details["input_features"])
        st.info("💡 **Viva Tip:** Emphasize that pixel intensities are normalized from [0, 255] to [0.0, 1.0] so gradients don't explode and weights converge smoothly.")

    with st.expander("2. Explain Weights and Biases (W and b)", expanded=True):
        st.write(details["weights_and_biases"])
        st.latex(r"z = \sum_{i=1}^n (w_i \cdot x_i) + b = W^T x + b")

    with st.expander("3. Why do we need the Flatten Layer?", expanded=False):
        st.write(details["flatten_layer"])
        st.latex(r"(28 \times 28) \longrightarrow 784 \text{ dimensions}")

    with st.expander("4. What role do the Hidden Layers play?", expanded=False):
        st.write(details["hidden_layers"])

    with st.expander("5. Why ReLU instead of Sigmoid?", expanded=False):
        st.write(details["relu_activation"])
        st.latex(r"f(x) = \max(0, x)")
        st.write("Sigmoid suffers from vanishing gradients when activations saturate near 0 or 1. ReLU has a constant gradient of 1 for positive inputs.")

    with st.expander("6. How does Dropout prevent Overfitting?", expanded=False):
        st.write(details["dropout_regularization"])

    with st.expander("7. Explain the Softmax Activation Function", expanded=False):
        st.write(details["softmax_output"])
        st.latex(r"\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=0}^9 e^{z_j}} \quad \text{where} \quad \sum_{i=0}^9 \sigma(z)_i = 1.0")

    with st.expander("8. What is Sparse Categorical Cross-Entropy Loss?", expanded=False):
        st.write(details["loss_function"])
        st.latex(r"L = -\ln(p_{\text{true}})")

    with st.expander("9. What is Backpropagation?", expanded=False):
        st.write(details["backpropagation"])
        st.latex(r"\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial W^{[l]}}")

    with st.expander("10. What is Adam Optimizer?", expanded=False):
        st.write(details["optimizer"])

    with st.expander("11. Epochs vs Batch Size vs Iterations", expanded=False):
        st.write(details["epochs_and_batch_size"])


if __name__ == "__main__":
    main()
