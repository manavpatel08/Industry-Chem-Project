# ============================================================
#  app.py  –  Iris Species Predictor  |  Streamlit Dashboard
#  Run:  streamlit run app.py
#  (Make sure iris_model.pkl exists — run train_model.py first)
# ============================================================

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Predictor",
    page_icon="🌸",
    layout="wide",
)

# ── Load model bundle ────────────────────────────────────────
@st.cache_resource
def load_bundle():
    with open("iris_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("❌ iris_model.pkl not found. Run `python train_model.py` first.")
    st.stop()

model         = bundle["model"]
scaler        = bundle["scaler"]
feature_names = bundle["feature_names"]
target_names  = bundle["target_names"]
X_train       = bundle["X_train"]
X_test        = bundle["X_test"]
y_train       = bundle["y_train"]
y_test        = bundle["y_test"]
accuracy      = bundle["accuracy"]

COLORS = ["#3B6D11", "#185FA5", "#BA7517"]   # green, blue, amber

# ════════════════════════════════════════════════════════════
#  SIDEBAR  –  Input sliders
# ════════════════════════════════════════════════════════════
st.sidebar.title("🌸 Iris Predictor")
st.sidebar.markdown("Adjust the flower measurements below:")

sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sw = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
pw = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 1.3, 0.1)

input_arr    = np.array([[sl, sw, pl, pw]])
input_scaled = scaler.transform(input_arr)
pred_class   = model.predict(input_scaled)[0]
pred_proba   = model.predict_proba(input_scaled)[0]
pred_name    = target_names[pred_class]

st.sidebar.markdown("---")
st.sidebar.markdown(f"### Prediction: **{pred_name.capitalize()}**")
st.sidebar.markdown(f"Confidence: **{pred_proba[pred_class]*100:.1f}%**")

# ════════════════════════════════════════════════════════════
#  MAIN  –  Title + KPI row
# ════════════════════════════════════════════════════════════
st.title("🌸 Iris Flower Species Predictor")
st.caption("Fisher's Iris Dataset · Logistic Regression · Sklearn")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset Size",   "150 samples")
col2.metric("Features",       "4")
col3.metric("Classes",        "3 species")
col4.metric("Test Accuracy",  f"{accuracy*100:.1f}%")

st.markdown("---")

# ════════════════════════════════════════════════════════════
#  ROW 1  –  Prediction result  +  Probability bar chart
# ════════════════════════════════════════════════════════════
r1_left, r1_right = st.columns([1, 1])

with r1_left:
    st.subheader("Prediction Result")
    badge_color = COLORS[pred_class]
    st.markdown(
        f"""
        <div style="border:1px solid #ddd; border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:48px;">🌸</div>
            <div style="font-size:28px; font-weight:600; color:{badge_color}; margin:8px 0;">
                Iris {pred_name}
            </div>
            <div style="font-size:16px; color:#666;">
                Confidence: <strong>{pred_proba[pred_class]*100:.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Your inputs:**")
    input_df = pd.DataFrame(
        [input_arr[0]],
        columns=["Sepal L", "Sepal W", "Petal L", "Petal W"]
    )
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with r1_right:
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(
        [f"Iris {n}" for n in target_names],
        pred_proba * 100,
        color=COLORS,
        height=0.5,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    for bar, val in zip(bars, pred_proba * 100):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=11)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ════════════════════════════════════════════════════════════
#  ROW 2  –  Confusion Matrix  +  PCA scatter plot
# ════════════════════════════════════════════════════════════
r2_left, r2_right = st.columns([1, 1])

with r2_left:
    st.subheader("Confusion Matrix (Test Set)")
    y_pred = model.predict(scaler.transform(X_test))
    cm     = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=[n.capitalize() for n in target_names])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Predicted vs Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with r2_right:
    st.subheader("PCA Scatter (2D View of Dataset)")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    pca    = PCA(n_components=2)
    X_pca  = pca.fit_transform(scaler.fit_transform(X_all))
    inp_pca = pca.transform(scaler.transform(input_arr))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, name in enumerate(target_names):
        mask = y_all == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=COLORS[i], label=f"Iris {name}", alpha=0.65, s=40, edgecolors='none')

    ax.scatter(inp_pca[0, 0], inp_pca[0, 1],
               c="red", s=180, marker="*", zorder=5, label="Your input")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ════════════════════════════════════════════════════════════
#  ROW 3  –  Feature distributions  +  Dataset preview
# ════════════════════════════════════════════════════════════
r3_left, r3_right = st.columns([1, 1])

with r3_left:
    st.subheader("Feature Distributions by Species")
    feature_labels = ["Sepal L", "Sepal W", "Petal L", "Petal W"]
    X_all  = np.vstack([X_train, X_test])
    y_all  = np.concatenate([y_train, y_test])
    df_all = pd.DataFrame(X_all, columns=feature_labels)
    df_all["Species"] = [target_names[i].capitalize() for i in y_all]

    selected_feat = st.selectbox("Choose feature", feature_labels)
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, name in enumerate(target_names):
        vals = df_all[df_all["Species"] == name.capitalize()][selected_feat]
        ax.hist(vals, bins=12, alpha=0.65, color=COLORS[i],
                label=f"Iris {name}", edgecolor='white', linewidth=0.5)
    ax.axvline(input_arr[0][feature_labels.index(selected_feat)],
               color="red", linestyle="--", linewidth=2, label="Your value")
    ax.set_xlabel(selected_feat)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with r3_right:
    st.subheader("Dataset Preview")
    df_preview = pd.DataFrame(X_all, columns=feature_labels)
    df_preview["Species"] = [f"Iris {target_names[i]}" for i in y_all]
    st.dataframe(df_preview.head(20), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
#  Footer
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.caption(
    "Dataset: Fisher's Iris (1936) · Model: Logistic Regression (sklearn) · "
    "Assignment: Regression on Biological Dataset with Web App"
)