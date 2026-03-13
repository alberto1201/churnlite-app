"""
app.py — ChurnLite Streamlit demo app.

This app lets users:
  1. Upload a CSV file with customer data
  2. Run churn predictions using the churnlite library
  3. View the results in a table
  4. Download the predictions as a CSV
"""

import pandas as pd
import streamlit as st
from churnlite import ChurnPipeline

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnLite",
    page_icon="📉",
    layout="wide",
)

# ── Title and description ─────────────────────────────────────────────────────
st.title("📉 ChurnLite — Customer Churn Predictor")
st.markdown(
    """
    Upload a CSV file with customer data to predict who is likely to churn.
    
    **Required columns:** `id`, `longdist`, `internat`, `local`, `int_disc`, 
    `billtype`, `age`, `gender`, `marital`, `children`, `est_inco`, `car`, 
    `pay_Bank`, `pay_CreditCard`, `pay_Cash`  
    
    **Optional column:** `churn` (0/1) — if present, the model will train on 80% 
    and predict on the remaining 20%, showing accuracy metrics.
    """
)

st.divider()

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    # Load the data
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Data preview")
    st.write(f"Rows: **{len(df)}** | Columns: **{len(df.columns)}**")
    st.dataframe(df.head(10), use_container_width=True)

    # Button to run the pipeline
    if st.button("🚀 Run Churn Prediction", type="primary"):

        with st.spinner("Training model and generating predictions..."):
            pipe = ChurnPipeline()
            result = pipe.run(df, id_col="id", target_col="churn")
            preds = result["predictions"]
            metrics = result.get("metrics", {})

        st.success("✅ Done!")

        # ── Metrics ───────────────────────────────────────────────────────────
        if metrics:
            st.subheader("📊 Model performance (test set)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="ROC-AUC Score",
                    value=f"{metrics['roc_auc']:.4f}",
                    help="1.0 = perfect model. 0.5 = random guessing."
                )
            with col2:
                n_churners = int(preds["prediction"].sum())
                st.metric(
                    label="Predicted churners (test set)",
                    value=n_churners,
                )

        # ── Predictions table ─────────────────────────────────────────────────
        st.subheader("📄 Predictions")
        st.dataframe(
            preds.style.background_gradient(
                subset=["churn_probability"], cmap="RdYlGn_r"
            ),
            use_container_width=True,
        )

        # ── Download button ───────────────────────────────────────────────────
        csv_bytes = preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download predictions CSV",
            data=csv_bytes,
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a CSV file to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with [ChurnLite](https://github.com) · Powered by scikit-learn & Streamlit")
