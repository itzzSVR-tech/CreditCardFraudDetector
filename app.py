import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Fraud Detector", layout="wide")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('model/fraud_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

st.title("💳 Credit Card Fraud Detection App")
st.markdown("Use this app to detect fraud in individual or multiple credit card transactions.")

tabs = st.tabs(["🔍 Single Transaction", "📂 Batch Prediction", "📊 Fraud Analytics"])

# Tab 1: Single Prediction
with tabs[0]:
    st.subheader("🔍 Predict a Single Transaction")
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
    input_data = []

    cols = st.columns(3)
    for idx, feature in enumerate(feature_names):
        with cols[idx % 3]:
            value = st.number_input(f"{feature}", value=0.0, format="%.4f")
            input_data.append(value)

    if st.button("Predict"):
        transaction = np.array(input_data).reshape(1, -1)
        transaction[:, -1] = scaler.transform(transaction[:, -1].reshape(-1, 1))
        prediction = model.predict(transaction)[0]
        prob = model.predict_proba(transaction)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ Legitimate Transaction. (Confidence: {1 - prob:.2%})")

# Tab 2: Batch Prediction
with tabs[1]:
    st.subheader("📂 Batch Upload for Multiple Predictions")
    uploaded_file = st.file_uploader("Upload a CSV file with transactions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("🔎 Preview of Uploaded Data:")
        st.dataframe(df.head())

        if 'Amount' not in df.columns or 'Time' not in df.columns:
            st.warning("CSV must contain all required features including 'Time' and 'Amount'.")
        else:
            df_scaled = df.copy()
            df_scaled['Amount'] = scaler.transform(df_scaled['Amount'].values.reshape(-1, 1))

            if 'Class' in df_scaled.columns:
                        df_scaled = df_scaled.drop(columns=['Class'])

            predictions = model.predict(df_scaled)
            probs = model.predict_proba(df_scaled)[:, 1]

            df['Prediction'] = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]
            df['Confidence'] = probs

            st.success("✅ Predictions Complete")
            st.dataframe(df)

            # Download button
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Predictions", csv_buffer.getvalue(), file_name="predictions.csv", mime="text/csv")

# Tab 3: Visualization
with tabs[2]:
    st.subheader("📊 Fraud Insights & Charts")
    demo_data = pd.read_csv("data/creditcard.csv")  # Real data required

    fig1, ax1 = plt.subplots()
    labels = ['Not Fraud', 'Fraud']
    sizes = demo_data['Class'].value_counts()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#16c784", "#ff4b4b"])
    ax1.axis('equal')
    st.pyplot(fig1)

    st.markdown("### 🔍 Feature Distributions (Fraud vs Non-Fraud)")
    col = st.selectbox("Choose a feature to view distribution", demo_data.columns[:-1])
    fig2, ax2 = plt.subplots()
    sns.histplot(data=demo_data, x=col, hue="Class", bins=50, ax=ax2, palette={0: "#16c784", 1: "#ff4b4b"}, element="step", stat="density")
    st.pyplot(fig2)
