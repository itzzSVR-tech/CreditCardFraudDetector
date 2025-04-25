# 💳 Credit Card Fraud Detection App

A Streamlit web app that uses an AI model to detect fraudulent credit card transactions. You can test a single transaction manually or upload a CSV file for batch fraud detection. The app also includes data visualizations to help understand transaction trends.

---

## 🚀 Features

- 🔍 **Single Transaction Prediction**  
  Input features like V1–V28, Amount, and Time to instantly predict if it's a fraud or not.

- 📂 **Batch Prediction**  
  Upload a CSV of transactions to analyze many at once.

- 📊 **Fraud Analytics Dashboard**  
  Visualize the distribution of fraud vs non-fraud and explore feature trends.

- 📥 **Download Results**  
  Export batch prediction results (with confidence scores) as a CSV file.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: Python (scikit-learn, pandas, numpy)
- **Model**: Logistic Regression / Custom Trained Model
- **Visualization**: Matplotlib, Seaborn

---

## 🗂️ Folder Structure

```kotlin
fraud_detector_app/
├── app.py                     ← Streamlit app
├── model/
│   ├── fraud_model.pkl
│   └── scaler.pkl
├── data/
│   └── creditcard.csv         ← (Optional: used for visualizations)
├── requirements.txt
├── README.md
```

---

## 📦 Installation

### 🔧 Local Setup

```bash
# Clone the repo
git clone https://github.com/itzzSVR-tech/CreditCardFraudDetector.git
cd CreditCardFraudDetector

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
