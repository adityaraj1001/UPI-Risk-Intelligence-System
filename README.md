# 🚨 UPI Risk Intelligence System 💳

An AI-powered fraud detection system designed to analyze UPI transactions and identify suspicious activities using machine learning, real-time analytics, and interactive visualizations.

This project simulates how banks and financial institutions monitor digital payment fraud and assess transaction risk intelligently.

---

## 🔍 Project Overview

With the rapid growth of UPI transactions in India, fraud detection has become critical.  
The **UPI Risk Intelligence System** uses machine learning and behavioral analysis to detect potentially fraudulent transactions and assist fraud analysts in decision-making.

The system evaluates transaction patterns such as amount, time, device, bank risk, velocity, and user behavior to generate a fraud risk score.

---

## ✨ Key Features

- ✅ AI-based fraud prediction using Machine Learning  
- 📊 Real-time transaction risk scoring  
- ⚡ Advanced feature engineering (velocity, night-time activity, risk zones)  
- 🖥️ Interactive Streamlit dashboard with KPIs  
- 🛡️ Security scan and fraud analysis module  
- 🕸️ Fraud network visualization using graphs  
- 🔐 Secure login system for analysts  

---

## 🧠 Machine Learning Details

- **Algorithm Used:** Random Forest Classifier  
- **Data Processing:**  
  - Feature scaling and encoding  
  - Time-based and behavioral feature engineering  
- **Prediction Output:**  
  - Fraud probability score  
  - Risk level indication  

The model is trained to handle imbalanced fraud data using optimized parameters.

---

## 🛠️ Tech Stack

| Category | Technologies |
|--------|-------------|
| Programming | Python |
| Frontend | Streamlit |
| ML & Data | Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, NetworkX |
| AI Integration | Gemini API |
| Tools | Git, GitHub |

---

## ▶️ How to Run the Project

1. Clone the repository  
2. Install required dependencies  
3. Run the Streamlit application  

```bash
pip install -r requirements.txt
streamlit run streamlit.py
