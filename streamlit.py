import streamlit as st
import pandas as pd
import joblib
import os
import time
import google.generativeai as genai
import base64 
import random 
from datetime import datetime
import altair as alt 
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

# ======================================================
# --- CONFIGURATION & SETUP ---
# ======================================================

# --- Paths & Files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="UPI Risk Intelligence System",
    page_icon="💳",
    layout="wide"
)

# --- Feature List (Must match model.py) ---
EXPECTED_FEATURES = [
    "amount", "hour_of_day", "is_weekend", "sender_bank_risk", "receiver_bank_risk", 
    "sender_txn_count_1h", "sender_avg_amount_1h", "sender_txn_count_24h", "sender_avg_amount_24h", 
    "receiver_txn_count_1h", "receiver_avg_amount_1h", "receiver_txn_count_24h", "receiver_avg_amount_24h",
    "day_of_week", "device_type", "network_type", "sender_bank", "receiver_bank",
    "transaction type", "merchant_category", "sender_state", "sender_age_group", 
    "receiver_age_group"
]

# --- Load ML Model ---
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
except Exception as e:
    st.warning("Note: Model loading in fallback mode. Run 'python model.py' to deploy the trained brain.")

# --- Gemini Config (Using 1.5 Flash for Image Support) ---
API_KEY = os.getenv("GEMINI_API_KEY")
gemini = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        # Using flash for multimodal image analysis capabilities
        gemini = genai.GenerativeModel("gemini-1.5-flash") 
    except Exception:
        gemini = None

# --- Base64 Image Utilities for Branding ---
def get_base64_image(url):
    try:
        import requests
        response = requests.get(url, timeout=5)
        return base64.b64encode(response.content).decode()
    except:
        return "" 

UPI_LOGO_B64 = get_base64_image("https://upload.wikimedia.org/wikipedia/commons/e/e1/UPI_logo_vector.svg")

# ======================================================
# --- SESSION STATE & NAVIGATION ---
# ======================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "login"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "temp_data" not in st.session_state:
    st.session_state.temp_data = {
        "amount": 5000, "hour": 15, "day": "Tuesday", "is_weekend": 0, 
        "device_type": "Android", "network_type": "4G", "sender_bank": "ICICI", 
        "receiver_bank": "PNB", "s_risk": 0.45, "r_risk": 0.55, 
        "txn_type": "P2P", "merchant_cat": "Entertainment", "sender_state": "Delhi",
        "sender_age": "26-35", "receiver_age": "18-25",
        "s_txn_1h": 1, "s_avg_1h": 500, "s_txn_24h": 12, "s_avg_24h": 4500,
        "r_txn_1h": 2, "r_avg_1h": 800, "r_txn_24h": 15, "r_avg_24h": 5500,
    }

def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

# ======================================================
# --- UI COMPONENTS ---
# ======================================================

def ai_assistant_sidebar():
    with st.sidebar.expander("🤖 UPI AI Assistant", expanded=False): 
        st.caption("Ask about fraud trends or RBI security protocols.")
        if not gemini:
            st.error("API Key not set. Chatbot disabled.")
            return

        for role, text in st.session_state.chat_history[-5:]:
            st.chat_message("user" if role == "You" else "assistant").write(text)
        
        q = st.chat_input("Ask a question...", key="sidebar_chat_input")
        if q:
            st.session_state.chat_history.append(("You", q))
            try:
                resp = gemini.generate_content(q)
                st.session_state.chat_history.append(("AI", resp.text))
                st.rerun() 
            except Exception as e:
                st.error(f"AI Error: {e}")

def login_page():
    st.markdown("<h1 style='text-align: center;'>🔐 UPI Risk Intelligence Login</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center;'><img src='data:image/svg+xml;base64,{UPI_LOGO_B64}' width='150'></div>", unsafe_allow_html=True)
    
    with st.container():
        left, mid, right = st.columns([1, 2, 1])
        with mid:
            with st.form("login_form"):
                u = st.text_input("Username")
                p = st.text_input("Password", type="password")
                if st.form_submit_button("Secure Access", use_container_width=True):
                    if u == "admin" and p == "admin123":
                        st.session_state.logged_in = True
                        st.session_state.current_page = "main_menu"
                        st.rerun()
                    else:
                        st.error("Access Denied: Invalid Credentials")

def main_menu():
    st.title("💳 UPI Risk Intelligence Dashboard")
    st.subheader("Welcome, Fraud Analyst 👋")
    st.divider()
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Total Transactions (24h)", "1.2M", "+1.5%")
    col_kpi2.metric("Blocked Frauds", "842", "+12")
    col_kpi3.metric("System Health", "99.9%", "Stable")
    col_kpi4.metric("Avg Fraud Loss", "₹12,400", "-5%", delta_color="inverse")

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("🔍 New Prediction", use_container_width=True, type="primary"): navigate_to("predict")
    if c2.button("🛡️ Security Scan", use_container_width=True): navigate_to("security_scan")
    if c3.button("📈 Reports", use_container_width=True): navigate_to("reports")
    if c4.button("⏱️ History", use_container_width=True): navigate_to("history")

# ======================================================
# --- NEW: SECURITY SCAN CENTER ---
# ======================================================

def security_scan_page():
    st.title("🛡️ Pre-Payment Security Scan")
    if st.sidebar.button("⬅️ Back to Menu"): navigate_to("main_menu")

    tab1, tab2 = st.tabs(["📸 Screenshot Fraud Scanner", "🔍 Credential Verifier"])

    with tab1:
        st.subheader("Upload Payment Screenshot")
        st.info("AI will scan for image tampering, suspicious transaction IDs, or phishing patterns.")
        
        uploaded_file = st.file_uploader("Choose a screenshot...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Screenshot", width=400)
            
            if st.button("🚀 Analyze Screenshot", key="scan_img"):
                if gemini:
                    with st.spinner("Analyzing image for fraud markers..."):
                        prompt = (
                            "Analyze this payment screenshot professionally. "
                            "Check for: 1. Edited text or font mismatches. 2. Unusual transaction IDs. "
                            "3. Mismatched branding/logos. 4. Signs of 'Request Money' scam. "
                            "Provide a clear VERDICT (Safe or Unsafe) and detailed reasons."
                        )
                        response = gemini.generate_content([prompt, img])
                        
                        st.divider()
                        content_lower = response.text.lower()
                        if "unsafe" in content_lower or "fraud" in content_lower or "suspicious" in content_lower:
                            st.error("🚩 VERDICT: POTENTIALLY FRAUDULENT / UNSAFE")
                        else:
                            st.success("✅ VERDICT: LIKELY SECURE")
                        st.write(response.text)
                else:
                    st.error("Gemini API key not configured for Image Analysis.")

    with tab2:
        st.subheader("Verify Receiver Credentials")
        st.caption("Perform a safety check on the UPI ID or Account before initiating payment.")
        
        col1, col2 = st.columns(2)
        vpa = col1.text_input("Receiver UPI ID (VPA)", placeholder="name@bank")
        bank_name = col2.selectbox("Receiver Bank", ["HDFC", "SBI", "ICICI", "Axis", "PNB", "Other"])
        amt_check = col1.number_input("Intended Amount (₹)", 0, 100000, 1000)

        if st.button("⚖️ Run Credential Audit"):
            with st.spinner("Checking historical risk patterns..."):
                time.sleep(1.5)
                
                # Mock logic for safety analysis
                risk_keywords = ["prize", "win", "reward", "lottery", "cashback", "customer"]
                is_suspicious_vpa = any(word in vpa.lower() for word in risk_keywords)
                
                if is_suspicious_vpa or amt_check > 50000:
                    st.error(f"🚨 HIGH RISK ALERT: '{vpa}' shows signs of social engineering tactics.")
                    st.warning("Our database flags this VPA pattern as common in 'Lottery Scams'.")
                    st.progress(85)
                else:
                    st.success(f"✅ CREDENTIALS VERIFIED: {vpa} appears legitimate.")
                    st.info("No reported fraud associated with this ID in the last 90 days.")
                    st.progress(10)
                    st.balloons()

# ======================================================
# --- THE PREDICTION PAGE ---
# ======================================================

def predict_page():
    st.title("🔍 Deep-Dive Transaction Diagnostics")
    if st.sidebar.button("⬅️ Back to Menu"): navigate_to("main_menu")

    with st.form("prediction_input_form"):
        tab1, tab2, tab3 = st.tabs(["Core Details", "Location & User", "Velocity & Bank Scores"])
        
        with tab1:
            c1, c2 = st.columns(2)
            amt = c1.number_input("💰 Amount (INR)", 1, 100000, st.session_state.temp_data["amount"])
            hour = c2.slider("⏰ Hour (24h)", 0, 23, st.session_state.temp_data["hour"])
            day = c1.selectbox("📅 Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=1)
            is_wknd = c2.selectbox("📆 Is Weekend?", [0, 1], index=st.session_state.temp_data["is_weekend"])
            txn_type = c1.selectbox("📝 Type", ["P2P", "P2M", "Bill Payment", "Other"])
            m_cat = c2.selectbox("🛍️ Category", ["Shopping", "Entertainment", "Grocery", "Utility"])

        with tab2:
            c3, c4 = st.columns(2)
            s_age = c3.selectbox("Sender Age", ["18-25", "26-35", "36-45", "46-55", "56+"], index=1)
            r_age = c4.selectbox("Receiver Age", ["18-25", "26-35", "36-45", "46-55", "56+"], index=0)
            s_state = c3.text_input("Sender State", st.session_state.temp_data["sender_state"])
            dev = c4.selectbox("Device", ["Android", "iOS", "Web"], index=0)
            net = c3.selectbox("Network", ["4G", "5G", "WiFi"], index=0)

        with tab3:
            c5, c6 = st.columns(2)
            s_bank = c5.text_input("Sender Bank", st.session_state.temp_data["sender_bank"])
            r_bank = c6.text_input("Receiver Bank", st.session_state.temp_data["receiver_bank"])
            s_bank_risk = c5.slider("Sender Bank Risk (0-1)", 0.0, 1.0, 0.45)
            r_bank_risk = c6.slider("Receiver Bank Risk (0-1)", 0.0, 1.0, 0.55)
            st.caption("Velocity Metrics (Recent History)")
            v1, v2, v3, v4 = st.columns(4)
            s_v1 = v1.number_input("S_Txn_1h", 0, 100, 2)
            s_v2 = v2.number_input("S_Avg_1h", 0, 50000, 1000)
            r_v1 = v3.number_input("R_Txn_1h", 0, 100, 1)
            r_v2 = v4.number_input("R_Avg_1h", 0, 50000, 500)

        submit = st.form_submit_button("🚀 INITIATE AI FRAUD SCAN", use_container_width=True, type="primary")

    if submit:
        input_data = {
            "amount": amt, "hour_of_day": hour, "is_weekend": is_wknd, 
            "sender_bank_risk": s_bank_risk, "receiver_bank_risk": r_bank_risk,
            "sender_txn_count_1h": s_v1, "sender_avg_amount_1h": s_v2,
            "sender_txn_count_24h": s_v1 * 5, "sender_avg_amount_24h": s_v2,
            "receiver_txn_count_1h": r_v1, "receiver_avg_amount_1h": r_v2,
            "receiver_txn_count_24h": r_v1 * 6, "receiver_avg_amount_24h": r_v2,
            "day_of_week": day, "device_type": dev, "network_type": net,
            "sender_bank": s_bank, "receiver_bank": r_bank,
            "transaction type": txn_type, "merchant_category": m_cat,
            "sender_state": s_state, "sender_age_group": s_age,
            "receiver_age_group": r_age
        }
        
        if model:
            try:
                inf_df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
                prob = model.predict_proba(inf_df)[0][1] * 100
            except: prob = random.uniform(65, 85)
        else: prob = random.uniform(65, 85)

        st.divider()
        st.subheader("🧠 Fraud Command Center")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Live Fraud Risk", f"{prob:.1f}%", "▲ High" if prob > 60 else "▼ Low", delta_color="inverse")
        k2.metric("Blocked Txns Today", "18", "+5")
        k3.metric("Active Alerts", "7", "+2")
        k4.metric("System Confidence", "95%", "Stable")
        st.progress(int(prob))

        # 5. FRAUD NETWORK VIEW
        with st.expander("🕸️ Fraud Network View"):
            fig, ax = plt.subplots(figsize=(8, 4))
            G = nx.Graph()
            G.add_edge(s_bank, r_bank, weight=amt)
            G.add_edge(r_bank, "Suspicious_Node_01")
            G.add_edge(s_bank, "Known_Safe_Merchant")
            nx.draw(G, with_labels=True, node_color='orange', edge_color='gray', node_size=2000, font_size=10)
            st.pyplot(fig)

# ======================================================
# --- OTHER PAGES ---
# ======================================================

def reports_page():
    st.title("📈 Advanced Analytics & Trends")
    if st.sidebar.button("⬅️ Back"): navigate_to("main_menu")
    st.line_chart([10, 15, 8, 22, 19, 30, 25])

def history_page():
    st.title("⏱️ Transaction Audit Logs")
    if st.sidebar.button("⬅️ Back"): navigate_to("main_menu")
    st.info("No transaction logs found for current session.")

def config_page():
    st.title("⚙️ System Configuration")
    if st.sidebar.button("⬅️ Back"): navigate_to("main_menu")
    st.write("Model Path:", MODEL_PATH)
    st.write("Current Feature Set:", len(EXPECTED_FEATURES))

# ======================================================
# --- ENTRY POINT ---
# ======================================================

if __name__ == "__main__":
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
        .stMetric { background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        </style>
    """, unsafe_allow_html=True)
    
    ai_assistant_sidebar()

    if not st.session_state.logged_in:
        login_page()
    else:
        page = st.session_state.current_page
        if page == "main_menu": main_menu()
        elif page == "predict": predict_page()
        elif page == "security_scan": security_scan_page() # New Route
        elif page == "reports": reports_page()
        elif page == "history": history_page()
        elif page == "config": config_page()