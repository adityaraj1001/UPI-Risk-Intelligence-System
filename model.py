import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ==================================================
# DATA & PATH SETUP
# ==================================================
DATA_PATH = "upi_transactions_2024.csv" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# LOAD & PREPARE DATA
# ==================================================
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Dataset not found at: {DATA_PATH}")
    raise

# Clean and rename columns
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={"amount (inr)": "amount"})

# ==================================================
# --- ADVANCED FEATURE ENGINEERING ---
# ==================================================
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').set_index('timestamp')

# 1. Base Risks & Time Features
df['is_weekend'] = df['is_weekend'].astype(int)
df['sender_bank_risk'] = 0.5 
df['receiver_bank_risk'] = 0.5
df['is_night_txn'] = df['hour_of_day'].apply(lambda x: 1 if x <= 5 or x >= 23 else 0) # NEW FEATURE

# 2. Velocity & Ratio Features (Spike Detection)
def create_advanced_velocity(df, entity_col, time_window, prefix):
    # Rolling Count (Velocity)
    df[f'{prefix}_txn_count_{time_window}h'] = df.groupby(entity_col)['amount'].transform(
        lambda x: x.rolling(f'{time_window}h', closed='left').count()
    ).fillna(0)
    
    # Rolling Mean (History)
    df[f'{prefix}_avg_amount_{time_window}h'] = df.groupby(entity_col)['amount'].transform(
        lambda x: x.rolling(f'{time_window}h', closed='left').mean()
    ).fillna(0)
    
    # Spike Detection Ratio: Current Amount / Avg History [NEW FEATURES]
    df[f'{prefix}_amt_ratio_{time_window}h'] = df['amount'] / (df[f'{prefix}_avg_amount_{time_window}h'] + 1)
    return df

df = create_advanced_velocity(df, 'sender_bank', 1, 'sender')
df = create_advanced_velocity(df, 'sender_bank', 24, 'sender')
df = create_advanced_velocity(df, 'receiver_bank', 1, 'receiver')
df = create_advanced_velocity(df, 'receiver_bank', 24, 'receiver')

# 3. Categorical Risk Scoring (NEW FEATURES)
high_risk_states = ['Delhi', 'Haryana', 'Jharkhand'] # Example regions with high cyber-crime alerts
df['is_high_risk_zone'] = df['sender_state'].apply(lambda x: 1 if x in high_risk_states else 0)

# Reset index for selection
df = df.reset_index(drop=False)

# ==================================================
# TARGET & FEATURE SELECTION (30 FEATURES TOTAL)
# ==================================================
y = df['fraud_flag']

numeric_features = [
    "amount", "hour_of_day", "is_weekend", "sender_bank_risk", "receiver_bank_risk",
    "is_night_txn", "is_high_risk_zone",
    "sender_txn_count_1h", "sender_avg_amount_1h", "sender_amt_ratio_1h",
    "sender_txn_count_24h", "sender_avg_amount_24h", "sender_amt_ratio_24h",
    "receiver_txn_count_1h", "receiver_avg_amount_1h", "receiver_amt_ratio_1h",
    "receiver_txn_count_24h", "receiver_avg_amount_24h", "receiver_amt_ratio_24h",
]

categorical_features = [
    "day_of_week", "device_type", "network_type", "sender_bank", "receiver_bank",
    "transaction type", "merchant_category", "sender_state", "sender_age_group", "receiver_age_group",
    "transaction_status" # Added status as a potential signal
]

X = df[numeric_features + categorical_features]

# ==================================================
# PREPROCESSING & MODEL PIPELINE
# ==================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, 
            max_depth=20, 
            class_weight="balanced", 
            random_state=42, 
            n_jobs=-1))
    ]
)

# ==================================================
# TRAIN & SAVE
# ==================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🧠 Training AI with 30 parameters...")
model.fit(X_train, y_train)
print("✅ Training Complete.")

joblib.dump(model, MODEL_PATH)
print(f"📁 Brain deployed to: {MODEL_PATH}")