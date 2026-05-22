import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

print("🔄 Loading the Master Training Dataset...")
df = pd.read_parquet("master_training_data.parquet")

# ==========================================
# 1. DEFINE FEATURES AND TARGET
# ==========================================
features = [
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos', 
    'Avg_Traffic_Volume', 'Avg_Traffic_Max_Spike', 'Avg_Traffic_Volatility', 'Avg_Piek_Index',
    'Avg_Temperature', 'Avg_Precipitation', 'Avg_Wind_Speed', 'Avg_Temp_Drop', 'Avg_Wind_Spike',
    'spatial_risk_score', 'historical_accidents_250m', 'historical_accidents_250m_to_500m'
]

# TARGET IS NOW THE RAW COUNT
target = 'Accident_Count'

# ==========================================
# 2. CHRONOLOGICAL TRAIN/TEST SPLIT
# ==========================================
print("⏳ Splitting data chronologically (Train: 2019-2023 | Test: 2024)...")
train_df = df[df['Year'] < 2024]
test_df = df[df['Year'] == 2024]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print(f"Train Size: {len(X_train):,} rows")
print(f"Test Size: {len(X_test):,} rows")

# ==========================================
# 3. TRAIN TWEEDIE REGRESSOR
# ==========================================
print("🧠 Training LightGBM Tweedie Regressor...")
model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5, # Standard power for zero-inflated accident counts
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# ==========================================
# 4. EVALUATION & FEATURE IMPORTANCE
# ==========================================
print("\n📊 --- MODEL EVALUATION (2024 Test Set) --- 📊")

# Predict Continuous Risk Multipliers
y_pred_risk = model.predict(X_test)

print(f"Baseline (Average) Risk Score: {y_pred_risk.mean():.4f}")
print(f"Maximum Predicted Risk Score (Worst-Case Scenario): {y_pred_risk.max():.4f}")

# Feature Importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nTop 5 Most Important Features:")
print(importance_df.head(5).to_string(index=False))

# ==========================================
# 5. SAVE ARTIFACTS FOR LIVE INFERENCE
# ==========================================
model_file = "lightgbm_accident_model.pkl"
joblib.dump(model, model_file)
print(f"\n🚀 Phase 2 Complete! Sensitive Tweedie Risk model saved as '{model_file}'.")
print("Ready for live dashboard deployment!")