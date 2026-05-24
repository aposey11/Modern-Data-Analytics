import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Trains the GTRI LightGBM Tweedie regressor on the master training dataset produced
# by GTRI_feature_engineering.py. The Tweedie objective with variance power 1.5 is
# appropriate for accident count data: sparse, zero-inflated, and strictly non-negative.
#
# The chronological split (train: 2019-2023, test: 2024) mirrors real deployment
# conditions — the model is evaluated on data it could not have seen during training.
#
# Output: GTRI_model_artifacts.pkl containing the trained model, the dynamic baseline
# risk value, the feature name list, and evaluation metrics. The baseline risk is the
# mean predicted risk across the full dataset and is used by the dashboard to express
# all site scores as multiples of the average (relative risk).

print("Loading the master training dataset...")
df = pd.read_parquet("gtri_master_training_data.parquet")

# ==========================================
# 1. DEFINE FEATURES AND TARGET
# ==========================================
features = [
    # Temporal (cyclical encoding — both sin and cos required per pair)
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
    # Traffic
    'Avg_Traffic_Volume', 'Avg_Traffic_Max_Spike', 'Avg_Traffic_Volatility',
    'Traffic_Weekend_Ratio',
    'Traffic_Volume_Delta_Mean',    # mean traffic change into this hour (positive = building)
    # Weather
    'Avg_Temperature', 'Avg_Precipitation', 'Avg_Wind_Speed', 'Avg_Temp_Drop', 'Avg_Wind_Spike',
    'Rain_Mean_Intensity', 'Rain_Surge_Ratio',
    'Rain_Lag1h_Mean', 'Rain_Lag1h_Max',  # genuine previous-hour rain level
    'Rain_Surge',                          # log-difference onset signal (see feature engineering)
    # Spatial
    'spatial_risk_score', 'historical_accidents_250m', 'historical_accidents_250m_to_500m'
]

target = 'Accident_Count'

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}. "
        f"Run GTRI_feature_engineering.py first."
    )

# ==========================================
# 2. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================
print("Splitting data chronologically (train: 2019-2023 | test: 2024)...")
train_df = df[df['Year'] < 2024]
test_df  = df[df['Year'] == 2024]

X_train, y_train = train_df[features], train_df[target]
X_test,  y_test  = test_df[features],  test_df[target]

print(f"Train: {len(X_train):,} rows  |  accident hours: {(y_train > 0).sum():,}")
print(f"Test:  {len(X_test):,} rows   |  accident hours: {(y_test  > 0).sum():,}")

# ==========================================
# 3. TRAIN TWEEDIE REGRESSOR
# ==========================================
print("\nTraining LightGBM Tweedie regressor...")
model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,   # standard choice for zero-inflated count data
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=15,
    min_child_samples=50,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# ==========================================
# 4. EVALUATION METRICS
# ==========================================
print("\n--- Model evaluation (2024 test set) ---")

y_pred         = model.predict(X_test)
y_pred_clipped = np.maximum(y_pred, 1e-9)  # guard against log(0) in deviance formula

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
spearman_r, spearman_p = stats.spearmanr(y_test, y_pred)

# Mean Tweedie deviance with p=1.5 is the proper scoring rule for this objective.
# It penalises systematic bias in the predicted risk ordering more directly than MAE.
p = 1.5
tweedie_dev = np.mean(
    2 * (
        (y_test.values ** (2 - p)) / ((1 - p) * (2 - p)) -
        (y_test.values * y_pred_clipped ** (1 - p)) / (1 - p) +
        (y_pred_clipped ** (2 - p)) / (2 - p)
    )
)

print(f"  MAE             : {mae:.6f}")
print(f"  RMSE            : {rmse:.6f}")
print(f"  Spearman rho    : {spearman_r:.4f}  (p={spearman_p:.2e})")
print(f"  Tweedie deviance: {tweedie_dev:.4f}  (primary metric; lower is better)")

# Decile lift table: the model is useful if the highest-risk decile captures a
# disproportionate share of actual accidents.
print("\nDecile lift table:")
eval_df = pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred})
eval_df['decile'] = pd.qcut(eval_df['y_pred'], q=10, labels=False, duplicates='drop')
lift_table = eval_df.groupby('decile').agg(
    avg_predicted=('y_pred', 'mean'),
    avg_actual=('y_true', 'mean'),
    total_accidents=('y_true', 'sum'),
    n_rows=('y_true', 'count')
).reset_index()
lift_table['pct_of_accidents'] = (
    lift_table['total_accidents'] / lift_table['total_accidents'].sum() * 100
).round(1)
print(lift_table.to_string(index=False))

# Feature importance
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

# ==========================================
# 5. SAVE ARTIFACTS
# ==========================================
# Baseline risk is the mean predicted value across the full dataset (train + test).
# Using all data gives a stable reference for normalising site scores in the dashboard.
y_pred_all    = model.predict(df[features])
baseline_risk = float(y_pred_all.mean())

print(f"\nComputed baseline_risk = {baseline_risk:.6f}")

top_decile_pct = float(
    lift_table.loc[lift_table['decile'] == lift_table['decile'].max(), 'pct_of_accidents'].values[0]
)

artifacts = {
    'model': model,
    'baseline_risk': baseline_risk,
    'feature_names': features,
    'training_period': '2019-2023',
    'test_period': '2024',
    'metrics': {
        'rmse': rmse,
        'spearman_rho': spearman_r,
        'tweedie_deviance': tweedie_dev,
        'top_decile_pct': top_decile_pct,
    }
}

artifact_file = "GTRI_model_artifacts.pkl"
joblib.dump(artifacts, artifact_file)

importance_file = "GTRI_feature_importance.csv"
importance_df.to_csv(importance_file, index=False)

print(f"\nModel artifacts saved to : {artifact_file}")
print(f"Feature importance saved to: {importance_file}")
print(f"Baseline risk: {baseline_risk:.6f}")
