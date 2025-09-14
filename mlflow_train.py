# mlflow_log_models.py
import joblib
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# ---------------------------
# Load Test Data
# ---------------------------
X_test = joblib.load("notebook/df/X_test.pkl")
y_test = joblib.load("notebook/df/y_test.pkl")

# ---------------------------
# Load Models
# ---------------------------
lgb_model = joblib.load("models/lgb_model.pkl")       # LightGBM
print(type(lgb_model))
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model.json")        # XGBoost

# ---------------------------
# Setup MLflow
# ---------------------------
mlflow.set_experiment("dynamic_pricing_models")

with mlflow.start_run():

    # ---------------------------
    # LightGBM Evaluation
    # ---------------------------
    y_pred_lgb = lgb_model.predict(X_test)
    lgb_mae = mean_absolute_error(y_test, y_pred_lgb)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    lgb_r2 = r2_score(y_test, y_pred_lgb)

    # Log LightGBM model & metrics
    mlflow.lightgbm.log_model(lgb_model, artifact_path="lgb_model")
    mlflow.log_metric("lgb_mae", lgb_mae)
    mlflow.log_metric("lgb_rmse", lgb_rmse)
    mlflow.log_metric("lgb_r2", lgb_r2)

    print(f"✅ LightGBM - MAE: {lgb_mae:.4f}, RMSE: {lgb_rmse:.4f}, R²: {lgb_r2:.4f}")

    # ---------------------------
    # XGBoost Evaluation
    # ---------------------------
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred_xgb = xgb_model.predict(dtest)
    xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_r2 = r2_score(y_test, y_pred_xgb)

    # Log XGBoost model & metrics
    mlflow.xgboost.log_model(xgb_model, artifact_path="xgb_model")
    mlflow.log_metric("xgb_mae", xgb_mae)
    mlflow.log_metric("xgb_rmse", xgb_rmse)
    mlflow.log_metric("xgb_r2", xgb_r2)

    print(f"✅ XGBoost - MAE: {xgb_mae:.4f}, RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")

    print("🎉 All models and metrics logged to MLflow successfully!")
