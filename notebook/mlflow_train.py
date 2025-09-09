import mlflow
import mlflow.lightgbm

# Start experiment
mlflow.set_experiment("dynamic_pricing")

with mlflow.start_run():
    lgb_model.fit(X_train, y_train)
    
    # Log model
    mlflow.lightgbm.log_model(lgb_model, artifact_path="lgb_model")
    
    # Log metrics
    mlflow.log_metric("mae", mae_value)
    mlflow.log_metric("rmse", rmse_value)
