import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

lgb_model = joblib.load("models/lgb_model.pkl")
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgb_model.json")  
# xgb_model = joblib.load("models/xgb_model.pkl")


def evaluate_business_metrics(y_true, y_pred, cost=20):
    """
    Business-focused metrics:
    - Revenue = demand * price
    - Profit = demand * (price - cost)
    """
    demand = np.maximum(y_true - 0.5 * (y_pred - np.mean(y_pred)), 0)
    revenue = np.sum(demand * y_pred)
    profit = np.sum(demand * (y_pred - cost))
    return float(revenue), float(profit)


def predict_price(input_data: dict):
    df = pd.DataFrame([input_data])
    
    # Feature engineering for input
    df['moving_avg_demand'] = df['units_sold']  # simple placeholder
    df['price_elasticity'] = 0  # placeholder
    df['trend_factor'] = df['units_sold']  # placeholder

    features = ['units_sold', 'competitor_price', 'stock_level', 'day_of_week', 'holiday_flag',
                'views', 'moving_avg_demand', 'price_elasticity', 'trend_factor']
    
  
    price_lgb = lgb_model.predict(df[features])[0]
    # price_xgb = xgb_model.predict(df[features])[0]
    price_xgb = xgb_model.predict(xgb.DMatrix(df[features]))[0]

    revenue, profit=evaluate_business_metrics(
        y_true=np.array([df['units_sold'][0]]),
        y_pred=np.array([price_lgb])
    )
    
    return {
        "lightgbm_price": float(price_lgb), 
        "xgboost_price": float(price_xgb),
        "estimate_revenue":revenue,
        "estimated_profit": profit
        }
