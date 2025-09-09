import pandas as pd
import joblib

# Load trained models
lgb_model = joblib.load("../models/lgb_model.pkl")
xgb_model = joblib.load("../models/xgb_model.pkl")

def predict_price(input_data: dict):
    df = pd.DataFrame([input_data])
    
    # Feature engineering for input
    df['moving_avg_demand'] = df['units_sold']  # simple placeholder
    df['price_elasticity'] = 0  # placeholder
    df['trend_factor'] = df['units_sold']  # placeholder

    features = ['units_sold', 'competitor_price', 'stock_level', 'day_of_week', 'holiday_flag',
                'views', 'moving_avg_demand', 'price_elasticity', 'trend_factor']
    
    price_lgb = lgb_model.predict(df[features])[0]
    price_xgb = xgb_model.predict(df[features])[0]
    
    return {"lightgbm_price": float(price_lgb), "xgboost_price": float(price_xgb)}
