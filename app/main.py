from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import predict_price

app = FastAPI(title="Dynamic Pricing Engine")

class ProductData(BaseModel):
    product_id: int
    units_sold: int
    competitor_price: float
    stock_level: int
    day_of_week: int
    holiday_flag: int
    views: int

@app.post("/predict-price")
def get_price(data: ProductData):
    return predict_price(data.dict())
