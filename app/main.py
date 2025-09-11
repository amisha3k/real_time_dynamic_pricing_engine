
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from app.utils import predict_price

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dynamic-pricing")

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
async def post_predict_price(data: ProductData, request: Request):
    logger.info("POST /predict-price called from %s", request.client)
    logger.info("Payload: %s", data.dict())
    return predict_price(data.dict())


@app.get("/predict-price")
async def get_predict_price(
    product_id: int,
    units_sold: int,
    competitor_price: float,
    stock_level: int,
    day_of_week: int,
    holiday_flag: int,
    views: int,
    request: Request = None
):
    payload = {
        "product_id": product_id,
        "units_sold": units_sold,
        "competitor_price": competitor_price,
        "stock_level": stock_level,
        "day_of_week": day_of_week,
        "holiday_flag": holiday_flag,
        "views": views
    }
    logger.info("GET /predict-price called from %s", request.client if request else "unknown")
    logger.info("Query payload: %s", payload)
    return predict_price(payload)

