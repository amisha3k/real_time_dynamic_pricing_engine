import requests

url = "http://127.0.0.1:8000/predict-price"

data = {
    "product_id": 101,
    "units_sold": 50,
    "competitor_price": 20.5,
    "stock_level": 200,
    "day_of_week": 2,
    "holiday_flag": 0,
    "views": 500
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
