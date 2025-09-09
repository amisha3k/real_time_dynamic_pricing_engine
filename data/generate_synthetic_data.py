import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Parameters
num_products = 500
date_range = pd.date_range(start='2024-01-01', end='2024-12-31')

data_list = []

np.random.seed(42)
for product_id in range(1, num_products + 1):
    base_price = np.random.randint(50, 500)
    for date in date_range:
        units_sold = max(0, int(np.random.normal(20, 5) - 0.05*base_price))
        competitor_price = base_price + np.random.randint(-20, 20)
        stock_level = np.random.randint(0, 100)
        day_of_week = date.weekday()
        holiday_flag = 1 if date in pd.to_datetime(['2024-01-26', '2024-08-15', '2024-10-02']) else 0
        views = max(0, int(units_sold*5 + np.random.normal(0, 10)))
        data_list.append([product_id, date, base_price, units_sold, competitor_price,
                          stock_level, day_of_week, holiday_flag, views])

# Create DataFrame
df = pd.DataFrame(data_list, columns=['product_id', 'date', 'historical_price', 'units_sold',
                                      'competitor_price', 'stock_level', 'day_of_week',
                                      'holiday_flag', 'views'])

# Save to CSV
df.to_csv("data/sales_data.csv", index=False)
print("Synthetic dataset created at data/sales_data.csv")
