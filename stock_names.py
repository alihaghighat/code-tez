import os
import pandas as pd
import random

# مسیر پوشه‌ای که فایل‌های CSV سهام‌ها در آن قرار دارند
directory_path = './100stocks'

# دیکشنری برای نگهداری دیتافریم‌های سهام‌ها
dataframes = {}

# خواندن تمام فایل‌های CSV سهام‌ها
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        stock_name = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dataframes[stock_name] = df

# فهرست نام سهام‌ها
stock_names = list(dataframes.keys())

# تنظیم تصادفی برای سهام‌ها
random.shuffle(stock_names)

# تقسیم سهام‌ها به دو دسته
half = len(stock_names) // 2
stocks_server_1 = stock_names[:half]  # سهام‌های سرور اول
stocks_server_2 = stock_names[half:]  # سهام‌های سرور دوم

# ذخیره اسامی سهام‌های سرور 1 در فایل CSV
df_server_1 = pd.DataFrame(stocks_server_1, columns=['Stock'])
df_server_1.to_csv('stocks_server_1.csv', index=False)

# ذخیره اسامی سهام‌های سرور 2 در فایل CSV
df_server_2 = pd.DataFrame(stocks_server_2, columns=['Stock'])
df_server_2.to_csv('stocks_server_2.csv', index=False)

print(f"Saved stocks for Server 1 and Server 2 into 'stocks_server_1.csv' and 'stocks_server_2.csv' respectively.")
