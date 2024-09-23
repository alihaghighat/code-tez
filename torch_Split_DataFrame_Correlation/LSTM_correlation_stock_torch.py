import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from copy import deepcopy as dc
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# تنظیم دانه تصادفی
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # می‌توانید مقدار دانه را تغییر دهید

# Data loading
directory_path = '../100stocks'
dataframes = {}

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        stock_name = filename.split('.')[0]
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        dataframes[stock_name] = df

def pearson_correlation(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    deviation_product = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / len(x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / len(y)) ** 0.5
    correlation = deviation_product / (len(x) * std_x * std_y)
    return correlation
def calculate_lagged_correlation(stock_data_1, stock_data_2, start_date, window_size, lag):
    # Filter data starting from the specified date
    stock_data_1_filtered = stock_data_1[stock_data_1['Date'] <= start_date].reset_index(drop=True)
    stock_data_2_filtered = stock_data_2[stock_data_2['Date'] <= start_date].reset_index(drop=True)

    if len(stock_data_1_filtered) == 0 or len(stock_data_2_filtered) == 0:
        return -1

    # Ensure both dataframes have the same length after filtering
    min_length = min(len(stock_data_1_filtered), len(stock_data_2_filtered))
    stock_data_1_filtered = stock_data_1_filtered.iloc[:min_length]
    stock_data_2_filtered = stock_data_2_filtered.iloc[:min_length]
    
    # Apply lag (shift one stock's data by the lag amount) without reducing the number of records
    if lag > 0:
        # Shift stock_data_1 forward by the lag amount
        if len(stock_data_1_filtered) < window_size + lag:
            return -1  # Not enough data for the window and lag
        window_1 = stock_data_1_filtered['Adj Close'].iloc[-window_size - lag:-lag].reset_index(drop=True)
        window_2 = stock_data_2_filtered['Adj Close'].iloc[-window_size:].reset_index(drop=True)
        
    elif lag < 0:
        # Shift stock_data_2 forward by the lag amount
        if len(stock_data_2_filtered) < window_size + abs(lag):
            return -1  # Not enough data for the window and lag
        window_1 = stock_data_1_filtered['Adj Close'].iloc[-window_size:].reset_index(drop=True)
        window_2 = stock_data_2_filtered['Adj Close'].iloc[-window_size - abs(lag):-abs(lag)].reset_index(drop=True)
        
    else:
        # No lag, both windows are taken as is
        if len(stock_data_1_filtered) < window_size or len(stock_data_2_filtered) < window_size:
            return -1  # Not enough data for the window size
        window_1 = stock_data_1_filtered['Adj Close'].iloc[-window_size:].reset_index(drop=True)
        window_2 = stock_data_2_filtered['Adj Close'].iloc[-window_size:].reset_index(drop=True)

    # Check if the windows are valid and have the same length
    if len(window_1) != len(window_2):
        return -1

    # Calculate Pearson correlation
    correlation = pearson_correlation(window_1, window_2)

    return correlation

# Sliding Window Creation

def create_time_based_sliding_window(dataFrame, n_steps, stock_name, date_step='1D', n_top=3, delay=1):
    df = dc(dataFrame[stock_name])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Adj Close']]
    df.set_index('Date', inplace=True)
    df = df.resample(date_step).last().dropna()
    window_size=n_steps
    other_stocks = [col for col in dataFrame.keys()if col != stock_name]
    # ایجاد پنجره‌های زمانی
    shifted_columns = [df['Adj Close'].shift(i).rename(f'Adj Close(t-{i})') for i in range(1, n_steps + 1)]
    df = pd.concat([df] + shifted_columns, axis=1)
    df.dropna(inplace=True)
    stock_data = dataFrame[stock_name].copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    # برای هر ردیف df همبستگی محاسبه می‌شود
    for row_index in df.index:
        correlations = []
        for other_stock in other_stocks:
            # داده‌های سهام مقایسه‌ای
            other_stock_data = dataFrame[other_stock].copy()
            other_stock_data['Date'] = pd.to_datetime(other_stock_data['Date'])  
            date_str = row_index.strftime('%Y-%m-%d')
           
            # محاسبه همبستگی
            correlation = calculate_lagged_correlation(
                stock_data,
                other_stock_data,
                date_str,
                window_size,
                delay
            )
          
            correlations.append((other_stock, correlation))

            # مرتب‌سازی همبستگی‌ها و نمایش n مورد با بیشترین مقدار
        
        correlations = [(stock, corr) for stock, corr in correlations if corr > 0]
        if correlations:
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"For row {row_index}: Top {n_top} correlated stocks: {correlations[:n_top]}")
        else:
            print(f"For row {row_index}: No valid correlations.")
        
       
    
    
    return df

# Prepare Data
lookback = 180
shifted_aapl_df = create_time_based_sliding_window(dataframes, lookback, 'AAPL', '1D',3,-2)
print(shifted_aapl_df)
