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
        return -1,None

    # Ensure both dataframes have the same length after filtering
    min_length = min(len(stock_data_1_filtered), len(stock_data_2_filtered))
    stock_data_1_filtered = stock_data_1_filtered.iloc[:min_length]
    stock_data_2_filtered = stock_data_2_filtered.iloc[:min_length]
    
    
    if lag > 0:
        # Check if both filtered datasets have enough data for the window and lag
        if len(stock_data_1_filtered) < window_size + lag or len(stock_data_2_filtered) < window_size:
            return -1,None
        
        # Extract windows with dates included
        window_1_with_dates = stock_data_1_filtered[['Date', 'Adj Close']].iloc[-window_size - lag:-lag].reset_index(drop=True)
        window_2_with_dates = stock_data_2_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        
        # # Print windows with dates
        # print("Window 1 with dates:\n", window_1_with_dates)
        # print("Window 2 with dates:\n", window_2_with_dates)
        
        # Remove dates for correlation calculation
        window_1 = window_1_with_dates['Adj Close']
        window_2 = window_2_with_dates['Adj Close']

    elif lag < 0:
        # Check if both filtered datasets have enough data for the window and lag
        if len(stock_data_2_filtered) < window_size + abs(lag) or len(stock_data_1_filtered) < window_size:
            return -1,None

        # Extract windows with dates included
        window_1_with_dates = stock_data_1_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        window_2_with_dates = stock_data_2_filtered[['Date', 'Adj Close']].iloc[-window_size - abs(lag):-abs(lag)].reset_index(drop=True)
        
        # # Print windows with dates
        # print("Window 1 with dates:\n", window_1_with_dates)
        # print("Window 2 with dates:\n", window_2_with_dates)
        
        # Remove dates for correlation calculation
        window_1 = window_1_with_dates['Adj Close']
        window_2 = window_2_with_dates['Adj Close']

    else:
        # No lag, use data as is
        if len(stock_data_1_filtered) < window_size or len(stock_data_2_filtered) < window_size:
             return -1,None

        # Extract windows with dates included
        window_1_with_dates = stock_data_1_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        window_2_with_dates = stock_data_2_filtered[['Date', 'Adj Close']].iloc[-window_size:].reset_index(drop=True)
        
        # # Print windows with dates
        # print("Window 1 with dates:\n", window_1_with_dates)
        # print("Window 2 with dates:\n", window_2_with_dates)
        
        # Remove dates for correlation calculation
        window_1 = window_1_with_dates['Adj Close']
        window_2 = window_2_with_dates['Adj Close']

    # Check if the windows are valid and have the same length
    if len(window_1) != len(window_2):
        return -1,None

    # Calculate Pearson correlation
    correlation = pearson_correlation(window_1, window_2)

    return correlation,window_2_with_dates
def create_time_based_sliding_window(dataFrame, n_steps, stock_name, date_step='1D', n_top=3, delay=1, start_date='2018-01-01'):
    # Copy the specific stock data and resample
    df = dc(dataFrame[stock_name])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Adj Close']]
    df.set_index('Date', inplace=True)
    df = df.resample(date_step).last().dropna()

    # Create shifted columns (time windows) for the target stock
    shifted_columns = [df['Adj Close'].shift(i).rename(f'Adj Close(t-{i})') for i in range(1, n_steps + 1)]
    df = pd.concat([df] + shifted_columns, axis=1).dropna()

    # Prepare the other stocks data in advance
    other_stocks_data = {col: dataFrame[col].copy() for col in dataFrame.keys() if col != stock_name}
    for stock, stock_df in other_stocks_data.items():
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        other_stocks_data[stock] = stock_df

    # Convert start_date to datetime and filter target stock data
    start_date = pd.to_datetime(start_date)
    df = df[df.index >= start_date]

    # Create a dictionary to store new columns for each row
    all_new_columns = {}

    # Iterate over rows from the start_date onward
    for row_index in df.index:
        print(f"Processing row: {row_index}")
        correlations = []
        
        # Compare each stock with the target stock
        for other_stock, other_stock_data in other_stocks_data.items():
            date_str = row_index.strftime('%Y-%m-%d')
            
            # Calculate correlation
            correlation, window_2_with_dates = calculate_lagged_correlation(
                dataFrame[stock_name],
                other_stock_data,
                date_str,
                n_steps,
                delay
            )
            
            correlations.append((other_stock, correlation, window_2_with_dates))

        # Sort correlations and keep the top n
        correlations = [(stock, corr, window_2) for stock, corr, window_2 in correlations if corr > 0]
        if correlations:
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            top_correlated_stocks = correlations[:n_top]
            
            # Create a dictionary to store new columns for this row
            new_columns = {f'Top_{i+1} Close(t-{j+1})': value
                           for i, (_, _, window_data) in enumerate(top_correlated_stocks)
                           for j, value in enumerate(window_data['Adj Close'].to_numpy()[::-1])}
            
            # Store the new columns for the current row in the global dictionary
            all_new_columns[row_index] = new_columns
        
    # After processing all rows, create a DataFrame from the dictionary
    new_columns_df = pd.DataFrame.from_dict(all_new_columns, orient='index')
     
    # Concatenate the original DataFrame with the new columns at once
    df = pd.concat([df, new_columns_df], axis=1)

    return df





# Prepare Data
lookback = 180
shifted_aapl_df = create_time_based_sliding_window(dataframes, lookback, 'AAPL', '1D',3,-2,start_date='2018-01-01')
print(shifted_aapl_df)
